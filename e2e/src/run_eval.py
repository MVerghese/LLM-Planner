"""
ALFRED Evaluation Script with LLM Planning
------------------------------------------
Main evaluation framework that tests language model-based planning in AI2-THOR environments.
Connects LLMs with the simulation to perform household tasks, supporting dynamic replanning,
vision integration, and comprehensive metrics collection. Serves as the primary entry point
for running experiments on the ALFRED benchmark.
"""

# ALL in one script to run LLM-Planner on ALFRED tasks

import os
import base64
import sys
import json
import yaml
import cv2
import argparse
import time
import datetime
from tqdm import tqdm
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging
import textwrap
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
from utils import encode_image

from alfred.thor_connector import ThorConnector
from alfred.utils import dotdict, load_task_json
from alfred.data.preprocess import Dataset
from hlp_planner import LLM_Planner, Tree_Search_LLM_Planner, Entropy_reduction_LLM_Planner, GT_Planner
import threading
from multiprocessing.pool import ThreadPool

PLANNERS = {
    "llm": LLM_Planner,
    "tree_search": Tree_Search_LLM_Planner,
    "entropy_reduction": Entropy_reduction_LLM_Planner,
    "gt": GT_Planner
}

sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

# Configure the root logger to print to console
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Get a logger for this module
log = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

class AlfredEvaluator:
    def __init__(self, config_file):
        # Load configuration
        with open(config_file) as reader:
            self.config = yaml.safe_load(reader)
        
        # # Initialize OpenAI client
        # self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        # Initialize VLLM client
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        
        # Initialize LLM planner
        self.llm_planner = PLANNERS[self.config["llm_planner"]["planner"]](  # Use the planner name from config
            llm_config=self.config["llm_planner"],
            client=self.client,
            logger=log
        )
        # Initialize environment
        num_envs = self.config.get("num_workers", 1)
        if num_envs > 1:
            self.envs = [ThorConnector(x_display=self.config["alfred"]["x_display"]) for _ in range(num_envs)]
            self.env_lock = threading.Lock()
            self.envs_in_use = [False] * num_envs
        else:
            self.env = ThorConnector(x_display=self.config["alfred"]["x_display"])
        
        log.debug(f"Initialized {'multiple environments' if num_envs > 1 else 'single environment'} for evaluation")
        
        # Load task splits
        with open(self.config["alfred"]["splits"]) as f:
            self.splits = json.load(f)
            
        # Prepare tasks
        self.tasks = self._prepare_tasks()
        
        # Initialize the sentence transformer for object name matching
        self.obj_encoder = SentenceTransformer(self.config["llm_planner"]["emb_model_name"], device="cuda:1")
        self.obj_sim_threshold = self.config["llm_planner"].get("obj_sim_threshold", 0.8)  # Default similarity threshold
        self.knn_set = pd.read_pickle(self.config["llm_planner"]["knn_dataset_path"])

    def _prepare_tasks(self):
        """Prepare tasks for evaluation"""
        assert self.config["alfred"]["eval_set"] in self.splits.keys()
        tasks = []
        
        # exclude two obj task
        for e in self.splits[self.config["alfred"]["eval_set"]]:
            if 'pick_two_obj_and_place' not in e['task']:
                tasks.append(e)
                
        # Debug mode
        if self.config.get("debug", False):
            for task in tasks:
                if 'trial_T20190909_061130_844814' in task['task']: #NOTE Change this to the task you want to debug
                    new_task = [task]
                    break
            tasks = new_task
            
        return tasks
    
    def llm(self, prompt, engine, images=None, stop=["\n"]):
        """
        Interface to LLM models
        This function is deprecated with the shift of LLM processing to the planner
        """

        if engine == 'gpt-4o-mini' or engine == 'gpt-4o':
            # Create the base message content
            message_content = []
            
            # Add image if provided
            if images and len(images) > 0:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{images[0]}"
                    }
                })
            
            # Add text prompt
            message_content.append({
                "type": "text",
                "text": prompt,
            })
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=engine,
                messages=[
                    {
                        "role": "user",
                        "content": message_content,
                    }
                ],
                max_tokens=300,
                temperature=0.0
            )
            return response.choices[0].message.content
        elif engine == 'Qwen/Qwen3-VL-8B-Instruct' or engine == 'Qwen/Qwen3-VL-30B-A3B-Instruct-FP8':
             # Create the base message content
            message_content = []
            

            # Add image if provided
            if images and len(images) > 0:
                for image in images:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    })
            
            # Add text prompt
            message_content.append({
                "type": "text",
                "text": prompt,
            })
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=engine,
                messages=[
                    {
                        "role": "user",
                        "content": message_content,
                    }
                ],
                max_tokens=300,
                temperature=0.0
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"{engine} is not supported!")



    def evaluate_task(self, engine, traj_data, r_idx, dynamic=False, to_print=True, vision=False, ob='', env=None):
        """Evaluate a single task"""

        # This handles using multiple environments for multiprocessing
        if env is None:
            env = self.env

        # Initialize frame history and plan tracking
        frame_history = []
        completed_plans = []
        failed_plans = []
        dynamic_plans = []
        
        # Setup scene
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = f'FloorPlan{scene_num}'
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # Initialize and save initial frame
        env.step(dict(traj_data['scene']['init_action']))
        frame_history.append(env.last_event.frame.copy())
        env.set_task(traj_data, dotdict(self.config["alfred"]["env_args"]), reward_type='dense')

        # Get task instructions
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        
        # Always get step instructions, but control their usage with includeLow flag
        step_instrs = [ann["high_descs"] for ann in traj_data["turk_annotations"]["anns"]]
        
        # Get configuration for using step instructions
        use_step_instructions = self.config["llm_planner"].get("use_step_instructions", True)
        
        log.debug(f"Task: {goal_instr}")
        if use_step_instructions:
            log.debug("Step instructions: Enabled")
        else:
            log.debug("Step instructions: Disabled")

        # Setup for evaluation
        done, success = False, False
        reward = 0

        # Get initial frames if vision model is used
        encoded_frames = []
        if vision:
            init_frames = [Image.fromarray(env.last_event.frame)]
            encoded_frames = [encode_image(frame) for frame in init_frames]
            # print("Added initial frame")

        # Setup for LLM-Planner
        seen_objs = env.get_visible_objects().split(", ")

        # Get initial high-level plan
        curr_task = {
            "task_instr": [goal_instr],
            "step_instr": step_instrs,
            "vis_objs": ", ".join(seen_objs), 
            "completed_plans": []
        }

        # Format the current task for better readability
        formatted_task = {
            "task_instr": curr_task["task_instr"][0],
            "step_instr": [step for sublist in curr_task["step_instr"] for step in sublist] if curr_task["step_instr"] and isinstance(curr_task["step_instr"][0], list) else curr_task["step_instr"],
            "vis_objs": curr_task["vis_objs"],
            "completed_plans": curr_task["completed_plans"]
        }
        
        log.debug("Current task:")
        log.debug(f"  Task instruction: {formatted_task['task_instr']}")
        if use_step_instructions:
            log.debug(f"  Step instructions: {formatted_task['step_instr']}")
        else:
            log.debug("  Step instructions: Disabled")
        log.debug(f"  Visible objects: {formatted_task['vis_objs']}")
        log.debug(f"  Completed plans: {formatted_task['completed_plans']}")

        high_level_plans = self.llm_planner.get_high_level_plan(curr_task, encoded_frames=encoded_frames)
        initial_high_level_plans = high_level_plans.copy()  # Store the initial plans
        
        # Display the full high-level plan more prominently
        log.debug("=" * 50)
        log.debug("GENERATED HIGH-LEVEL PLAN:")
        for i, plan in enumerate(high_level_plans, 1):
            log.debug(f"  {i}. {plan}")
        log.debug("=" * 50)
        log.debug(f"Visible objects: {seen_objs}")
        
        # Get max retries and max replanning from config
        max_retries = self.config["llm_planner"].get("max_retries", 3)
        max_replanning = self.config["llm_planner"].get("max_replanning", 10)
        
        # Setup counters
        retry_count = 0
        replanning_count = 0  # Track total replanning attempts for this task
        
        # Run until high-level plans are exhausted
        while high_level_plans and replanning_count <= max_replanning:
            plan = high_level_plans.pop(0).strip()
            log.debug(f"Plan: {plan}")
            
            # Try to parse the plan to extract object names
            try:
                plan_parts = plan.split()
                action = plan_parts[0]
                object_name = ' '.join(plan_parts[1:]) if len(plan_parts) > 1 else ""
            except:
                action = plan
                object_name = ""
            
            try:
                action_ret = env.llm_skill_interact(plan)
                
            except Exception as e:
                # Handle assertion errors or other exceptions from Thor connector
                log.warning(f"Error executing '{plan}': {str(e)}")
                if "instruction not supported" in str(e).lower():
                    log.warning(f"Instruction '{plan}' not supported. Skipping and continuing.")
                    # Add to completed plans with a note that it was skipped
                    completed_plans.append(f"{plan} [SKIPPED - UNSUPPORTED]")
                    retry_count = 0  # Reset retry counter
                    continue
                elif "no closest" in str(e).lower():
                    log.warning(f"No closest object found for '{plan}'. Skipping and continuing.")
                    completed_plans.append(f"{plan} [SKIPPED - NO CLOSEST OBJECT]")
                    retry_count = 0  # Reset retry counter
                    continue
                elif "object" in str(e).lower() and "not found" in str(e).lower() and object_name:
                    # Try fuzzy matching if object not found
                    available_objects = env.get_visible_objects().split(", ")
                    matched_obj, similarity = self.match_object_name(object_name, available_objects)
                    
                    if matched_obj:
                        # Create a new plan with the matched object name
                        matched_plan = f"{action} {matched_obj}"
                        log.debug(f"Object '{object_name}' not found. Using fuzzy match '{matched_obj}' (similarity: {similarity:.2f})")
                        
                        try:
                            # Try again with the matched object
                            action_ret = env.llm_skill_interact(matched_plan)
                            if action_ret['success']:
                                log.debug(f"SUCCESS: '{matched_plan}' executed successfully (fuzzy match)")
                                completed_plans.append(matched_plan)
                                retry_count = 0  # Reset retry counter on success
                                continue
                        except Exception as new_e:
                            log.warning(f"Error executing fuzzy matched plan: {str(new_e)}")
                    else:
                        log.warning(f"No matching object found for '{object_name}'. Best similarity: {similarity:.2f} (threshold: {self.obj_sim_threshold})")
                else:
                    # For other exceptions, re-raise
                    raise
            # Save frame after each successful action
            if action_ret['success']:
                frame_history.append(env.last_event.frame.copy())
            

            if not action_ret['success']:
                log.warning(action_ret['message'])
                failed_plans.append({"plan": plan, "error": action_ret['message']})
                
                # Check for unsupported instruction
                if "instruction not supported" in action_ret['message'].lower() or "not supported" in action_ret['message'].lower():
                    log.warning(f"Instruction '{plan}' not supported. Skipping and continuing.")
                    # Add to completed plans with a note that it was skipped
                    completed_plans.append(f"{plan} [SKIPPED - UNSUPPORTED]")
                    retry_count = 0  # Reset retry counter
                    continue
                
                # Check for object not found and try fuzzy matching
                if "object" in action_ret['message'].lower() and "not found" in action_ret['message'].lower() and object_name:
                    # Try fuzzy matching if object not found
                    available_objects = env.get_visible_objects().split(", ")
                    matched_obj, similarity = self.match_object_name(object_name, available_objects)
                    
                    if matched_obj:
                        # Create a new plan with the matched object name
                        matched_plan = f"{action} {matched_obj}"
                        log.debug(f"Object '{object_name}' not found. Using fuzzy match '{matched_obj}' (similarity: {similarity:.2f})")
                        
                        try:
                            # Try again with the matched object
                            action_ret = env.llm_skill_interact(matched_plan)
                            if action_ret['success']:
                                log.debug(f"SUCCESS: '{matched_plan}' executed successfully (fuzzy match)")
                                completed_plans.append(matched_plan)
                                retry_count = 0  # Reset retry counter on success
                                continue
                        except Exception as new_e:
                            log.warning(f"Error executing fuzzy matched plan: {str(new_e)}")
                    else:
                        log.warning(f"No matching object found for '{object_name}'. Best similarity: {similarity:.2f} (threshold: {self.obj_sim_threshold})")
            if vision:
                new_frames = [Image.fromarray(env.last_event.frame)]
                encoded_frames.extend([encode_image(frame) for frame in new_frames])


            # Dynamic re-planning if enabled
            if not action_ret['success'] and dynamic:
                # Check if we've exceeded max retries for this action
                if retry_count >= max_retries:
                    log.warning(f"Exceeded maximum retries ({max_retries}) for this action. Moving on.")
                    retry_count = 0
                    continue
                
                # Check if we've exceeded max replanning for this task
                if replanning_count >= max_replanning:
                    log.warning(f"Exceeded maximum replanning attempts ({max_replanning}) for this task. Continuing with remaining plans.")
                    retry_count = 0
                    continue
                
                retry_count += 1
                replanning_count += 1
                log.debug(f"Dynamic replanning attempt {retry_count}/{max_retries} for current action (Total: {replanning_count}/{max_replanning})")
                
                curr_vis_objs = env.get_visible_objects().split(", ")
                
                # Add new objects to seen_objs without duplicates
                for obj in curr_vis_objs:
                    if obj and obj not in seen_objs:  # Check if obj is not empty
                        seen_objs.append(obj)
                
                # Sort for consistent output
                seen_objs.sort()
                
                # Update the task with current visible objects and completed plans
                curr_task = {
                    "task_instr": [goal_instr],
                    "step_instr": step_instrs,
                    "vis_objs": ", ".join(seen_objs),  # Convert back to comma-separated string
                    "completed_plans": completed_plans,
                    "failed_plans": failed_plans
                }
                
                # Format the updated task for better readability
                formatted_task = {
                    "task_instr": curr_task["task_instr"][0],
                    "step_instr": [step for sublist in curr_task["step_instr"] for step in sublist] if curr_task["step_instr"] and isinstance(curr_task["step_instr"][0], list) else curr_task["step_instr"],
                    "vis_objs": curr_task["vis_objs"],
                    "completed_plans": curr_task["completed_plans"]
                }
                
                log.debug("Updated task for dynamic replanning:")
                log.debug(f"  Task instruction: {formatted_task['task_instr']}")
                if use_step_instructions:
                    log.debug(f"  Step instructions: {formatted_task['step_instr']}")
                else:
                    log.debug("  Step instructions: Disabled")
                log.debug(f"  Visible objects: {formatted_task['vis_objs']}")
                log.debug(f"  Completed plans: {formatted_task['completed_plans']}")
                
                
                # Get current frame if vision is used
                # encoded_frames = []
                # if vision:
                #     curr_frame = [Image.fromarray(env.last_event.frame)]
                #     encoded_frames.extend([self.encode_image(frame) for frame in curr_frame])

                # Generate new plans for dynamic replanning
                high_level_plans = self.llm_planner.get_high_level_plan(curr_task, encoded_frames=encoded_frames)
                dynamic_plans.append(high_level_plans.copy())  # Keep track of dynamic plans generated

                # Display the dynamically generated high-level plan more prominently
                log.debug("=" * 50)
                log.debug("DYNAMICALLY GENERATED HIGH-LEVEL PLAN:")
                for i, plan in enumerate(high_level_plans, 1):
                    log.debug(f"  {i}. {plan}")
                log.debug("=" * 50)
                log.debug(f"Visible objects: {seen_objs}")
            elif not action_ret['success']:
                pass
            else:
                log.debug(f"SUCCESS: '{plan}' executed successfully")
                completed_plans.append(plan)
                retry_count = 0  # Reset retry counter on success
        
        # Check if we hit the replanning limit
        if replanning_count >= max_replanning:
            log.warning(f"Task ended because maximum replanning attempts ({max_replanning}) were reached.")
        
        # Check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        log.debug('target goal: ' + json.dumps(env.task.get_targets()))
        log.debug('success: ' + str(goal_satisfied))
        if goal_satisfied:
            success = True

        # Record results with detailed plan information
        log_entry = {
            'trial': traj_data['task_id'],
            'scene': scene_name,
            'type': traj_data['task_type'],
            'repeat_idx': int(r_idx),
            'goal_instr': goal_instr,
            'initial_high_level_plans': initial_high_level_plans,
            'completed_plans': completed_plans,
            'dynamic_plans': [", ".join(plans) for plans in dynamic_plans],
            'failed_plans': failed_plans,
            'success': success,
            'frame_history': frame_history
        }

        return log_entry
                
    def run_evaluation(self, dry_run=False, enforce_knn_set=False):
        """Run evaluation on all tasks"""
        results = []
        save_path = self.config["out_dir"]
        
        # Create the output directory if it doesn't exist
        if self.config.get("save_results", False):
            save_path = os.path.join(self.config["out_dir"], "results")
            os.makedirs(save_path, exist_ok=True)

        if dry_run:
            log.info("Dry run mode enabled. Only evaluating 3 task.")
            self.tasks = self.tasks[:3]
        
        # Main eval loop
        start = time.time()
        for task_idx, task in tqdm(enumerate(self.tasks), 
                                  desc="Tasks"):
            try:
                log.debug(task)
                traj_data = load_task_json(task)
                r_idx = task['repeat_idx']

                if enforce_knn_set:
                    goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
                    # Check if the goal instruction is in the knn set
                    if goal_instr not in self.knn_set["task_instr"].values:
                        log.warning(f"Goal instruction not found in KNN set: {goal_instr}")
                        continue
                    else:
                        log.debug(f"Goal instruction found in KNN set: {goal_instr}")

                log.debug(f"Evaluating ({task_idx+1}/{len(self.tasks)}): {traj_data['root']}")
                
                result = self.evaluate_task(
                    self.config["llm_planner"]["engine"],
                    traj_data, 
                    r_idx,
                    dynamic=self.config["llm_planner"]["dynamic"],
                    vision=self.config["llm_planner"]["vision"]          
                )
                results.append(result)
                
                # Save result for debugging if enabled
                if self.config.get("save_results", False):
                    self.save_result(result, save_path)

            except Exception as e:
                import traceback
                traceback.print_exc()
                log.error(f"Error processing task {task_idx+1}/{len(self.tasks)}: {repr(e)}")
                
                # Create a failure result to keep tracking progress
                try:
                    failure_result = {
                        'trial': task['task_id'] if 'task_id' in task else f"unknown_task_{task_idx}",
                        'scene': f"scene_{task_idx}",
                        'type': "failed",
                        'repeat_idx': task['repeat_idx'] if 'repeat_idx' in task else 0,
                        'goal_instr': "Task processing failed with exception",
                        'inferred_steps': [f"ERROR: {repr(e)}"],
                        'success': False
                    }
                    results.append(failure_result)
                    
                    # Save failure result if saving is enabled
                    if self.config.get("save_results", False):
                        self.save_result(failure_result, save_path)
                except:
                    # If we can't even create a failure result, just continue
                    log.error("Failed to create failure result entry")
        
        # Print results
        self._print_results(results, start)
        return results

    def run_evaluation_helper(self, task, task_idx, save_path):
        """Helper function to run evaluation on a single task"""
        with self.env_lock:
            env_index = self.envs_in_use.index(False)
            if env_index is None:
                raise RuntimeError("No available environments for evaluation")
            self.envs_in_use[env_index] = True

        try:
            log.debug(task)
            traj_data = load_task_json(task)
            r_idx = task['repeat_idx']
            log.debug(f"Evaluating ({task_idx+1}/{len(self.tasks)}): {traj_data['root']}")
            
            result = self.evaluate_task(
                self.config["llm_planner"]["engine"],
                traj_data, 
                r_idx,
                dynamic=self.config["llm_planner"]["dynamic"],
                vision=self.config["llm_planner"]["vision"],
                env=self.envs[env_index]
            )
            with self.env_lock:
                self.envs_in_use[env_index] = False

            # Save result for debugging if enabled
            if self.config.get("save_results", False):
                self.save_result(result, save_path)
            
            return result

        except Exception as e:
            with self.env_lock:
                self.envs_in_use[env_index] = False
            import traceback
            traceback.print_exc()
            log.error(f"Error processing task {task_idx+1}/{len(self.tasks)}: {repr(e)}")
            
            # Create a failure result to keep tracking progress
            try:
                failure_result = {
                    'trial': task['task_id'] if 'task_id' in task else f"unknown_task_{task_idx}",
                    'scene': f"scene_{task_idx}",
                    'type': "failed",
                    'repeat_idx': task['repeat_idx'] if 'repeat_idx' in task else 0,
                    'goal_instr': "Task processing failed with exception",
                    'inferred_steps': [f"ERROR: {repr(e)}"],
                    'success': False
                }
                
                
                # Save failure result if saving is enabled
                if self.config.get("save_results", False):
                    self.save_result(failure_result, save_path)

                return failure_result
            except:
                # If we can't even create a failure result, just continue
                log.error("Failed to create failure result entry")

    def run_evaluation_multithreaded(self, dry_run=False, enforce_knn_set=False):
        """Run evaluation on all tasks"""
        results = []
        save_path = self.config["out_dir"]
        
        # Create the output directory if it doesn't exist
        if self.config.get("save_results", False):
            save_path = os.path.join(self.config["out_dir"], "results")
            os.makedirs(save_path, exist_ok=True)

        if dry_run:
            log.info("Dry run mode enabled. Only evaluating 3 task.")
            self.tasks = self.tasks[:3]
        
        # Main eval loop
        start = time.time()
        task_idxs = range(len(self.tasks))
        save_paths = [save_path] * len(self.tasks)
        print(self.tasks[0])
        with ThreadPool(processes=self.config.get("num_workers", 1)) as pool:
            results = pool.starmap(self.run_evaluation_helper, zip(self.tasks, task_idxs, save_paths))

        # Print results
        self._print_results(results, start)
        return results
    
    def _print_results(self, results, start_time):
        """Print evaluation results"""
        n = len(results)
        
        if n == 0:
            log.warning("No results collected - all tasks may have failed with exceptions")
            log.debug(f'success rate: 0.00 % (0/0)')
            log.debug(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start_time)))}')
            log.debug('------------------------')
            log.debug(yaml.dump(self.config))
            return
        
        n_success = sum(1 for e in results if e != None and e['success'])
        
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start_time)))}')
        log.info('------------------------')
        log.info(yaml.dump(self.config))
    
    def save_result(self, result_dict, base_path):
        """Save result for debugging with complete trajectory"""
        # Create task-specific directory
        task_dir = os.path.join(base_path, f"{result_dict['trial']}_{result_dict['repeat_idx']}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Save result as JSON (excluding frame_history to avoid huge files)
        result_for_json = {k: v for k, v in result_dict.items() if k != 'frame_history'}
        with open(os.path.join(task_dir, "result.json"), "w") as f:
            json.dump(result_for_json, f, indent=2)
        
        # Save all frames from the trajectory
        for i, frame in enumerate(result_dict['frame_history']):
            # Create PIL image
            img = Image.fromarray(frame)
            
            # Add text for frame number and success status
            draw = ImageDraw.Draw(img)
            text = f"Frame {i}"
            if i == len(result_dict['frame_history']) - 1:  # Last frame
                success_str = 'SUCCESS' if result_dict['success'] else 'FAIL'
                text += f" ({success_str})"
            
            # Add the text to the image
            draw.text((10, 10), text, fill=(255, 255, 255))
            
            # Save the frame
            img.save(os.path.join(task_dir, f"frame_{i:04d}.png"))
        
        # Save detailed plan information
        with open(os.path.join(task_dir, "plans.txt"), "w") as f:
            # Task description
            f.write(f"Task: {result_dict['goal_instr']}\n\n")
            
            # Initial high-level plans
            f.write("Initial High-Level Plans:\n")
            for i, plan in enumerate(result_dict['initial_high_level_plans'], 1):
                f.write(f"{i}. {plan}\n")
            f.write("\n")
            
            # Completed plans
            f.write("Completed Plans:\n")
            if result_dict['completed_plans']:
                for i, plan in enumerate(result_dict['completed_plans'], 1):
                    f.write(f"{i}. {plan}\n")
            else:
                f.write("None\n")
            f.write("\n")

            # Dynamic plans
            if result_dict['dynamic_plans']:
                f.write("Dynamic Plans Generated During Replanning:\n")
                for i, dynamic_plan_set in enumerate(result_dict['dynamic_plans'], 1):
                    f.write(f"Replanning Attempt {i}:\n")
                    f.write(dynamic_plan_set + "\n")
                f.write("\n")
            

            # Failed plans (only if there are any)
            if result_dict['failed_plans']:
                f.write("Failed Plans:\n")
                for i, plan_data in enumerate(result_dict['failed_plans'], 1):
                    f.write(f"{i}. {plan_data['plan']}\n")
                    f.write(f"   Error: {plan_data['error']}\n")
                f.write("\n")
            
            # Final result
            f.write(f"Final Status: {'SUCCESS' if result_dict['success'] else 'FAILURE'}")
    
    def preprocess_dataset(self):
        """Preprocess dataset if needed"""
        args_dict = self.config["alfred"]["env_args"]
        number_of_dirs = len(list(os.listdir(args_dict['data'])))
        do_preprocessing = number_of_dirs < 50  # one-time process
        
        if do_preprocessing:
            log.info("\nPreprocessing dataset... Do this once as required:")
            vocab = None  # todo
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(self.splits)

    def clean_llm_output(self, llm_out):
        """
        Clean the LLM output by removing the 'Next Plans:' prefix and converting to a comma-separated list.
        
        Args:
            llm_out (str): Raw LLM output string like "Next Plans: Navigation countertop, OpenObject microwave, ..."
            
        Returns:
            list: List of cleaned plan steps
        """
        # Remove "Next Plans:" prefix if present
        if "Next Plans:" in llm_out:
            cleaned_text = llm_out.split("Next Plans:")[1].strip()
        else:
            cleaned_text = llm_out.strip()
        
        # Split by comma and strip whitespace from each item
        plans = [plan.strip() for plan in cleaned_text.split(',')]
        
        return plans

    def match_object_name(self, generated_name, available_objects):
        """
        Match a generated object name to available objects using sentence embeddings.
        
        Args:
            generated_name: Object name generated by the LLM
            available_objects: List of available objects in the environment
            
        Returns:
            Tuple of (matched_object, similarity_score) or (None, 0) if no match above threshold
        """
        if not generated_name or not available_objects:
            return None, 0
            
        # Try direct matching first (case insensitive)
        generated_name_lower = generated_name.lower()
        for obj in available_objects:
            if obj.lower() == generated_name_lower:
                return obj, 1.0  # Perfect match
                
        # If no direct match, use sentence embeddings for fuzzy matching
        try:
            # Encode the generated name
            generated_embedding = self.obj_encoder.encode(generated_name, convert_to_tensor=True, show_progress_bar=False)
            
            # Encode all available objects
            available_embeddings = self.obj_encoder.encode(available_objects, convert_to_tensor=True, show_progress_bar=False)
            
            # Calculate similarity scores
            similarities = cos_sim(generated_embedding, available_embeddings)[0]
            
            # Find the best match above threshold
            best_match_idx = similarities.argmax().item()
            best_match_score = similarities[best_match_idx].item()
            
            # Check if the best match exceeds the threshold
            if best_match_score >= self.obj_sim_threshold:
                return available_objects[best_match_idx], best_match_score
                
            return None, best_match_score
            
        except Exception as e:
            log.warning(f"Error during fuzzy object matching: {str(e)}")
            return None, 0


def run_with_config(config_path, dry_run=False, enforce_knn_set=False):
    """Helper function to run evaluation with a specific config file"""
    evaluator = AlfredEvaluator(config_path)
    evaluator.preprocess_dataset()
    results = evaluator.run_evaluation(dry_run=dry_run, enforce_knn_set=enforce_knn_set)
    return results

def run_multiple_configs(config_paths, dry_run=False):
    """Helper function to run evaluation with multiple config files"""
    all_results = {}
    for config_path in config_paths:
        log.info(f"Running evaluation with config: {config_path}")
        results = run_with_config(config_path, dry_run=dry_run)
        all_results[config_path] = results
    for results in all_results.values():
        n = len(results)
        n_success = sum(1 for e in results if e['success'])
        log.info(f"Config: {config_path} - Success rate: {n_success / n * 100:.2f} % ({n_success}/{n})")
    return all_results

def run_evaluation_helper(task, task_idx, save_path):
    """Helper function to run evaluation on a single task"""
    created = multiprocessing.Process()
    worker_number = created._identity[0]
    print(f"Initialized worker {worker_number} for evaluation")
    try:
        log.debug(task)
        traj_data = load_task_json(task)
        r_idx = task['repeat_idx']
        log.debug(f"Evaluating ({task_idx+1}/{len(self.tasks)}): {traj_data['root']}")
        
        result = self.evaluate_task(
            self.config["llm_planner"]["engine"],
            traj_data, 
            r_idx,
            dynamic=self.config["llm_planner"]["dynamic"],
            vision=self.config["llm_planner"]["vision"],
            env=self.envs[worker_number - 1]
        )
        
        
        # Save result for debugging if enabled
        if self.config.get("save_results", False):
            self.save_result(result, save_path)
        
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        log.error(f"Error processing task {task_idx+1}/{len(self.tasks)}: {repr(e)}")
        
        # Create a failure result to keep tracking progress
        try:
            failure_result = {
                'trial': task['task_id'] if 'task_id' in task else f"unknown_task_{task_idx}",
                'scene': f"scene_{task_idx}",
                'type': "failed",
                'repeat_idx': task['repeat_idx'] if 'repeat_idx' in task else 0,
                'goal_instr': "Task processing failed with exception",
                'inferred_steps': [f"ERROR: {repr(e)}"],
                'success': False
            }
            
            
            # Save failure result if saving is enabled
            if self.config.get("save_results", False):
                self.save_result(failure_result, save_path)

            return failure_result
        except:
            # If we can't even create a failure result, just continue
            log.error("Failed to create failure result entry")

def remove_repeated_steps(completed_plans):
    """Remove any repeated steps in completed plans"""
    unique_plans = [completed_plans[0]]
    prev_step = completed_plans[0]
    for plan in completed_plans[1:]:
        if plan != prev_step and not (plan.startswith("Navigation") and prev_step.startswith("Navigation")):
            unique_plans.append(plan)
            prev_step = plan
    return unique_plans

def convert_plan_to_knn_format(plan):
    # Split plan into words and cast to tuple
    plan = [tuple(p.split(" ")) for p in plan]
    # Turn the whole tuple into a single string
    plan = [str(p) for p in plan]
    # Make each step a single item list
    plan = [[p] for p in plan]
    return plan



def update_knn_set(config, results_path, prev_results_path, knn_set_path, knn_set_save_path):
    knn_set = pd.read_pickle(knn_set_path)

    result_files = os.listdir(results_path)
    json_files = [f for f in result_files if f.endswith('.json')]
    if len(json_files) > 1:
        log.warning(f"Multiple JSON files found in results path: {json_files}. Using the first one: {json_files[0]}")
    result_file = os.path.join(results_path, json_files[0])
    with open(result_file, 'r') as f:
        results = json.load(f)
        
    prev_result_files = os.listdir(prev_results_path)
    prev_json_files = [f for f in prev_result_files if f.endswith('.json')]
    if len(prev_json_files) > 1:
        log.warning(f"Multiple JSON files found in previous results path: {prev_json_files}. Using the first one: {prev_json_files[0]}")
    prev_result_file = os.path.join(prev_results_path, prev_json_files[0])
    with open(prev_result_file, 'r') as f:
        prev_results = json.load(f)

    for i in range(len(results)):
        result = results[i]
        prev_result = prev_results[i]
        
        if config["criteria"] == "task_success":
            if not result["success"]:
                continue
        elif config["criteria"] == "subgoal_success_rate":
            result_completed_subgoals = result["completed_plans"] if "completed_plans" in result else []
            if result_completed_subgoals:
                result_completed_subgoals = remove_repeated_steps(result_completed_subgoals)
            prev_completed_subgoals = prev_result["completed_plans"] if "completed_plans" in prev_result else []
            if prev_completed_subgoals:
                prev_completed_subgoals = remove_repeated_steps(prev_completed_subgoals)
            if len(result_completed_subgoals) < len(prev_completed_subgoals):
                continue
        else:
            raise ValueError(f"Unsupported criteria: {config['criteria']}")
        
        completed_plans = result["completed_plans"] if "completed_plans" in result else []
        if not completed_plans:
            continue
        else:
            remove_repeated_steps(completed_plans)
        completed_plans_knn = convert_plan_to_knn_format(completed_plans)

        goal_instr = result["goal_instr"]
        knn_set_task_instrs = knn_set["task_instr"].tolist()
        task_index = knn_set_task_instrs.index(goal_instr)

        # Set gold_traj in the knn set to the completed plans from this result
        knn_set.at[task_index, 'gold_traj'] = completed_plans_knn
    knn_set.to_pickle(knn_set_save_path)
    log.info(f"KNN set updated and saved to {knn_set_save_path}")





def run_prompt_iteration():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_alfred.yaml')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--enforce_knn_set', action='store_true')
    args = parser.parse_args()

    with open(args.config) as reader:
        config = yaml.safe_load(reader)

    # Make a directory for KNN sets if it doesn't exist
    os.makedirs(os.path.join(config["out_dir"],"knn_sets"), exist_ok=True)
    # Make a directory for results if it doesn't exist
    os.makedirs(os.path.join(config["out_dir"],"results"), exist_ok=True)

    prompt_iterations = config["iterations"]

    # Create a new config for this iteration with updated KNN set path
    iter_config = config.copy()
    prev_results_path = config["base_results_path"]
    knn_set_path = config["llm_planner"]["knn_dataset_path"]
    knn_set_save_path = os.path.join(config["out_dir"],"knn_sets", "knn_set_initial.pkl")
    update_knn_set(config, prev_results_path, prev_results_path, knn_set_path, knn_set_save_path)
    iter_config["llm_planner"]["knn_dataset_path"] = knn_set_save_path
    iter_config["out_dir"] = os.path.join(config["out_dir"], f"iteration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}") 
    os.makedirs(iter_config["out_dir"], exist_ok=True)
    iter_config_path = os.path.join(iter_config["out_dir"], "config.yaml")
    with open(iter_config_path, 'w') as f:
        yaml.dump(iter_config, f)

    for i in range(prompt_iterations):
        log.info(f"Starting prompt iteration {i+1}/{prompt_iterations}")
        results = run_with_config(iter_config_path, dry_run=args.dry_run, enforce_knn_set=True)
        # Save results for this iteration
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(iter_config["out_dir"], f"results_{timestamp}.json")
        
        # Create JSON-serializable version of results (without frame_history)
        json_results = []
        for result in results:
            # Create a copy without frame_history
            json_result = {k: v for k, v in result.items() if k != 'frame_history'}
            json_results.append(json_result)
            
        with open(result_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        log.info(f"Results saved to {result_file}")

        # Update KNN set for the next iteration
        knn_set_save_path = os.path.join(config["out_dir"], "knn_sets", f"knn_set_iteration_{i+1}.pkl")
        update_knn_set(config, iter_config["out_dir"], prev_results_path, knn_set_path, knn_set_save_path)

        prev_results_path = iter_config["out_dir"]

        # Update config for the next iteration
        iter_config["llm_planner"]["knn_dataset_path"] = knn_set_save_path
        iter_config["out_dir"] = os.path.join(config["out_dir"], f"iteration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(iter_config["out_dir"], exist_ok=True)
        iter_config_path = os.path.join(iter_config["out_dir"], "config.yaml")
        with open(iter_config_path, 'w') as f:
            yaml.dump(iter_config, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_alfred.yaml')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--enforce_knn_set', action='store_true')
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = AlfredEvaluator(args.config)
    
    # Run preprocess step only once on new installation
    evaluator.preprocess_dataset()
    
    # Run evaluation
    num_workers = evaluator.config.get("num_workers", 1)
    if num_workers > 1:
        results = evaluator.run_evaluation_multithreaded(dry_run=args.dry_run, enforce_knn_set=args.enforce_knn_set)
    else:
        results = evaluator.run_evaluation(dry_run=args.dry_run, enforce_knn_set=args.enforce_knn_set)

    # Save results if configured
    if evaluator.config.get("save_results", False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(evaluator.config["out_dir"], f"results_{timestamp}.json")
        
        # Create JSON-serializable version of results (without frame_history)
        json_results = []
        for result in results:
            # Create a copy without frame_history
            json_result = {k: v for k, v in result.items() if k != 'frame_history'}
            json_results.append(json_result)
            
        with open(result_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        log.info(f"Results saved to {result_file}")

if __name__ == '__main__':
    main()
    # run_prompt_iteration()
    # config_files = [
    #     "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/config/config_alfred_Qwen3-VL-30B.yaml",
    #     "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/config/config_alfred_Qwen3-VL-30B_vision.yaml",
    #     "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/config/config_alfred_Qwen3-VL-30B_dynamic.yaml",
    #     "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/config/config_alfred_Qwen3-VL-30B_dynamic_vision.yaml",
    # ]
    # all_results = run_multiple_configs(config_files, dry_run=False)




