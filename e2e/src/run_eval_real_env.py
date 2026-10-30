import os
import re
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
from typing import List, Union
from e2e_utils import encode_image
from hlp_planner import LLM_Planner, Tree_Search_LLM_Planner, Entropy_reduction_LLM_Planner, GT_Planner
from planning_env import build_planning_env
from VLLM_Interface import VLLM_Interface
from Inference_Interface import InferenceInterface

PLANNERS = {
    "llm": LLM_Planner,
    "tree_search": Tree_Search_LLM_Planner,
    "entropy_reduction": Entropy_reduction_LLM_Planner,
    "gt": GT_Planner
}

sys.path.insert(0, '..')
sys.path.insert(0, '')


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
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

success_obj_replace_dict = {
    "drawer": "drawer basket",
}

class VLM_Functions:
    """
    Thin helpers over VLLM_Interface for scene understanding and action-outcome checks.
    """

    def __init__(self, client: Union[OpenAI, None] = None, engine: str = "Qwen/Qwen3-VL-8B-Instruct") -> None:
        self.client = client
        self.engine = engine
        self.vlm = InferenceInterface(engine, client=client)

    def _image_to_base64(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        if isinstance(image, str):
            return image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            return encode_image(image)
        raise TypeError(
            f"image must be base64 str, PIL.Image, or ndarray; got {type(image).__name__}"
        )

    def get_success_socratic(
        self,
        image: Union[str, Image.Image, np.ndarray],
        most_recent_action: str,
    ) -> bool:
        """
        True if the VLM judges that `most_recent_action` was successfully carried out
        given the current image (post-action), using token logprobs for Yes/No.
        """
        encoded = self._image_to_base64(image)
        action = most_recent_action.strip()
        action_parts = action.split()

        if action_parts[0].lower() == "putobject":
            subject = ' '.join(action_parts[1:-1])
        else:
            subject = ' '.join(action_parts[1:])

        state_prompt = f"Whats the state of the {subject}?"
        state = self.vlm.generate(state_prompt, images=[encoded], temperature=0.0, n=1)[0].strip()

        if action_parts[0].lower() == "openobject":
            prompt = f"The following is a description of the state of {subject}: \n{state}. Is the {subject} open in this image? Answer with exactly one word: Yes or No."
        elif action_parts[0].lower() == "closeobject":
            prompt = f"The following is a description of the state of {subject}: \n{state}. Is the {subject} closed in this image? Answer with exactly one word: Yes or No."
        elif action_parts[0].lower() == "pickupobject":
            prompt = f"The following is a description of the state of {subject}: \n{state}. Is the robot currently holding the {subject}? Answer with exactly one word: Yes or No."
        elif action_parts[0].lower() == "putobject":
            prompt = f"The following is a description of the state of {subject}: \n{state}. Is the {subject} in the {action_parts[-1]}? Answer with exactly one word: Yes or No."

        print(f"Prompt for success check: {prompt}")
       
        probs = self.vlm.yes_no_probabilities(prompt, images=[])
        return bool(probs[0] >= probs[1])
    
    def get_success(
        self,
        image: Union[str, Image.Image, np.ndarray],
        most_recent_action: str,
    ) -> bool:
        """
        True if the VLM judges that `most_recent_action` was successfully carried out
        given the current image (post-action), using token logprobs for Yes/No.
        """
        encoded = self._image_to_base64(image)
        action = most_recent_action.strip()
        action_parts = action.split()

        lower_action_parts = [part.lower() for part in action_parts]
        lower_action_parts = [success_obj_replace_dict.get(part, part) for part in lower_action_parts]

        if action_parts[0].lower() == "putobject":
            subject = ' '.join(lower_action_parts[1:-1])
        else:
            subject = ' '.join(lower_action_parts[1:])

        if action_parts[0].lower() == "openobject":
            prompt = f"Is the {subject} open in this image? Answer with exactly one word: Yes or No."
        elif action_parts[0].lower() == "closeobject":
            prompt = f"Is the {subject} closed in this image? Answer with exactly one word: Yes or No."
        elif action_parts[0].lower() == "pickupobject":
            prompt = f"Is the robot currently holding the {subject}? Answer with exactly one word: Yes or No."
        elif action_parts[0].lower() == "putobject":
            prompt = f"Is the {subject} in the {lower_action_parts[-1]}? Answer with exactly one word: Yes or No."

        print(f"Prompt for success check: {prompt}")
       
        probs = self.vlm.yes_no_probabilities(prompt, images=[encoded])
        return bool(probs[0] >= probs[1])
    
    def get_success_pair(
        self,
        before_image: Union[str, Image.Image, np.ndarray],
        after_image: Union[str, Image.Image, np.ndarray],
        most_recent_action: str,
    ) -> bool:
        """
        True if the VLM judges that `most_recent_action` was successfully carried out,
        comparing the scene before the action (`before_image`) with the scene after
        the action (`after_image`).
        """
        before_encoded = self._image_to_base64(before_image)
        after_encoded = self._image_to_base64(after_image)
        action = most_recent_action.strip()
        action_parts = action.split()

        lower_action_parts = [part.lower() for part in action_parts]
        lower_action_parts = [success_obj_replace_dict.get(part, part) for part in lower_action_parts]

        if action_parts[0].lower() == "putobject":
            subject = ' '.join(lower_action_parts[1:-1])
        else:
            subject = ' '.join(lower_action_parts[1:])

        header = (
            "Image 1 shows the scene before the robot attempted an action. "
            "Image 2 shows the scene after. "
        )
        if action_parts[0].lower() == "openobject":
            question = f"In Image 2, is the {subject} more open than in Image 1?"
        elif action_parts[0].lower() == "closeobject":
            question = f"In Image 2, is the {subject} more closed than in Image 1?"
        elif action_parts[0].lower() == "pickupobject":
            question = (
                f"In Image 2, is the {subject} currently grasped by the robot's gripper? "
                f"Answer Yes only if the {subject} is clearly held in the gripper in Image 2. "
                f"Answer No if the {subject} is resting on any surface or inside any container in Image 2."
            )
        elif action_parts[0].lower() == "putobject":
            receptacle = lower_action_parts[-1]
            question = (
                f"In Image 1, was the {subject} held by the gripper, and in Image 2, is the {subject} "
                f"now resting inside the {receptacle} with the gripper empty?"
            )
        else:
            raise ValueError(f"Unsupported action for get_success_pair: {action_parts[0]}")

        prompt = header + question + " Answer with exactly one word: Yes or No."
        print(f"Prompt for success-pair check: {prompt}")

        probs = self.vlm.yes_no_probabilities(prompt, images=[before_encoded, after_encoded])
        return bool(probs[0] >= probs[1])

    def get_failed_reason(
        self,
        image: Union[str, Image.Image, np.ndarray],
        most_recent_action: str,
    ) -> str:
        """
        Return a reason for the failure of the most recent action.
        """
        encoded = self._image_to_base64(image)
        action = most_recent_action.strip()
        prompt = (
            f"This image shows the scene after the robot attempted this action: {action}. "
            "Based only on what you see, why was that action not successfully completed? "
            "Answer with a single sentence."
        )
        text = self.vlm.generate(prompt, images=[encoded], temperature=0.0, n=1)[0].strip()
        return text.splitlines()[0].strip()


class RealEnvEvaluator:
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
        
        self.env = build_planning_env(
            "Put Away Fruit in the Drawer", load_policy_execution=True
        )
        
        # Initialize the sentence transformer for object name matching
        self.obj_encoder = SentenceTransformer(self.config["llm_planner"]["emb_model_name"], device="cuda:0")
        self.obj_sim_threshold = self.config["llm_planner"].get("obj_sim_threshold", 0.8)  # Default similarity threshold
        self.knn_set = pd.read_pickle(self.config["llm_planner"]["knn_dataset_path"])

        # Main VLM (shared with the planner backend) for scene-level helpers
        # like ``get_visible_objects``. Success/failure checks still go through
        # ``self.vlm_functions``, which can run on a different (e.g. Claude) backbone.
        self.vlm = InferenceInterface(
            self.config["llm_planner"]["engine"], client=self.client
        )

        # Initialize the VLM functions
        self.vlm_functions = VLM_Functions(client=None, engine="claude-sonnet-4-6")

        self._planning_action_strings = list(self.env.action_strings)
        self._planning_action_embeddings = self.obj_encoder.encode(
            self._planning_action_strings,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    def get_visible_objects(self, image) -> List[str]:
        """
        Return a list of object names visible in the image, using the main VLM.
        """
        if isinstance(image, np.ndarray):
            encoded = encode_image(Image.fromarray(image))
        elif isinstance(image, Image.Image):
            encoded = encode_image(image)
        else:
            encoded = image
        prompt = (
            "List every distinct physical object on the countertop that you can see in this image. "
            "Do not list objects in the background on the desk. "
            "Reply with a single comma-separated list of short object names only "
            "(no numbering, no extra sentences)."
        )
        text = self.vlm.generate(prompt, images=[encoded], temperature=0.0, n=1)[0].strip()
        first_line = text.splitlines()[0].strip()
        return [p.strip() for p in first_line.split(",") if p.strip()]

    def match_action(self, plan: str) -> int:
        """
        Map a free-form plan string to the discrete action index for ``env.step`` by
        cosine similarity against precomputed embeddings of ``PlanningEnv`` action strings.
        """
        plan = (plan or "").strip()
        if not plan:
            raise ValueError("plan must be a non-empty string")
        if not self._planning_action_strings:
            raise RuntimeError("Planning env has no discrete actions")
        plan_emb = self.obj_encoder.encode(
            plan, convert_to_tensor=True, show_progress_bar=False
        )
        sims = cos_sim(plan_emb, self._planning_action_embeddings)[0]
        return int(sims.argmax().item())

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


    def evaluate_task(self, goal_instr, dynamic=False, vision=False, env=None):
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
        obs, info = env.reset()

        # Initialize and save initial frame
        frame_history.append(obs)


        # Setup for evaluation
        done, success = False, False
        reward = 0

        # Get initial frames if vision model is used
        encoded_frames = []
        if vision:
            init_frames = [Image.fromarray(obs)]
            encoded_frames = [encode_image(frame) for frame in init_frames]
            # print("Added initial frame")

        # Setup for LLM-Planner
        use_objects = self.config["llm_planner"].get("use_objects", False)
        use_failure_reason = self.config["llm_planner"].get("use_failure_reason", True)
        seen_objs = self.get_visible_objects(obs) if use_objects else []

        # Get initial high-level plan
        curr_task = {
            "task_instr": [goal_instr],
            "vis_objs": ", ".join(seen_objs) if use_objects else [],
            "step_instr": [],
            "completed_plans": []
        }

        # Format the current task for better readability
        formatted_task = {
            "task_instr": curr_task["task_instr"][0],
            "vis_objs": curr_task["vis_objs"],
            "completed_plans": curr_task["completed_plans"]
        }
        
        log.debug("Current task:")
        log.debug(f"  Task instruction: {formatted_task['task_instr']}")
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
        previous_plan = ""
        
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
                # Reset retry count if the plan is different from the previous plan
                if plan != previous_plan:
                    retry_count = 0
                previous_plan = plan
                action_index = self.match_action(plan)
                prev_obs = obs.copy()
                obs, reward, terminated, truncated, info = env.step(action_index)
                # Save the image to a file
                cv2.imwrite(f"frame_{len(frame_history)}.png", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

                # Use the policy's own success signal for PickupObject; the VLM
                # is unreliable for grasp confirmation. Use the VLM for the rest.
                if action == "PickupObject":
                    success = info["policy_execution_success"]
                else:
                    success = self.vlm_functions.get_success_pair(prev_obs, obs, plan)
                log.debug(f"Action: {plan} Success: {success}")
                message = ""
                if not success and use_failure_reason:
                    message = self.vlm_functions.get_failed_reason(obs, plan)
                action_ret = {
                    'action': plan,
                    'success': success,
                    'message': message
                }
            
            except Exception as e:
                log.warning(f"Error executing '{plan}': {str(e)}")
                
            # Save frame after each  action
            if action_ret['success']:
                frame_history.append(obs)
            

            if not action_ret['success']:
                log.warning(action_ret['message'])
                failed_plans.append({"plan": plan, "error": action_ret['message']})
                
            if vision:
                new_frames = [Image.fromarray(obs)]
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
                
                if use_objects:
                    curr_vis_objs = self.get_visible_objects(obs)

                    # Add new objects to seen_objs without duplicates
                    for obj in curr_vis_objs:
                        if obj and obj not in seen_objs:  # Check if obj is not empty
                            seen_objs.append(obj)

                    # Sort for consistent output
                    seen_objs.sort()

                # Update the task with current visible objects and completed plans
                curr_task = {
                    "task_instr": [goal_instr],
                    "step_instr": [],
                    "vis_objs": ", ".join(seen_objs) if use_objects else [],
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
        
        success = input("Was the goal satisfied? (y/n)") == "y"

        print(f"Success: {success}")

        # Record results with detailed plan information
        log_entry = {
            'goal_instr': goal_instr,
            'initial_high_level_plans': initial_high_level_plans,
            'completed_plans': completed_plans,
            'dynamic_plans': [", ".join(plans) for plans in dynamic_plans],
            'failed_plans': failed_plans,
            'success': success,
            'frame_history': frame_history
        }

        return log_entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_fruit.yaml')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of consecutive evaluation runs to perform.')
    args = parser.parse_args()


    evaluator = RealEnvEvaluator(args.config)
    goal_instr = "Put Away Fruit in the Drawer"
    trial = re.sub(r"[^A-Za-z0-9]+", "_", goal_instr).strip("_").lower()

    for run_idx in range(args.num_runs):
        log.info(f"=== Starting run {run_idx + 1}/{args.num_runs} ===")
        result = evaluator.evaluate_task(goal_instr=goal_instr, dynamic=True, vision=False)
        print(result)

        if evaluator.config.get("save_results", False):
            out_dir = evaluator.config["out_dir"]
            os.makedirs(out_dir, exist_ok=True)
            # Timestamp (with milliseconds) keeps multiple runs against the same
            # out_dir from clobbering each other; save_result builds the per-run
            # directory as f"{trial}_{repeat_idx}".
            repeat_idx = datetime.datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:-3]
            result["trial"] = trial
            result["repeat_idx"] = repeat_idx
            evaluator.save_result(result, out_dir)
            log.info(f"Saved result to {os.path.join(out_dir, f'{trial}_{repeat_idx}')}")
            _ = input("Press Enter to continue...")

if __name__ == "__main__":
    main()