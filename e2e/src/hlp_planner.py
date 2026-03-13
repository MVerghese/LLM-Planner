"""
High-Level Planner for LLM-based Task Planning
----------------------------------------------
Implements the planning system that uses large language models to generate plans for
embodied AI tasks. Features KNN-based example retrieval, prompt engineering, and
plan formatting.
"""

import pandas as pd
import random
import logging
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
# from transformers import GPT2Tokenizer
from openai import OpenAI
import logging
from typing import Union
from VLLM_Interface import VLLM_Interface
import numpy as np
from utils import encode_image, load_images
import os
import json
import yaml
import re

# Get a logger for this module
log = logging.getLogger(__name__)

ACT_TO_STR = {
    'OpenObject': "Open",
    'CloseObject': "Close",
    'PickupObject': "Pickup",
    'PutObject': "Put",
    'ToggleObjectOn': "Toggle on",
    'ToggleObjectOff': "Toggle off",
    'SliceObject': "Slice",
    'Navigation': "Navigate"
}

def remove_duplicate_images(encoded_images):
    unique_images = []
    seen = set()
    # Iterate through images backwards to keep later copies of duplicate images
    for img in encoded_images[::-1]:
        if img not in seen:
            seen.add(img)
            unique_images = [img] + unique_images
    return unique_images

class GT_Planner:
    def __init__(self, llm_config: dict, client: OpenAI, debug: bool=False, logger: Union[logging.Logger, None] = None):
        self.config = llm_config
        self.knn_set = pd.read_pickle(self.config["knn_dataset_path"])
        self.debug=debug
        self.logger = logger

    def format_plan_str(self, plan_list):
        if not plan_list:
            return ""

        # Lowercase object names in (action, plan) tuple
        formatted_plans = []
        for item in plan_list:
            if isinstance(item, tuple):
                # Handle tuples (normal case)
                item_list = list(item)
                # Make object names lowercase
                if len(item_list) > 1:
                    item_list[1] = item_list[1].lower()
                if len(item_list) > 2:
                    item_list[2] = item_list[2].lower()
                # Join action and object with a space
                formatted_plans.append(" ".join(item_list))
            else:
                # Handle string items (in case they're already formatted)
                formatted_plans.append(item)

        # Join plans with comma and space
        plan_str = ', '.join(formatted_plans)

        return plan_str


    def get_high_level_plan(self, curr_task, encoded_frames):
        # For GT planner, we just return the gold plan from the knn set for the current task
        task_instr = curr_task["task_instr"][0] if isinstance(curr_task["task_instr"], list) else curr_task["task_instr"]
        knn_set_task_instrs = self.knn_set["task_instr"].tolist()
        # This line will throw a value error if the task is not in the knn set and skip the task
        task_index = knn_set_task_instrs.index(task_instr)
        matched_task = self.knn_set.iloc[task_index]

        gold_plan = matched_task["gold_traj"]
        gold_plan = [literal_eval(listItem) for rowItem in gold_plan for listItem in rowItem]
        plan =  self.format_plan_str(gold_plan)
        self.logger.debug(f"GT Planner found matching task in KNN set: {task_instr}")
        self.logger.debug(f"GT Planner gold plan: {plan}")
        plans = [plan.strip() for plan in plan.split(',')]
        return plans

      
class LLM_Planner():
    def __init__(self, llm_config: dict, client: OpenAI, debug: bool=False, logger: Union[logging.Logger, None] = None):
        print(f"LLM Planner Got logger: {logger}")
        self.config = llm_config
        self.sentence_embedder = SentenceTransformer(self.config["emb_model_name"], device="cuda:1")
        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.knn_set = pd.read_pickle(self.config["knn_dataset_path"])
        self.debug=debug
        self.llm_client = client
        self.engine = self.config["engine"]
        self.logger = logger
        self.llm = VLLM_Interface(client=self.llm_client, engine=self.engine)
        self.num_task_calls = 0


        
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

    # def knn_retrieval(self, curr_task, k):
    #     # Find K train examples with closest sentence embeddings to test example
    #     traj_emb = self.sentence_embedder.encode(curr_task["task_instr"], show_progress_bar=False)
    #     topK = []
    #     for idxTrain, trainItem in self.knn_set.iterrows():

    #         train_emb = self.sentence_embedder.encode(trainItem["task_instr"], show_progress_bar=False)

    #         dist = -1 * cos_sim(traj_emb, train_emb)

    #         if len(topK) < k:
    #             topK.append((trainItem["task"], dist))
    #             topK = sorted(topK, key = lambda x : x[1])
    #         else:
    #             if float(dist) < topK[-1][1]:
    #                 if (trainItem["task"], dist) not in topK:
    #                     topK.append((trainItem["task"], dist))
    #                     topK = sorted(topK, key = lambda x : x[1])
    #                     topK = topK[:k]

    #     return [entry[0] for entry in topK]
    
    def base_retrieval(self, curr_task, k):
        # Find K train examples with closest sentence embeddings to test example
        traj_emb = self.sentence_embedder.encode(curr_task["task_instr"], show_progress_bar=False)
        embeddings = np.array(self.knn_set["embeddings"].tolist(), dtype=np.float32)
        dists = -1 * cos_sim(traj_emb, embeddings)
        dists = dists.numpy().flatten()
        sorted_indices = np.argsort(dists)
        start_index = 0
        if self.config.get("exclude_perfect_matches", False) and dists[sorted_indices[0]] <= -0.999:
            # Skip perfect matches (distance of -1.0)
            while start_index < len(sorted_indices) and dists[sorted_indices[start_index]] <= -0.999:
                start_index += 1
        top_k_indices = np.argsort(dists)[:k]
        top_k_tasks = self.knn_set.iloc[top_k_indices]["task"].tolist()
        return top_k_tasks
    
    def knn_retrieval(self, curr_task, k):
        if self.config.get("use_mmr", False):
            lambda_param = self.config.get("mmr_lambda", 0.5)
            return self.mmr_retrieval(curr_task, k, lambda_param)
        else:
            return self.base_retrieval(curr_task, k)

    def mmr_retrieval(self, curr_task, k, lambda_param=0.5):
        traj_emb = self.sentence_embedder.encode(curr_task["task_instr"], show_progress_bar=False)
        embeddings = np.array(self.knn_set["embeddings"].tolist(), dtype=np.float32)
        task_sims = cos_sim(traj_emb, embeddings).numpy().flatten()
        self_sims = cos_sim(embeddings, embeddings).numpy()
        selected_indices = []
        for i in range(k):
            if i == 0:
                # Select the most similar task for the first example
                selected_index = np.argmax(task_sims)
            else:
                # Compute MMR scores for remaining tasks
                mmr_scores = lambda_param * task_sims - (1 - lambda_param) * np.max(self_sims[selected_indices,:])
                mmr_scores[selected_indices] = -np.inf  # Exclude already selected tasks
                selected_index = np.argmax(mmr_scores)
            selected_indices.append(selected_index)
        top_k_tasks = self.knn_set.iloc[selected_indices]["task"].tolist()
        return top_k_tasks


    def generate_prompt(self, curr_task, k, removeNav=False, naturalFormat=False, includeLow=False, dynamic=True, diverse=False, prev_plans = []):
        log.debug(f"Generating prompt with {k} examples, includeLow={includeLow}, dynamic={dynamic}")
        # Define the valid objects list
        VALID_OBJECTS = [
            'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub',
            'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bowl', 'Box', 'Bread',
            'ButterKnife', 'Cabinet', 'Candle', 'Cart', 'CD', 'CellPhone', 'Chair',
            'Cloth', 'CoffeeMachine', 'CounterTop', 'CreditCard', 'Cup', 'Curtains',
            'Desk', 'DeskLamp', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'FloorLamp',
            'Footstool', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel',
            'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle',
            'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LightSwitch',
            'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan',
            'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow',
            'Plate', 'Plunger', 'Poster', 'Pot', 'Potato', 'RemoteControl', 'Safe',
            'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'ShowerGlass', 'Sink',
            'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle',
            'Statue', 'StoveBurner', 'StoveKnob', 'DiningTable', 'CoffeeTable', 'SideTable',
            'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet',
            'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'Towel',
            'TowelHolder', 'TVStand', 'Vase', 'Watch', 'WateringCan', 'Window', 'WineBottle'
        ]
        
        # Format the valid objects list for the prompt
        valid_objects_str = ", ".join(VALID_OBJECTS)
        
        #header
        prompt = "Create a high-level plan for completing a household task using the allowed actions. Follow the exact output format described in the examples. Only output the next steps of the plan. Always try to navigate to the object before interacting with it. If an object is on the CounterTop, navigate to the object itself rather than the CounterTop."

        if naturalFormat:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.values())}" 
        else:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.keys())}" 
        
        # Add the valid objects constraint
        prompt += f"\n\nIMPORTANT: Only use objects from this list: {valid_objects_str}"
        
        # Run KNN retrieval
        knn_retrieved_examples = self.knn_retrieval(curr_task, k)

        # Add in-context examples from knn retrieval
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]

            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

            #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned
            
            # Split past and next plans randomly
            planSplit = random.sample(range(len(step_list)),1)[0]
            
            # In-context examples components
            high_level_str = str(trainTaskRow["task_instr"])
            step_by_step_str = '. '.join(trainTaskRow["step_instr"])
            past_plan_str = self.format_plan_str(step_list[:planSplit])
            next_plans_str = self.format_plan_str(step_list[planSplit:])
            in_context_obj_str = self.format_object_str(trainTaskRow["vis_objs"])

            # In-context examples
            prompt += "\n\nTask description: " + high_level_str
                
            # Include low-level instructions
            if includeLow:
                prompt += "\nStep by step instructions: " + step_by_step_str

            prompt +=  "\nCompleted plans: " + past_plan_str
            
            # Only include visible objects if dynamic planning is enabled
            if dynamic:
                if not in_context_obj_str == "":
                    prompt += "\nVisible objects are " + in_context_obj_str
            
            prompt += "\nNext Plans: " + next_plans_str
        
        # Add the task prompt for GPT
        ## In-context examples components
        completed_plans = curr_task["completed_plans"]
        vis_objs = curr_task["vis_objs"]
        failed_plans = curr_task.get("failed_plans", [])

        task_high_level_str = str(curr_task["task_instr"][0])
        
        # Fix for step_instr being a list of lists or empty
        if curr_task["step_instr"] and isinstance(curr_task["step_instr"][0], list):
            # Either flatten the list
            step_instr_flat = [step for sublist in curr_task["step_instr"] for step in sublist]
            task_step_by_step_str = '. '.join(step_instr_flat)
        elif curr_task["step_instr"]:
            # Original code if it's already a flat list
            task_step_by_step_str = '. '.join(curr_task["step_instr"])
        else:
            # Handle empty step_instr
            task_step_by_step_str = ""
        
        task_past_plan_str = self.format_plan_str(completed_plans)
        task_obj_str = self.format_object_str(vis_objs)

        if diverse and prev_plans:
            prompt += "\n\nNow generate a diverse set of plans for the following task. Avoid repeating the same plans you generated before. If a plan is partially generated, pickup where the plan left off."
            for i, plan in enumerate(prev_plans):
                prompt += f"\n\nDiverse Plan {i + 1}: "
                prompt += "\nTask description: " + task_high_level_str
                if includeLow:
                    prompt += "\nStep by step instructions: " + task_step_by_step_str
                prompt += "\nCompleted plans: " + task_past_plan_str
                if dynamic:
                    if not task_obj_str == "":
                        prompt += "\nVisible objects are " + task_obj_str

                if failed_plans:
                    last_failed_plan = failed_plans[-1]
                    prompt += "\nFailed plans: " + self.format_plan_str([last_failed_plan["plan"]]) +" with error: " + last_failed_plan["error"]
                prompt += "\nNext Plans: " + self.format_plan_str(plan)

        if not diverse:
            prompt += "\n\nTask description: " + task_high_level_str
        else:
            prompt += f"\n\nDiverse plan {len(prev_plans) + 1}: "
            prompt += "\nTask description: " + task_high_level_str

        if includeLow:
            prompt += "\nStep by step instructions: " + task_step_by_step_str

        prompt += "\nCompleted plans: " + task_past_plan_str
        
        # Only include visible objects if dynamic planning is enabled
        if dynamic:
            if not task_obj_str == "":
                prompt += "\nVisible objects are " + task_obj_str

        if failed_plans:
            last_failed_plan = failed_plans[-1]
            prompt += "\nFailed plans: " + self.format_plan_str([last_failed_plan["plan"]]) +" with error: " + last_failed_plan["error"]

        prompt += "\nNext Plans: "
            
        curr_task["Prompts"] = prompt
        
        # Only store vis_objs if dynamic planning is enabled
        if dynamic:
            curr_task["vis_objs"] = vis_objs
        
        log.debug("Prompt generation complete")
        return prompt

    # Main point of entry for LLM HLP prompt generator, use in run_eval
    def generate_gpt_prompt(self, curr_task, k, includeLow=False, dynamic=True, diverse=False, prev_plans = []):
        prompt = self.generate_prompt(curr_task, k, removeNav=False, naturalFormat=False, includeLow=includeLow, dynamic=dynamic, diverse=diverse, prev_plans=prev_plans)
        return prompt

    
    # Below are helper functions 

    # Change object list into object string:
    ## Example: ['Drawer', 'ButterKnife'] -> Drawer, ButterKnife
    def format_object_str(self, obj_list):
        """
        Format a list of objects or a comma-separated string into a properly formatted string.
        
        Args:
            obj_list: Either a list of object names or a comma-separated string of object names
            
        Returns:
            A properly formatted comma-separated string of object names
        """
        # If input is already a string, split it into a list
        if isinstance(obj_list, str):
            # Split by comma and strip whitespace
            obj_list = [obj.strip() for obj in obj_list.split(',')]
            # Filter out empty strings
            obj_list = [obj for obj in obj_list if obj]
        
        # If the list is empty, return an empty string
        if not obj_list:
            return ""
        
        # Join the list with commas and spaces
        obj_str = ", ".join(obj_list)
        
        return obj_str

    # Change plan list into plan string:
    ## Example: [('Navigation','Shelf'), ('PickupObject', 'knife')] -> Navigation Shelf, PickupObject Knife
    def format_plan_str(self, plan_list):
        if not plan_list:
            return ""

        # Lowercase object names in (action, plan) tuple
        formatted_plans = []
        for item in plan_list:
            if isinstance(item, tuple):
                # Handle tuples (normal case)
                item_list = list(item)
                # Make object names lowercase
                if len(item_list) > 1:
                    item_list[1] = item_list[1].lower()
                if len(item_list) > 2:
                    item_list[2] = item_list[2].lower()
                # Join action and object with a space
                formatted_plans.append(" ".join(item_list))
            else:
                # Handle string items (in case they're already formatted)
                formatted_plans.append(item)

        # Join plans with comma and space
        plan_str = ', '.join(formatted_plans)

        return plan_str


    def generate_high_level_plan(self, curr_task, encoded_frames, n_plans = 1):
        prompt = self.generate_gpt_prompt(
            curr_task, 
            k=self.config["num_in_context_examples"], 
            includeLow=self.config["use_step_instructions"], 
            dynamic=self.config["dynamic"]
            )
        if self.config.get("debug", False):
            self.logger.debug("=" * 50)
            self.logger.debug("FULL PROMPT:")
            self.logger.debug(prompt)
            self.logger.debug("=" * 50)

        llm_out = self.llm.generate(prompt, images=encoded_frames, temperature=0.0)[0]
        high_level_plans = [self.clean_llm_output(llm_out)]

        if self.config.get("debug", False):
            self.logger.debug("=" * 50)
            self.logger.debug("GENERATED PLAN:")
            self.logger.debug(high_level_plans)
            self.logger.debug("=" * 50)
        
        if n_plans > 1:
            prev_plans = high_level_plans[:]
            for i in range(1, n_plans):
                prompt = self.generate_gpt_prompt(
                    curr_task, 
                    k=self.config["num_in_context_examples"], 
                    includeLow=self.config["use_step_instructions"], 
                    dynamic=self.config["dynamic"],
                    diverse=True,
                    prev_plans=prev_plans
                )

                llm_out = self.llm.generate(prompt, images=encoded_frames, temperature=0.0)[0]
                new_plan = self.clean_llm_output(llm_out)
                high_level_plans.append(new_plan)
                prev_plans.append(new_plan)
    

        return high_level_plans
    
    def get_high_level_plan(self, curr_task, encoded_frames):
        """
        Main point of entry for getting high-level plan for the current task, use in run_eval
        """
        return self.generate_high_level_plan(curr_task, encoded_frames, 1)[0]

def print_tree(plan_tree, indent=0, print_function=print):
    for plan, subplans in plan_tree.items():
        print_function(' ' * indent + str(plan))
        if isinstance(subplans, dict):
            print_tree(subplans, indent + 4, print_function)

class Tree_Search_LLM_Planner(LLM_Planner):
    def __init__(self, llm_config: dict, client: OpenAI, debug: bool=False, logger: Union[logging.Logger, None] = None):
        print(f"Tree Search Got logger: {logger}")
        super().__init__(llm_config, client, debug, logger)
        self.mode = self.config.get("mode", "action_prob")

    def generate_action_probs_prompt(self, curr_task, k, plan_steps, plan_options, removeNav=False, naturalFormat=False, includeLow=False, dynamic=True, additional_info=[]):
        log.debug(f"Generating prompt with {k} examples, includeLow={includeLow}, dynamic={dynamic}")
        # Define the valid objects list
        VALID_OBJECTS = [
            'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub',
            'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bowl', 'Box', 'Bread',
            'ButterKnife', 'Cabinet', 'Candle', 'Cart', 'CD', 'CellPhone', 'Chair',
            'Cloth', 'CoffeeMachine', 'CounterTop', 'CreditCard', 'Cup', 'Curtains',
            'Desk', 'DeskLamp', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'FloorLamp',
            'Footstool', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel',
            'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle',
            'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LightSwitch',
            'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan',
            'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow',
            'Plate', 'Plunger', 'Poster', 'Pot', 'Potato', 'RemoteControl', 'Safe',
            'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'ShowerGlass', 'Sink',
            'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle',
            'Statue', 'StoveBurner', 'StoveKnob', 'DiningTable', 'CoffeeTable', 'SideTable',
            'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet',
            'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'Towel',
            'TowelHolder', 'TVStand', 'Vase', 'Watch', 'WateringCan', 'Window', 'WineBottle'
        ]
        LETTERS = ['A', 'B', 'C', 'D', 'E']
        
        # Format the valid objects list for the prompt
        valid_objects_str = ", ".join(VALID_OBJECTS)
        
        #header
        prompt = "Create a high-level plan for completing a household task using the allowed actions. Select the next best action from the provided options. Always try to navigate to the object before interacting with it. If an object is on the CounterTop, navigate to the object itself rather than the CounterTop."
        
        if naturalFormat:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.values())}" 
        else:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.keys())}" 
        
        # Add the valid objects constraint
        prompt += f"\n\nIMPORTANT: Only use objects from this list: {valid_objects_str}"
        
        # Run KNN retrieval
        knn_retrieved_examples = self.knn_retrieval(curr_task, k)

        all_steps = []
        # Get all steps from the KNN examples
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]

            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

             #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned

            all_steps.extend(step_list)

        # Add in-context examples from knn retrieval
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]

            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

            #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned
            
            # Split past and next plans randomly
            planSplit = random.sample(range(len(step_list)),1)[0]
            decisionSplit = random.sample(range(planSplit,len(step_list)),1)[0]
            
            # In-context examples components
            high_level_str = str(trainTaskRow["task_instr"])
            step_by_step_str = '. '.join(trainTaskRow["step_instr"])
            past_plan_str = self.format_plan_str(step_list[:planSplit])
            next_plans_str = self.format_plan_str(step_list[planSplit:decisionSplit])
            gt_next_plan = step_list[decisionSplit]
            num_options = random.randint(2, 5)
            options = [gt_next_plan]
            for i in range(num_options - 1):
                random_step = random.choice(all_steps)
                while random_step in options:
                    random_step = random.choice(all_steps)
                options.append(random_step)
            random.shuffle(options)
            gt_letter = LETTERS[options.index(gt_next_plan)]
            options_str = "\n".join([f"{LETTERS[i]}. {' '.join(option)}" for i, option in enumerate(options)])
            

            in_context_obj_str = self.format_object_str(trainTaskRow["vis_objs"])

            # In-context examples
            prompt += "\n\nTask description: " + high_level_str
                
            # Include low-level instructions
            if includeLow:
                prompt += "\nStep by step instructions: " + step_by_step_str

            prompt +=  "\nCompleted plans: " + past_plan_str
            
            # Only include visible objects if dynamic planning is enabled
            if dynamic:
                if not in_context_obj_str == "":
                    prompt += "\nVisible objects are " + in_context_obj_str
            
            prompt += "\nNext Plans: " + next_plans_str
            prompt += "\nOptions:\n" + options_str
            prompt += f"\nAnswer: {gt_letter}"
        
        # Add the task prompt for GPT
        ## In-context examples components
        completed_plans = curr_task["completed_plans"]
        vis_objs = curr_task["vis_objs"]
        failed_plans = curr_task.get("failed_plans", [])

        task_high_level_str = str(curr_task["task_instr"][0])
        
        # Fix for step_instr being a list of lists or empty
        if curr_task["step_instr"] and isinstance(curr_task["step_instr"][0], list):
            # Either flatten the list
            step_instr_flat = [step for sublist in curr_task["step_instr"] for step in sublist]
            task_step_by_step_str = '. '.join(step_instr_flat)
        elif curr_task["step_instr"]:
            # Original code if it's already a flat list
            task_step_by_step_str = '. '.join(curr_task["step_instr"])
        else:
            # Handle empty step_instr
            task_step_by_step_str = ""
        
        task_past_plan_str = self.format_plan_str(completed_plans)
        task_obj_str = self.format_object_str(vis_objs)

       
        prompt += "\n\nTask description: " + task_high_level_str
       
        if includeLow:
            prompt += "\nStep by step instructions: " + task_step_by_step_str

        prompt += "\nCompleted plans: " + task_past_plan_str
        
        # Only include visible objects if dynamic planning is enabled
        if dynamic:
            if not task_obj_str == "":
                prompt += "\nVisible objects are " + task_obj_str

        if failed_plans:
            last_failed_plan = failed_plans[-1]
            prompt += "\nFailed plans: " + self.format_plan_str([last_failed_plan["plan"]]) +" with error: " + last_failed_plan["error"]


        prompt += "\nNext Plans: "

        prompt += self.format_plan_str(plan_steps)
        prompt += "\nOptions:\n" + "\n".join([f"{LETTERS[i]}. {option}" for i, option in enumerate(plan_options)])
        if additional_info:
            prompt += "\n" + "\n".join(additional_info)
        prompt += "\nAnswer:"
            
        curr_task["Prompts"] = prompt
        
        # Only store vis_objs if dynamic planning is enabled
        if dynamic:
            curr_task["vis_objs"] = vis_objs
        
        log.debug("Prompt generation complete")
        return prompt
    
    def generate_transition_probs_prompt(self, curr_task, k, plan_steps, future_plans, removeNav=False, naturalFormat=False, includeLow=False, dynamic=True, additional_info=[]):
        log.debug(f"Generating prompt with {k} examples, includeLow={includeLow}, dynamic={dynamic}")
        # Define the valid objects list
        VALID_OBJECTS = [
            'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub',
            'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bowl', 'Box', 'Bread',
            'ButterKnife', 'Cabinet', 'Candle', 'Cart', 'CD', 'CellPhone', 'Chair',
            'Cloth', 'CoffeeMachine', 'CounterTop', 'CreditCard', 'Cup', 'Curtains',
            'Desk', 'DeskLamp', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'FloorLamp',
            'Footstool', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel',
            'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle',
            'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LightSwitch',
            'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan',
            'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow',
            'Plate', 'Plunger', 'Poster', 'Pot', 'Potato', 'RemoteControl', 'Safe',
            'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'ShowerGlass', 'Sink',
            'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle',
            'Statue', 'StoveBurner', 'StoveKnob', 'DiningTable', 'CoffeeTable', 'SideTable',
            'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet',
            'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'Towel',
            'TowelHolder', 'TVStand', 'Vase', 'Watch', 'WateringCan', 'Window', 'WineBottle'
        ]
        # LETTERS = ['A', 'B', 'C', 'D', 'E']
        
        # Format the valid objects list for the prompt
        valid_objects_str = ", ".join(VALID_OBJECTS)
        
        #header
        prompt = "You are helping to create a high-level plan for completing a household task. Your job is to predict if a sequence of actions will be successful. Sequences may be unsuccessful for a multitude of reasons, including but not limited to:"
        prompt += "\n- The plan did not navigate to the object before interacting with it."
        prompt += "\n- The plan tried to put an object in a container that is not open."
        prompt += "\n- The plan tried to wash or clean a dirty object without putting it in the sink first."
        prompt += "\n- The plan tried to turn on the microwave without closing it first."
        prompt += "\n- The plan tried to navigate to the CounterTop instead of the object itself (the countertop is very large and objects on it are typically not reachable from the countertop itself)"
        prompt += "\n- The plan tried to put down an object without picking it up first."
        prompt += "\n- The plan tried to pickup an object while already holding another object."
        prompt += "\n\n For the following sequence of actions, answer 'Yes' if the sequence will be successful, and 'No' if the sequence will be unsuccessful."
        
        # if naturalFormat:
        #     prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.values())}" 
        # else:
        #     prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.keys())}" 
        
        # # Add the valid objects constraint
        # prompt += f"\n\nIMPORTANT: Only use objects from this list: {valid_objects_str}"
        
        # Run KNN retrieval
        knn_retrieved_examples = self.knn_retrieval(curr_task, k)

        all_steps = []
        # Get all steps from the KNN examples
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]

            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

             #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned

            all_steps.extend(step_list)

        # Add in-context examples from knn retrieval
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]

            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

            #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned
            
            # Split past and next plans randomly
            planSplit = random.sample(range(len(step_list)),1)[0]
            
            # In-context examples components
            high_level_str = str(trainTaskRow["task_instr"])
            step_by_step_str = '. '.join(trainTaskRow["step_instr"])
            past_plan_str = self.format_plan_str(step_list[:planSplit])
            next_plans_list = step_list[planSplit:]
            # Decide whether or not to perturb the plan to generate a failed example
            perturb_plan = random.random() < 0.5
            if perturb_plan:
                # Possible perturbations: Remove a navigation step, remove a pickup step before a putdown step, replace a step with a random step
                possible_perturbations = ["random_replace"]
                # Check if the plan contains a pickup step before a putdown step
                if any("PickupObject" in step for step in next_plans_list) and any("PutObject" in step for step in next_plans_list):
                    possible_perturbations.append("remove_pickup_before_putdown")
                # Check if the plan contains a navigation step
                if any("Navigation" in step for step in next_plans_list):
                    possible_perturbations.append("remove_navigation")
                # Randomly choose a perturbation
                perturbation = random.choice(possible_perturbations)
                if perturbation == "random_replace":
                    # Replace a step with a random step
                    replace_index = random.sample(range(len(next_plans_list)),1)[0]
                    next_plans_list[replace_index] = random.choice(all_steps)
                elif perturbation == "remove_pickup_before_putdown":
                    # Remove a single pickup step before a putdown step
                    pickup_index = ["PickupObject" in step for step in next_plans_list].index(True)
                    next_plans_list.pop(pickup_index)
                elif perturbation == "remove_navigation":
                    # Remove a navigation step
                    navigation_index = ["Navigation" in step for step in next_plans_list].index(True)
                    next_plans_list.pop(navigation_index)

            # Format the next plans list for the prompt
            next_plans_str = self.format_plan_str(next_plans_list)
            in_context_obj_str = self.format_object_str(trainTaskRow["vis_objs"])

            # In-context examples
            prompt += "\n\nTask description: " + high_level_str
                
            # Include low-level instructions
            if includeLow:
                prompt += "\nStep by step instructions: " + step_by_step_str

            prompt +=  "\nCompleted plans: " + past_plan_str
            
            # Only include visible objects if dynamic planning is enabled
            if dynamic:
                if not in_context_obj_str == "":
                    prompt += "\nVisible objects are " + in_context_obj_str
            
            prompt += "\nNext Plans: " + next_plans_str
            prompt += f"\nWill the next plans be successful?"
            prompt += f"\nAnswer: {'No' if perturb_plan else 'Yes'}"
        
        # Add the task prompt for GPT
        ## In-context examples components
        completed_plans = curr_task["completed_plans"]
        vis_objs = curr_task["vis_objs"]
        failed_plans = curr_task.get("failed_plans", [])

        task_high_level_str = str(curr_task["task_instr"][0])
        
        # Fix for step_instr being a list of lists or empty
        if curr_task["step_instr"] and isinstance(curr_task["step_instr"][0], list):
            # Either flatten the list
            step_instr_flat = [step for sublist in curr_task["step_instr"] for step in sublist]
            task_step_by_step_str = '. '.join(step_instr_flat)
        elif curr_task["step_instr"]:
            # Original code if it's already a flat list
            task_step_by_step_str = '. '.join(curr_task["step_instr"])
        else:
            # Handle empty step_instr
            task_step_by_step_str = ""
        
        task_past_plan_str = self.format_plan_str(completed_plans)
        task_obj_str = self.format_object_str(vis_objs)

       
        prompt += "\n\nTask description: " + task_high_level_str
       
        if includeLow:
            prompt += "\nStep by step instructions: " + task_step_by_step_str

        prompt += "\nCompleted plans: " + task_past_plan_str
        
        # Only include visible objects if dynamic planning is enabled
        if dynamic:
            if not task_obj_str == "":
                prompt += "\nVisible objects are " + task_obj_str

        if failed_plans:
            last_failed_plan = failed_plans[-1]
            prompt += "\nFailed plans: " + self.format_plan_str([last_failed_plan["plan"]]) +" with error: " + last_failed_plan["error"]


        prompt += "\nNext Plans: "

        prompt += self.format_plan_str(plan_steps + future_plans)
        if additional_info:
            prompt += "\n" + "\n".join(additional_info)
        prompt += f"\nWill the next plans be successful?"
        prompt += f"\nAnswer: "
            
        curr_task["Prompts"] = prompt
        
        # Only store vis_objs if dynamic planning is enabled
        if dynamic:
            curr_task["vis_objs"] = vis_objs
        
        log.debug("Prompt generation complete")
        return prompt
    
    def generate_gpt_action_prob_prompt(self, curr_task, plan_steps, plan_options, includeLow=False, dynamic=True, additional_info = []):
        prompt = self.generate_action_probs_prompt(
            curr_task, 
            k=self.config["num_in_context_examples"], 
            plan_steps=plan_steps, 
            plan_options=plan_options, 
            includeLow=includeLow, 
            dynamic=dynamic,
            additional_info=additional_info
            )
        return prompt
    
    def generate_gpt_transition_prob_prompt(self, curr_task, plan_steps, future_plans, includeLow=False, dynamic=True, additional_info = []):
        prompt = self.generate_transition_probs_prompt(
            curr_task, 
            k=self.config["num_in_context_examples"], 
            plan_steps=plan_steps, 
            future_plans=future_plans, 
            includeLow=includeLow, 
            dynamic=dynamic,
            additional_info=additional_info
            )
        return prompt

    def match_tokens_to_plan_steps(self, plan_tokens, cleaned_plan):
        token_index_assignment = []
        token_index = 0
        for step in cleaned_plan:
            assigned_tokens = []
            step_words = step.split()
            for word in step_words:
                word_tokens = []
                while word and token_index < len(plan_tokens):
                    token = plan_tokens[token_index]
                    # Check token versions  in the following order: exact match, rstrip, lstrip, strip
                    token_modifications = [token, token.rstrip(), token.lstrip(), token.strip()]
                    for token_variant in token_modifications:
                        if word.startswith(token_variant):
                            word_tokens.append(token_index)
                            # Remove the matched portion from the step and continue matching the rest of the step with the next tokens
                            word = word[len(token_variant):]
                            break
                    token_index += 1
                assigned_tokens.append(word_tokens)
            token_index_assignment.append(assigned_tokens)
        return token_index_assignment
                

    
    def generate_diverse_plans(self, curr_task, encoded_frames, n_plans = 5, word_sim_thresh = 0.7):
        prompt = self.generate_gpt_prompt(
            curr_task, 
            k=self.config["num_in_context_examples"], 
            includeLow=self.config["use_step_instructions"], 
            dynamic=self.config["dynamic"]
            )
        if self.config.get("debug", False):
            self.logger.debug("=" * 50)
            self.logger.debug("FULL PROMPT:")
            self.logger.debug(prompt)
            self.logger.debug("=" * 50)

        llm_out, logprobs = self.llm.generate(prompt, images=encoded_frames, temperature=0.0, return_logprobs=True, num_logprobs=10)
        llm_out, logprobs = llm_out[0], logprobs[0]
        high_level_plans = [self.clean_llm_output(llm_out)]

        if self.config.get("debug", False):
            self.logger.debug("=" * 50)
            self.logger.debug("GENERATED PLAN:")
            self.logger.debug(high_level_plans)
            self.logger.debug("=" * 50)
        
        for i in range(1, n_plans):
            prompt = self.generate_gpt_prompt(
                curr_task, 
                k=self.config["num_in_context_examples"], 
                includeLow=self.config["use_step_instructions"], 
                dynamic=self.config["dynamic"],
                diverse=True,
                prev_plans=high_level_plans
            )

            llm_out, logprobs = self.llm.generate(prompt, images=encoded_frames, temperature=0.0, return_logprobs=True, num_logprobs=10)
            llm_out, logprobs = llm_out[0], logprobs[0]
            new_plan = self.clean_llm_output(llm_out)
            if new_plan in high_level_plans:
                # The new plan is a duplicate so we have to force the LLM to generate an alternate plan
                # We do this by finding the most likely token not already in the plan and setting it as the next token
                if self.config.get("debug", False):
                    self.logger.debug("Forcing diversity due to duplicate plan generation")

                logprobs = logprobs  # Exclude the last token (EOS)
                plan_tokens = [prob.token for prob in logprobs]
                token_assignments = self.match_tokens_to_plan_steps(plan_tokens, new_plan)
                word_token_assignments = [word for step in token_assignments for word in step]
                first_token_indicies = [tokens[0] for tokens in word_token_assignments]  # Get the first token index for each word
                flattened_token_indicies = [token for sublist in word_token_assignments for token in sublist]
                
                # Create a list of tokens and their probabilities for each step
                plan_token_logprobs = []
                for token in logprobs:
                    step_tokens = {}
                    for logprob_token in token.top_logprobs:
                        step_tokens[logprob_token.token] = logprob_token.logprob
                    plan_token_logprobs.append(step_tokens)
                
                # Suppress any tokens that arent the first token in a word (we dont want to branch mid word) or tokens that are not english letters or commas by setting their logprobs to -inf
                for i in range(len(plan_token_logprobs)):
                    if i not in first_token_indicies:
                        for token in plan_token_logprobs[i]:
                            plan_token_logprobs[i][token] = float("-inf")
                    else:
                        for token in plan_token_logprobs[i]:
                            if not token.isascii():
                                plan_token_logprobs[i][token] = float("-inf")
                
                new_plan_words = " ".join(new_plan).split(" ")

                # Iterate through the previous plans and set the logprobs associated with words in the previous plans to -inf
                for prev_plan in high_level_plans:

                    prev_plan_words = " ".join(prev_plan).split(" ")
                    for word_idx in range(min(len(prev_plan_words), len(new_plan_words))):
                        # Check if the plan stems match between the old and new plan up to the current word. If they dont match, we can stop suppressing tokens for this plan since the plans have diverged. If they do match, we suppress the tokens associated with this word in the new plan to encourage the LLM to generate a different word at this position.
                        if prev_plan_words[:word_idx] == new_plan_words[:word_idx]:
                            # Get the tokens associated with this word
                            token_indexes = word_token_assignments[word_idx]
                            # Get the previous word
                            prev_word = prev_plan_words[word_idx]
                            
                            word_first_token_probs = plan_token_logprobs[token_indexes[0]]
                            word_first_tokens = list(word_first_token_probs.keys())
                            # Because we dont explicitly save previous plan tokens, we have to guess at the first token of the previous word
                            # Get the highest logprob token
                            highest_prob_token = max(word_first_tokens, key=word_first_token_probs.get).strip()

                            # If the highest prob token is a substring of the previous word, we can be reasonably confident that it is the first token of the previous word and we can use it for similarity comparison.
                            if prev_word.startswith(highest_prob_token):
                                prev_word = highest_prob_token

                            # Suppress similar first tokens by setting their logprobs to -inf
                            prev_word_embedding = self.sentence_embedder.encode(prev_word, show_progress_bar=False)
                            word_first_token_embeddings = self.sentence_embedder.encode(word_first_tokens, show_progress_bar=False)
                            similarities = cos_sim(prev_word_embedding, word_first_token_embeddings).numpy()[0]
                            similar_tokens = [word_first_tokens[i] for i in range(len(word_first_tokens)) if similarities[i] > word_sim_thresh]
                            for token in similar_tokens:
                                plan_token_logprobs[token_indexes[0]][token] = float("-inf")
                            
                        else:
                            break

                    # diverged = False
                    # # For each step in the previous plan, check if the new plan has the same step. If it does, suppress the tokens associated with that step. If it doesn't, move on to the next step and repeat until we find a step that is different between the two plans, at which point we stop suppressing tokens for that plan since the plans have diverged.
                    # for step_idx in range(len(token_assignments)):
                    #     if step_idx >= len(prev_plan):
                    #         diverged = True
                    #         break
                    #     prev_plan_step = prev_plan[step_idx]
                    #     new_plan_step = new_plan[step_idx]
                    #     prev_plan_step_words = prev_plan_step.split()
                    #     new_plan_step_words = new_plan_step.split()

                    #     for word_idx in range(len(new_plan_step_words)):
                    #         # Check if the previous words match between the old and new plan up to the current word. If they dont match, we can stop suppressing tokens for this plan since the plans have diverged. If they do match, we suppress the tokens associated with this word in the new plan to encourage the LLM to generate a different word at this position.
                    #         if prev_plan_step_words[:word_idx] != new_plan_step_words[:word_idx]:
                    #             diverged = True
                    #             break
                    #         else:
                    #             # Get the tokens associated with this word
                    #             token_indexes = token_assignments[step_idx][word_idx]
                    #             # Suppress similar first tokens by setting their logprobs to -inf
                    #             prev_word = prev_plan_step_words[word_idx]
                                

                    #             word_first_token_probs = plan_token_logprobs[token_indexes[0]]
                    #             word_first_tokens = list(word_first_token_probs.keys())
                    #             # Because we dont explicitly save previous plan tokens, we have to guess at the first token of the previous word
                    #             # Get the highest logprob token
                    #             highest_prob_token = max(word_first_tokens, key=word_first_token_probs.get).strip()

                    #             # If the highest prob token is a substring of the previous word, we can be reasonably confident that it is the first token of the previous word and we can use it for similarity comparison.
                    #             if prev_word.startswith(highest_prob_token):
                    #                 prev_word = highest_prob_token

                    #             prev_word_embedding = self.sentence_embedder.encode(prev_word, show_progress_bar=False)
                    #             word_first_token_embeddings = self.sentence_embedder.encode(word_first_tokens, show_progress_bar=False)
                    #             similarities = cos_sim(prev_word_embedding, word_first_token_embeddings).numpy()[0]
                    #             similar_tokens = [word_first_tokens[i] for i in range(len(word_first_tokens)) if similarities[i] > word_sim_thresh]
                    #             for token in similar_tokens:
                    #                 plan_token_logprobs[token_indexes[0]][token] = float("-inf")




                    #     if diverged:
                    #         break
                
                
                # Collect the set of most likely tokens after previous tokens have been surpressed
                most_likely_tokens = []
                most_likely_logprobs = []
                for step_tokens in plan_token_logprobs:
                    most_likely_token = max(step_tokens, key=step_tokens.get)
                    most_likely_logprob = step_tokens[most_likely_token]
                    most_likely_tokens.append(most_likely_token)
                    most_likely_logprobs.append(most_likely_logprob)
                
                # Find the most likely alternate token and collect the plan up to that point
                best_token_index = np.argmax(most_likely_logprobs)
                guidance = most_likely_tokens[best_token_index]
                best_token_word = [best_token_index in word_tokens for word_tokens in word_token_assignments].index(True)
                token_after_word = word_token_assignments[best_token_word][-1] + 1
                if "," in plan_tokens[token_after_word]:
                    guidance += ", "
                plan_stub = "".join(plan_tokens[:best_token_index]) + guidance
                if self.config.get("debug", False):
                    self.logger.debug("=" * 50)
                    self.logger.debug("PLAN STUB:")
                    self.logger.debug("".join(plan_tokens[:best_token_index]))
                    self.logger.debug(f"Most likely alternate token: {most_likely_tokens[best_token_index]} with logprob {most_likely_logprobs[best_token_index]:.4f}")
                    self.logger.debug("NEXT PLAN TOKEN")
                    self.logger.debug(plan_tokens[token_after_word])
                    self.logger.debug("=" * 50)
                prompt += plan_stub

                # Try to finish the plan with no generation prompt (this makes the llm better respect the plan stub)
                llm_out = self.llm.generate(prompt, images=encoded_frames, temperature=0.0, add_generation_prompt=False, continue_final_message=True)[0]
                # Sometimes generating without the generation prompt returns nothing, if so, run normal generation
                if not llm_out:
                    llm_out = self.llm.generate(prompt, images=encoded_frames, temperature=0.0)[0]

                if llm_out.startswith(guidance.strip()):
                    llm_out = llm_out[len(guidance.strip()):]
                if self.config.get("debug", False):
                    self.logger.debug("=" * 50)
                    self.logger.debug("PLAN STUB:")
                    self.logger.debug(plan_stub)
                    self.logger.debug("LLM OUTPUT:")
                    self.logger.debug(llm_out)
                    self.logger.debug("=" * 50)
                new_plan = self.clean_llm_output(plan_stub + llm_out)

            high_level_plans.append(new_plan)
    

        return high_level_plans

    def generate_plan_tree(self, curr_task, encoded_frames, n_plans=5):
        """
        Generate a tree of high-level plans for exploration. Each plan is generated with the same prompt but with a diversity constraint to encourage different plans.
        """
        diverse_plans = self.generate_diverse_plans(curr_task, encoded_frames, n_plans=n_plans)
        plan_tree = {}
        for plan in diverse_plans:
            current_level = plan_tree
            for i in range(len(plan)):
                if plan[i] not in current_level:
                    current_level[plan[i]] = {}
                current_level = current_level[plan[i]]
        return plan_tree
    
    def action_probs_tree_search(self, curr_task, encoded_frames):
        plan_tree = self.generate_plan_tree(curr_task, encoded_frames, n_plans=self.config.get("num_diverse_plans", 5))
        if self.config.get("debug", False):
            self.logger.debug("=" * 50)
            self.logger.debug("PLAN TREE:")
            print_tree(plan_tree, print_function=self.logger.debug)
            self.logger.debug("=" * 50)
        selected_steps = []
        current_level = plan_tree
        while len(current_level) > 0:
            if len(current_level) == 1:
                next_step = list(current_level.keys())[0]
                selected_steps.append(next_step)
                current_level = current_level[next_step]
            elif len(current_level) > 1:
                plan_options = list(current_level.keys())
                action_prob_prompt = self.generate_gpt_action_prob_prompt(
                    curr_task, 
                    selected_steps, 
                    plan_options, 
                    includeLow=self.config["use_step_instructions"], 
                    dynamic=self.config["dynamic"]
                    )
                if self.config.get("debug", False):
                    self.logger.debug("=" * 50)
                    self.logger.debug("ACTION PROB PROMPT:")
                    self.logger.debug(action_prob_prompt)
                    self.logger.debug("=" * 50)
                action_probs = self.llm.MCQ_probabilities(action_prob_prompt, images=encoded_frames)[:len(plan_options)]
                action_probs /= np.sum(action_probs)
                if self.config.get("debug", False):
                    self.logger.debug("=" * 50)
                    self.logger.debug("ACTION PROBABILITIES:")
                    for i in range(len(plan_options)):
                        self.logger.debug(f"{plan_options[i]}: {action_probs[i]:.4f}")
                    self.logger.debug("=" * 50)

                selected_option = plan_options[np.argmax(action_probs)]
                selected_steps.append(selected_option)
                current_level = current_level[selected_option]
        return selected_steps
    
    def transition_probs_tree_search(self, curr_task, encoded_frames):
        plan_tree = self.generate_plan_tree(curr_task, encoded_frames, n_plans=self.config.get("num_diverse_plans", 5))
        if self.config.get("debug", False):
            self.logger.debug("=" * 50)
            self.logger.debug("PLAN TREE:")
            print_tree(plan_tree, print_function=self.logger.debug)
            self.logger.debug("=" * 50)
        selected_steps = []
        current_level = plan_tree
        while len(current_level) > 0:
            if len(current_level) == 1:
                next_step = list(current_level.keys())[0]
                selected_steps.append(next_step)
                current_level = current_level[next_step]
            elif len(current_level) > 1:
                plan_options = list(current_level.keys())
                transition_probs = []
                for plan_option in plan_options:
                    # Collect the set of steps from this step untill the next branching point or the end of the plan
                    future_plans = [plan_option]
                    future_level = current_level[plan_option]
                    while len(future_level) > 0:
                        if len(future_level) == 1:
                            next_step = list(future_level.keys())[0]
                            future_plans.append(next_step)
                            future_level = future_level[next_step]
                        elif len(future_level) > 1:
                            break

                    transition_prob_prompt = self.generate_gpt_transition_prob_prompt(
                        curr_task, 
                        selected_steps, 
                        future_plans, 
                        includeLow=self.config["use_step_instructions"], 
                        dynamic=self.config["dynamic"]
                        )
                    if self.config.get("debug", False):
                        self.logger.debug("=" * 50)
                        self.logger.debug("TRANSITION PROB PROMPT:")
                        self.logger.debug(transition_prob_prompt)
                        self.logger.debug("=" * 50)
                    transition_probs.append(self.llm.yes_no_probabilities(transition_prob_prompt, images=encoded_frames)[0])
                if self.config.get("debug", False):
                    self.logger.debug("=" * 50)
                    self.logger.debug("TRANSITION PROBABILITIES:")
                    for i in range(len(plan_options)):
                        self.logger.debug(f"Step: {plan_options[i]}, Transition Probability: {transition_probs[i]:.4f}")
                    self.logger.debug("=" * 50)

                selected_option = plan_options[np.argmax(transition_probs)]
                selected_steps.append(selected_option)
                current_level = current_level[selected_option]
        return selected_steps
        
    
    def get_high_level_plan(self, curr_task, encoded_frames):
        """
        This planner returns a plan by generating a tree of possibilities and searching through it.
        """
        # print(f"Num Frames: {len(encoded_frames)}")
        if "failed_plans" in curr_task:
            self.logger.debug("In dynamic replanning mode, using action probabilities tree search to select next plan.")
            if self.mode == "action_prob":
                return self.action_probs_tree_search(curr_task, encoded_frames)
            elif self.mode == "transition_prob":
                return self.transition_probs_tree_search(curr_task, encoded_frames)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            return self.generate_high_level_plan(curr_task, encoded_frames, 1)[0]

def calculate_entropy(probabilities):
    """
    Calculate the entropy of a probability distribution.
    """
    return -np.sum(probabilities * np.log(probabilities + 1e-10))

class Entropy_reduction_LLM_Planner(Tree_Search_LLM_Planner):
    def __init__(self, llm_config: dict, client: OpenAI, debug: bool=False, logger: Union[logging.Logger, None] = None):
        print(f"Entropy Reduction Got logger: {logger}")
        super().__init__(llm_config, client, debug, logger)
        self.confidence_threshold = llm_config.get("confidence_threshold", 0.9)
        self.max_question_iterations = llm_config.get("max_question_iterations", 5)

    def generate_question_prompt(self, curr_task, k, plan_steps, plan_options, removeNav=False, naturalFormat=False, includeLow=False, dynamic=True, additional_info=[]):
        log.debug(f"Generating prompt with {k} examples, includeLow={includeLow}, dynamic={dynamic}")
        # Define the valid objects list
        VALID_OBJECTS = [
            'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub',
            'BathtubBasin', 'Bed', 'Blinds', 'Book', 'Boots', 'Bowl', 'Box', 'Bread',
            'ButterKnife', 'Cabinet', 'Candle', 'Cart', 'CD', 'CellPhone', 'Chair',
            'Cloth', 'CoffeeMachine', 'CounterTop', 'CreditCard', 'Cup', 'Curtains',
            'Desk', 'DeskLamp', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'FloorLamp',
            'Footstool', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel',
            'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle',
            'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LightSwitch',
            'Microwave', 'Mirror', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan',
            'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow',
            'Plate', 'Plunger', 'Poster', 'Pot', 'Potato', 'RemoteControl', 'Safe',
            'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'ShowerGlass', 'Sink',
            'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle',
            'Statue', 'StoveBurner', 'StoveKnob', 'DiningTable', 'CoffeeTable', 'SideTable',
            'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet',
            'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'Towel',
            'TowelHolder', 'TVStand', 'Vase', 'Watch', 'WateringCan', 'Window', 'WineBottle'
        ]
        LETTERS = ['A', 'B', 'C', 'D', 'E']
        
        # Format the valid objects list for the prompt
        valid_objects_str = ", ".join(VALID_OBJECTS)
        
        #header
        prompt = "Create a high-level plan for completing a household task using the allowed actions. Select the next best action from the provided options. Always try to navigate to the object before interacting with it. If an object is on the CounterTop, navigate to the object itself rather than the CounterTop."
        
        if naturalFormat:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.values())}" 
        else:
            prompt += f"\n\n\nAllowed actions: {', '.join(ACT_TO_STR.keys())}" 
        
        # Add the valid objects constraint
        prompt += f"\n\nIMPORTANT: Only use objects from this list: {valid_objects_str}"
        
        # Run KNN retrieval
        knn_retrieved_examples = self.knn_retrieval(curr_task, k)

        all_steps = []
        # Get all steps from the KNN examples
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]

            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

             #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned

            all_steps.extend(step_list)

        # Add in-context examples from knn retrieval
        for retrieved_task in knn_retrieved_examples:
            trainTaskRow = self.knn_set.loc[self.knn_set["task"] == retrieved_task]
            trainTaskRow = trainTaskRow.iloc[0]

            step_list = [literal_eval(listItem) for rowItem in trainTaskRow["gold_traj"] for listItem in rowItem]

            #REMOVE NAVIGATION STEPS if the flag is set
            if removeNav:
                stepListCleaned = []
                for listItem in step_list:
                    if "Navigation" not in listItem:
                        stepListCleaned.append(listItem)
                step_list = stepListCleaned
            
            # Format action names to be more natural
            if naturalFormat:
                stepListCleaned = []
                for listItem in step_list:
                    listItem = list(listItem)
                    act_str = ACT_TO_STR[listItem[0]] 
                    listItem[0] = act_str
                    stepListCleaned.append(tuple(listItem))
                step_list = stepListCleaned
            
            # Split past and next plans randomly
            planSplit = random.sample(range(len(step_list)),1)[0]
            decisionSplit = random.sample(range(planSplit,len(step_list)),1)[0]
            
            # In-context examples components
            high_level_str = str(trainTaskRow["task_instr"])
            step_by_step_str = '. '.join(trainTaskRow["step_instr"])
            past_plan_str = self.format_plan_str(step_list[:planSplit])
            next_plans_str = self.format_plan_str(step_list[planSplit:decisionSplit])
            gt_next_plan = step_list[decisionSplit]
            num_options = random.randint(2, 5)
            options = [gt_next_plan]
            for i in range(num_options - 1):
                random_step = random.choice(all_steps)
                while random_step in options:
                    random_step = random.choice(all_steps)
                options.append(random_step)
            random.shuffle(options)
            gt_letter = LETTERS[options.index(gt_next_plan)]
            options_str = "\n".join([f"{LETTERS[i]}. {' '.join(option)}" for i, option in enumerate(options)])
            

            in_context_obj_str = self.format_object_str(trainTaskRow["vis_objs"])

            # In-context examples
            prompt += "\n\nTask description: " + high_level_str
                
            # Include low-level instructions
            if includeLow:
                prompt += "\nStep by step instructions: " + step_by_step_str

            prompt +=  "\nCompleted plans: " + past_plan_str
            
            # Only include visible objects if dynamic planning is enabled
            if dynamic:
                if not in_context_obj_str == "":
                    prompt += "\nVisible objects are " + in_context_obj_str
            
            prompt += "\nNext Plans: " + next_plans_str
            prompt += "\nOptions:\n" + options_str
            prompt += f"\nAnswer: {gt_letter}"
        
        # Add the task prompt for GPT
        ## In-context examples components
        completed_plans = curr_task["completed_plans"]
        vis_objs = curr_task["vis_objs"]
        failed_plans = curr_task.get("failed_plans", [])

        task_high_level_str = str(curr_task["task_instr"][0])
        
        # Fix for step_instr being a list of lists or empty
        if curr_task["step_instr"] and isinstance(curr_task["step_instr"][0], list):
            # Either flatten the list
            step_instr_flat = [step for sublist in curr_task["step_instr"] for step in sublist]
            task_step_by_step_str = '. '.join(step_instr_flat)
        elif curr_task["step_instr"]:
            # Original code if it's already a flat list
            task_step_by_step_str = '. '.join(curr_task["step_instr"])
        else:
            # Handle empty step_instr
            task_step_by_step_str = ""
        
        task_past_plan_str = self.format_plan_str(completed_plans)
        task_obj_str = self.format_object_str(vis_objs)

       
        prompt += "\n\nTask description: " + task_high_level_str
       
        if includeLow:
            prompt += "\nStep by step instructions: " + task_step_by_step_str

        prompt += "\nCompleted plans: " + task_past_plan_str
        
        # Only include visible objects if dynamic planning is enabled
        if dynamic:
            if not task_obj_str == "":
                prompt += "\nVisible objects are " + task_obj_str

        if failed_plans:
            last_failed_plan = failed_plans[-1]
            prompt += "\nFailed plans: " + self.format_plan_str([last_failed_plan["plan"]]) +" with error: " + last_failed_plan["error"]

        prompt += "\nNext Plans: "

        prompt += self.format_plan_str(plan_steps)
        prompt += "\nOptions:\n" + "\n".join([f"{LETTERS[i]}. {option}" for i, option in enumerate(plan_options)])
        prompt += "\nBefore you answer, propose a question that might help clarify the task. This question should be answerable from visual information and the answer to this question should help disambiguate among the options. Possible good questions include asking about the existence of objects: 'Did you see a TableLamp?', asking about the state of objects: 'Is the apple sliced?', or asking if certain task conditions have been met: 'Was the bowl cleaned?'. Make sure the question is relevant to your decision."
        if additional_info:
            prompt += "\n" + "\n".join(additional_info)
        prompt += "\nQuestion:"
        curr_task["Prompts"] = prompt
        
        # Only store vis_objs if dynamic planning is enabled
        if dynamic:
            curr_task["vis_objs"] = vis_objs
        
        log.debug("Prompt generation complete")
        return prompt

    def generate_gpt_question_prompt(self, curr_task, plan_steps, plan_options, includeLow=False, dynamic=True, additional_info=[]):
        prompt = self.generate_question_prompt(
            curr_task, 
            k=self.config["num_in_context_examples"], 
            plan_steps=plan_steps, 
            plan_options=plan_options, 
            includeLow=includeLow, 
            dynamic=dynamic,
            additional_info=additional_info
            )
        return prompt
    
    def action_space_entropy_reduction_planner(self, curr_task, encoded_frames):
        """
        This planner aims to reduce the entropy of the action space collecting new information.
        """
        plan_tree = self.generate_plan_tree(curr_task, [], n_plans=self.config.get("num_diverse_plans", 5))
        if self.config.get("debug", False):
            self.logger.debug("=" * 50)
            self.logger.debug("PLAN TREE:")
            print_tree(plan_tree, print_function=self.logger.debug)
            self.logger.debug("=" * 50)
        selected_steps = []
        current_level = plan_tree
        while len(current_level) > 0:
            if len(current_level) == 1:
                # No uncertainty about next steps
                next_step = list(current_level.keys())[0]
                selected_steps.append(next_step)
                current_level = current_level[next_step]
            elif len(current_level) > 1:
                # Multiple options for next steps
                plan_options = list(current_level.keys())
                action_prob_prompt = self.generate_gpt_action_prob_prompt(
                    curr_task, 
                    selected_steps, 
                    plan_options, 
                    includeLow=self.config["use_step_instructions"], 
                    dynamic=self.config["dynamic"]
                    )
                if self.config.get("debug", False):
                    self.logger.debug("=" * 50)
                    self.logger.debug("ACTION PROB PROMPT:")
                    self.logger.debug(action_prob_prompt)
                    self.logger.debug("=" * 50)
                prior_action_probs = self.llm.MCQ_probabilities(action_prob_prompt, images=[])[:len(plan_options)]
                prior_action_probs /= np.sum(prior_action_probs)
                if self.config.get("debug", False):
                    self.logger.debug("=" * 50)
                    self.logger.debug("PRIOR ACTION PROBABILITIES:")
                    for i in range(len(plan_options)):
                        self.logger.debug(f"{plan_options[i]}: {prior_action_probs[i]:.4f}")
                    self.logger.debug("=" * 50)

                # Check if we are sufficiently confident about the next action
                if prior_action_probs.max() > self.confidence_threshold:
                    selected_option = plan_options[np.argmax(prior_action_probs)]
                    selected_steps.append(selected_option)
                    current_level = current_level[selected_option]
                
                else:
                    question_prompt = self.generate_gpt_question_prompt(
                        curr_task,
                        selected_steps,
                        plan_options,
                    )
                    questions = self.llm.generate(question_prompt, n=self.max_question_iterations, temperature=1.0, images=[])
                    questions = [self.clean_QA_output(question) for question in questions]
                    if self.config.get("debug", False):
                        self.logger.debug("=" * 50)
                        self.logger.debug("GENERATED QUESTIONS:")
                        for question in questions:
                            self.logger.debug(question)
                        self.logger.debug("=" * 50)
                    question_expected_info_gains = []
                    for question in questions:
                        imagined_answer_prompt = f"List potential answers to the question: {question} separated by newlines. Do not repeat answers."
                        imagined_answers = self.llm.generate(imagined_answer_prompt, n=1, temperature=0.0, images=[])[0]
                        imagined_answers = imagined_answers.split('\n')
                        info_gains = []
                        for answer in imagined_answers:
                            posterior_action_prob_prompt = self.generate_gpt_action_prob_prompt(
                                curr_task, 
                                selected_steps, 
                                plan_options, 
                                additional_info= [f"Question: {question} \n Answer: {answer}"]
                            )
                            posterior_action_probs = self.llm.MCQ_probabilities(posterior_action_prob_prompt, images=[])[:len(plan_options)]
                            posterior_action_probs /= np.sum(posterior_action_probs)
                            info_gain = calculate_entropy(prior_action_probs) - calculate_entropy(posterior_action_probs)
                            if self.config.get("debug", False):
                                self.logger.debug("=" * 50)
                                self.logger.debug(f"IMAGINED QUESTION: {question}")
                                self.logger.debug(f"IMAGINED ANSWER: {answer}")
                                self.logger.debug("POSTERIOR ACTION PROBABILITIES:")
                                for i in range(len(plan_options)):
                                    self.logger.debug(f"{plan_options[i]}: {posterior_action_probs[i]:.4f}")
                                self.logger.debug(f"PRIOR ENTROPY: {calculate_entropy(prior_action_probs):.4f}")
                                self.logger.debug(f"POSTERIOR ENTROPY: {calculate_entropy(posterior_action_probs):.4f}")
                                self.logger.debug(f"INFO GAIN: {info_gain:.4f}")
                                self.logger.debug("=" * 50)
                            info_gains.append(info_gain)
                        question_expected_info_gains.append(np.mean(info_gains))
                    sorted_indices = np.argsort(question_expected_info_gains)[::-1]
                    sorted_questions = [questions[idx] for idx in sorted_indices]
                    sorted_info_gains = [question_expected_info_gains[idx] for idx in sorted_indices]
                    if self.config.get("debug", False):
                        self.logger.debug("=" * 50)
                        self.logger.debug("SORTED QUESTIONS:")
                        for question, info_gain in zip(sorted_questions, sorted_info_gains):
                            self.logger.debug(f"Question: {question}")
                            self.logger.debug(f"Expected Info Gain: {info_gain}")
                        self.logger.debug("=" * 50)
                    additional_info = []
                    for question in sorted_questions:
                        answer = self.llm.VQA(question, images=encoded_frames)
                        answer = self.clean_QA_output(answer)
                        if self.config.get("debug", False):
                            self.logger.debug("=" * 50)
                            self.logger.debug("ADDITIONAL INFO:")
                            self.logger.debug(f"Question: {question}")
                            self.logger.debug(f"Answer: {answer}")
                            self.logger.debug("=" * 50)
                        posterior_action_prob_prompt = self.generate_gpt_action_prob_prompt(
                            curr_task, 
                            selected_steps, 
                            plan_options, 
                            additional_info=additional_info + [f"Question: {question} \n Answer: {answer}"]
                        )
                        posterior_action_probs = self.llm.MCQ_probabilities(posterior_action_prob_prompt, images=[])[:len(plan_options)]
                        posterior_action_probs /= np.sum(posterior_action_probs)
                        if self.config.get("debug", False):
                            self.logger.debug("=" * 50)
                            self.logger.debug("POSTERIOR ACTION PROBABILITIES:")
                            for i in range(len(plan_options)):
                                self.logger.debug(f"{plan_options[i]}: {posterior_action_probs[i]:.4f}")
                            self.logger.debug("=" * 50)
                        info_gain = calculate_entropy(prior_action_probs) - calculate_entropy(posterior_action_probs)
                        if info_gain > 0:
                            additional_info.append(f"Question: {question} \n Answer: {answer}")
                            self.logger.debug("=" * 50)
                            self.logger.debug("ADDITIONAL INFO:")
                            for info in additional_info:
                                self.logger.debug(info)
                            self.logger.debug("=" * 50)
                        if posterior_action_probs.max() > self.confidence_threshold:
                            self.logger.debug("Posterior action probabilities exceed confidence threshold.")
                            self.logger.debug(f"Max probability: {posterior_action_probs.max()}")
                            break
                    selected_option = plan_options[np.argmax(posterior_action_probs)]
                    selected_steps.append(selected_option)
                    current_level = current_level[selected_option]


        return selected_steps

    def clean_QA_output(self, output):
        if '\n' in output:
            lines = output.split('\n')
            # Return the first non-empty line
            for line in lines:
                if line.strip():
                    return line.strip()
        return output.strip()

    def generate_questions(self, dataset_path):
        result_dir = os.path.join(dataset_path, "results")
        trials = os.listdir(result_dir)
        trials.sort()
        for trial in trials:
            trial_path = os.path.join(result_dir, trial)
            trial_result_path = os.path.join(trial_path, "result.json")
            with open(trial_result_path) as f:
                trial_data = json.load(f)
            completed_plans = trial_data["completed_plans"]
            task_instr = trial_data["goal_instr"]
            image_paths = [f"{trial_path}/frame_{i:04d}.png" for i in range(len(completed_plans))]
            images = load_images(image_paths)
            best_completed_sequence = []
            best_options = []
            best_question_answer = ""
            best_info_gain = -np.inf
            print(f"Computing for task {task_instr}")
            for i in range(1, len(completed_plans)):
                completed_steps = completed_plans[:i]
                print(f"Completed steps: {completed_steps}")
                curr_task = {
                        "task_instr": [task_instr],
                        "step_instr": None,
                        "vis_objs": "", 
                        "completed_plans": completed_plans[:i]
                    }
                plan_tree = self.generate_plan_tree(curr_task, [])
                print_tree(plan_tree)
                # Check if the plan tree matches the future completed steps until the branching point
                good_tree = True
                current_level = plan_tree
                selected_steps = []
                for step in completed_plans[i:]:
                    if len(current_level) > 1:
                        break
                    elif step not in current_level:
                        good_tree = False
                        break
                    else:
                        selected_steps.append(step)
                    current_level = current_level[step]
                gt_step = step
                if good_tree:
                    plan_options = list(current_level.keys())
                    if gt_step not in plan_options:
                        plan_options = [gt_step] + plan_options
                        if len(plan_options) > 5:
                            plan_options = plan_options[:5]
                    curr_task = {
                        "task_instr": [task_instr],
                        "step_instr": None,
                        "vis_objs": "", 
                        "completed_plans": completed_plans[:i]
                    }
                    print(f"Plan options at branching point: {plan_options}")
                    action_prob_prompt = self.generate_gpt_action_prob_prompt(
                        curr_task, 
                        selected_steps, 
                        plan_options, 
                        includeLow=self.config["use_step_instructions"], 
                        dynamic=self.config["dynamic"]
                        )
                    prior_action_probs = self.llm.MCQ_probabilities(action_prob_prompt, images=[])[:len(plan_options)]
                    prior_action_probs /= np.sum(prior_action_probs)
                    prior_entropy = calculate_entropy(prior_action_probs)
                    question_prompt = self.generate_gpt_question_prompt(
                        curr_task,
                        selected_steps,
                        plan_options,
                        dynamic=False,
                    )
                    questions = self.llm.generate(question_prompt, n=5, temperature=1.0, images=[])
                    questions = [self.clean_QA_output(question) for question in questions]

                    encoded_images = [encode_image(image) for image in images[:len(selected_steps) + 1]]
                    answers = [self.llm.VQA(question, images=encoded_images) for question in questions]
                    answers = [self.clean_QA_output(answer) for answer in answers]
                    for question, answer in zip(questions, answers):
                        print(f"Question: {question}")
                        print(f"Answer: {answer}")
                        posterior_action_prob_prompt = self.generate_gpt_action_prob_prompt(
                            curr_task, 
                            selected_steps, 
                            plan_options, 
                            includeLow=self.config["use_step_instructions"], 
                            dynamic=self.config["dynamic"],
                            additional_info=f"Question: {question} \n Answer: {answer}"
                            )
                        posterior_action_probs = self.llm.MCQ_probabilities(posterior_action_prob_prompt, images=[])[:len(plan_options)]
                        posterior_action_probs /= np.sum(posterior_action_probs)
                        posterior_entropy = calculate_entropy(posterior_action_probs)
                        info_gain = posterior_entropy - prior_entropy
                        print(f"Prior entropy: {prior_entropy:.4f}, Posterior entropy: {posterior_entropy:.4f}, Information gain: {info_gain:.4f}")
                        gt_index = plan_options.index(gt_step)
                        # Test if we have become more confident about the right answer
                        if posterior_action_probs[gt_index] > prior_action_probs[gt_index]:
                            print(f"Information gain from question: {info_gain:.4f}")
                            if info_gain > best_info_gain:
                                best_info_gain = info_gain
                                best_completed_sequence = selected_steps
                                best_options = plan_options
                                best_question_answer = f"Question: {question} \n Answer: {answer}"

    def get_high_level_plan(self, curr_task, encoded_frames):
        """
        This planner returns a plan by generating a tree of possibilities and searching through it.
        """
        encoded_frames = remove_duplicate_images(encoded_frames)

        if "failed_plans" in curr_task:
            self.fail_count += 1
            self.logger.debug("In dynamic replanning mode, using action probabilities tree search to select next plan.")
            # First failure, use action probabilities tree search
            if self.fail_count == 1:
                return self.action_probs_tree_search(curr_task, [])
            # Subsequent failures, use action space entropy reduction planner
            elif self.fail_count > 1:
                return self.action_space_entropy_reduction_planner(curr_task, encoded_frames)
        else:
            self.fail_count = 0
            return self.generate_high_level_plan(curr_task, [], 1)[0]
        
def main():
    dataset_path = "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/train_results"
    config_file = "/home/mverghese/LLM_Grounding/LLM-Planner/e2e/config/config_alfred_train_data.yaml"
    with open(config_file) as reader:
            config = yaml.safe_load(reader)
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-abc123",
    )
    planner = Entropy_reduction_LLM_Planner(config["llm_planner"], client=client)
    planner.generate_questions(dataset_path)

if __name__ == "__main__":
    main()