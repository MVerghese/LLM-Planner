#!/bin/bash

python src/run_eval.py --config config/config_alfred_base_gpt.yaml &> output_base_gpt.txt
python src/run_eval.py --config config/config_alfred_base_vision_gpt.yaml &> output_base_vision_gpt.txt
python src/run_eval.py --config config/config_alfred_action_probs_tree_gpt.yaml &> output_action_probs_tree_gpt.txt
python src/run_eval.py --config config/config_alfred_entropy_reduction_gpt.yaml &> output_entropy_reduction_gpt.txt
