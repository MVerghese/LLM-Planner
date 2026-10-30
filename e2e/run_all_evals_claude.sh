#!/bin/bash

python src/run_eval.py --config config/config_alfred_base_claude.yaml &> output_base_claude.txt
python src/run_eval.py --config config/config_alfred_base_vision_claude.yaml &> output_base_vision_claude.txt
python src/run_eval.py --config config/config_alfred_action_probs_tree_claude.yaml &> output_action_probs_tree_claude.txt
python src/run_eval.py --config config/config_alfred_entropy_reduction_claude.yaml &> output_entropy_reduction_claude.txt
