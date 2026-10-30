#!/bin/bash

python src/run_eval.py --config config/config_alfred_base.yaml &> output_base.txt
python src/run_eval.py --config config/config_alfred_base_vision.yaml &> output_base_vision.txt
python src/run_eval.py --config config/config_alfred_action_probs_tree.yaml &> output_action_probs_tree.txt
python src/run_eval.py --config config/config_alfred_entropy_reduction.yaml &> output_entropy_reduction.txt
