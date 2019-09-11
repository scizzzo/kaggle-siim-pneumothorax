#!/bin/bash


GPU_IDX=0

CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_001/configs/0.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_001/configs/1.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_001/configs/2.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_001/configs/3.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_001/configs/4.json


CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_004/configs/0.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_004/configs/1.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_004/configs/2.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_004/configs/3.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_004/configs/4.json


CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_005/configs/0.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_005/configs/1.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_005/configs/2.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_005/configs/3.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_005/configs/4.json


CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_008/configs/0.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_008/configs/1.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_008/configs/2.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_008/configs/3.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_008/configs/4.json


CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_009/configs/0.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_009/configs/1.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_009/configs/2.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_009/configs/3.json
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_009/configs/4.json
