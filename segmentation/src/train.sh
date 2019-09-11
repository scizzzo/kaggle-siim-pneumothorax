#!/bin/bash

GPU_IDX=0

CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_001/configs/0.json --folds_dir ../folds5 --balanced 1
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_001/configs/1.json --folds_dir ../folds5 --balanced 1
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_001/configs/3.json --folds_dir ../folds5 --balanced 1
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_001/configs/4.json --folds_dir ../folds5 --balanced 1


CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/0.json --folds_dir ../filled_folds5 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/1.json --folds_dir ../filled_folds5 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/2.json --folds_dir ../filled_folds5 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/3.json --folds_dir ../filled_folds5 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/4.json --folds_dir ../filled_folds5 --balanced 0


CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/0.json --folds_dir ../filled_folds10 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/1.json --folds_dir ../filled_folds10 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/2.json --folds_dir ../filled_folds10 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/3.json --folds_dir ../filled_folds10 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/4.json --folds_dir ../filled_folds10 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/5.json --folds_dir ../filled_folds10 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/6.json --folds_dir ../filled_folds10 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/7.json --folds_dir ../filled_folds10 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/8.json --folds_dir ../filled_folds10 --balanced 0
CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py --config_path ../experiments/exp_003_filled/configs/9.json --folds_dir ../filled_folds10 --balanced 0
