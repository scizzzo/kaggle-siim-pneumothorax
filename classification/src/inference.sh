#!/bin/bash

GPU_IDX=1

# EXP 1
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_001/configs/0.json \
      --tta \
      --output=../experiments/exp_001/test_0_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_001/configs/1.json \
      --tta \
      --output=../experiments/exp_001/test_1_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_001/configs/2.json \
      --tta \
      --output=../experiments/exp_001/test_2_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy


CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_001/configs/3.json \
      --tta \
      --output=../experiments/exp_001/test_3_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_001/configs/4.json \
      --tta \
      --output=../experiments/exp_001/test_4_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy


# EXP 2
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_004/configs/0.json \
      --tta \
      --output=../experiments/exp_004/test_0_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_004/configs/1.json \
      --tta \
      --output=../experiments/exp_004/test_1_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_004/configs/2.json \
      --tta \
      --output=../experiments/exp_004/test_2_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy


CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_004/configs/3.json \
      --tta \
      --output=../experiments/exp_004/test_3_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_004/configs/4.json \
      --tta \
      --output=../experiments/exp_004/test_4_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy


# EXP 3

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_005/configs/0.json \
      --tta \
      --output=../experiments/exp_005/test_0_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_005/configs/1.json \
      --tta \
      --output=../experiments/exp_005/test_1_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_005/configs/2.json \
      --tta \
      --output=../experiments/exp_005/test_2_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy


CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_005/configs/3.json \
      --tta \
      --output=../experiments/exp_005/test_3_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_005/configs/4.json \
      --tta \
      --output=../experiments/exp_005/test_4_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy


# EXP 4
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_008/configs/0.json \
      --tta \
      --output=../experiments/exp_008/test_0_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_008/configs/1.json \
      --tta \
      --output=../experiments/exp_008/test_1_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_008/configs/2.json \
      --tta \
      --output=../experiments/exp_008/test_2_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy


CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_008/configs/3.json \
      --tta \
      --output=../experiments/exp_008/test_3_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_008/configs/4.json \
      --tta \
      --output=../experiments/exp_008/test_4_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

# EXP 5
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_009/configs/0.json \
      --tta \
      --output=../experiments/exp_009/test_0_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_009/configs/1.json \
      --tta \
      --output=../experiments/exp_009/test_1_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_009/configs/2.json \
      --tta \
      --output=../experiments/exp_009/test_2_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy


CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_009/configs/3.json \
      --tta \
      --output=../experiments/exp_009/test_3_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy

CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config=../experiments/exp_009/configs/4.json \
      --tta \
      --output=../experiments/exp_009/test_4_fold_res.csv \
      --checkpoint=best_model.pth \
      --split=test \
      --src_file=../data/subm_ids.npy