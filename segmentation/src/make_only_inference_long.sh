#!/bin/bash

GPU_IDX=0
# First step segmentation validation

for i in {0..5}
do
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config="../experiments/exp_001/configs/"$i".json" \
      --tta \
      --output="../experiments/exp_001/val_"$i"_fold" \
      --checkpoint=best_model.pth \
      --split=val \
      --src_file="../folds5/fold_"$i"_test.npy"

done

for i in {0..9}
do
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config="../experiments/exp_002/configs/"$i".json" \
      --tta \
      --output="../experiments/exp_002/val_"$i"_fold" \
      --checkpoint=best_model.pth \
      --split=val \
      --src_file="../folds10/fold_"$i"_test.npy"

done

python find_threshold.py  --pred_dir ../experiments/exp_001  --folds_path ../folds5 --pixel_thresh 2000 --out ../../tmp/first_stage_segm_thr.pkl
python find_threshold.py  --pred_dir ../experiments/exp_002  --folds_path ../folds10 --pixel_thresh 2000 --out ../../tmp/first_stage_segm_thr2.pkl


# First step segmentation inference

for i in {0..5}
do
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config="../experiments/exp_001/configs/"$i".json" \
      --tta \
      --output="../experiments/exp_001/test_"$i"_fold" \
      --checkpoint=best_model.pth \
      --split=val \
      --src_file=../data/subm_ids.npy

done


for i in {0..9}
do
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config="../experiments/exp_002/configs/"$i".json" \
      --tta \
      --output="../experiments/exp_002/test_"$i"_fold" \
      --checkpoint=best_model.pth \
      --split=val \
      --src_file=../data/subm_ids.npy

done

python make_preds_class.py --preds_paths ../experiments/exp_001 ../experiments/exp_002 --thresholds_paths ../../tmp/first_stage_segm_thr.pkl ../../tmp/first_stage_segm_thr2.pkl  --out ../../tmp

python create_second_stage_ids.py --segm_out ../../tmp/segm_class.csv --clas_out ../../tmp/cls_ids.npy --out ../../tmp/second_step_ids.npy


# Second step validation

for i in {0..4}
do
  CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
        --config="../experiments/exp_003_filled/configs/"$i".json" \
        --tta \
        --output="../experiments/exp_003_filled/val_"$i"_fold" \
        --checkpoint=best_model.pth \
        --split=val \
        --src_file="../filled_folds5/fold_"$i"_test.npy"
done


for i in {0..9}
do
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config="../experiments/exp_004_filled/configs/"$i".json" \
      --tta \
      --output="../experiments/exp_004_filled/val_"$i"_fold" \
      --checkpoint=best_model.pth \
      --split=val \
      --src_file="../filled_folds10/fold_"$i"_test.npy"

done


python find_threshold.py  --pred_dir ../experiments/exp_003_filled  --folds_path ../filled_folds5 --pixel_thresh 500 --out ../../tmp/exp_003_thr.pkl
python find_threshold.py  --pred_dir ../experiments/exp_004_filled  --folds_path ../filled_folds10 --pixel_thresh 500 --out ../../tmp/exp_004_thr.pkl


# Second step inference

for i in {0..4}
do
  CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
        --config="../experiments/exp_003_filled/configs/"$i".json" \
        --tta \
        --output="../experiments/exp_003_filled/test_"$i"_fold" \
        --checkpoint=best_model.pth \
        --split=val \
        --src_file=../../tmp/second_step_ids.npy
done


for i in {0..9}
do
CUDA_VISIBLE_DEVICES=$GPU_IDX python inference.py \
      --config="../experiments/exp_004_filled/configs/"$i".json" \
      --tta \
      --output="../experiments/exp_004_filled/test_"$i"_fold" \
      --checkpoint=best_model.pth \
      --split=val \
      --src_file=../../tmp/second_step_ids.npy

done


python make_preds_class.py --class_subm ../../tmp/segm_class.csv --pred_ids ../../tmp/second_step_ids.npy --preds_paths ../experiments/exp_003_filled ../experiments/exp_004_filled --thresholds_paths ../../tmp/exp_003_thr.pkl ../../tmp/exp_004_thr.pkl