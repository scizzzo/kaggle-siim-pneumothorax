#!/bin/bash

./train.sh

./inference.sh

python make_preds.py --pred_dir ../experiments/exp_001 ../experiments/exp_004 ../experiments/exp_005 ../experiments/exp_008 ../experiments/exp_009 --out ../../tmp