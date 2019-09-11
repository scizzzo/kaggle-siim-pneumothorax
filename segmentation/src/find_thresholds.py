import argparse
import os
import cv2
import pickle

import pandas as pd
import numpy as np


from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import label


masks_rle_path = '../data/train-rle.csv'
masks_rle = pd.read_csv(masks_rle_path)
masks_rle = masks_rle.set_index('ImageId')


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height).T


def get_mask(masks_rle):
    masks = []
    if isinstance(masks_rle, str):
        if masks_rle.strip() == '-1':
            masks.append(np.zeros((1024, 1024)))
        else:
            mask = rle2mask(masks_rle, 1024, 1024)
            masks.append(mask)
    else:
        for mask_rle in masks_rle.values:
            mask = rle2mask(mask_rle, 1024, 1024)
            masks.append(mask)
    mask = np.any(masks, axis=0).astype(np.uint8)
    return mask


def drop_trash(mask, threshold):
    new_mask = np.zeros_like(mask)
    labeled_mask, num_l = label(mask)
    for i in range(1, num_l + 1):
        temp_mask = (labeled_mask == i)
        if temp_mask.sum() > threshold:
            new_mask[temp_mask] = 1
    return new_mask


def dice(outputs, targets, thresh, drop_th=500):
    outputs[outputs >= thresh] = 1
    outputs[outputs <= thresh] = 0
    outputs = drop_trash(outputs, drop_th)
    outputs = outputs.reshape(-1)
    targets = targets.reshape(-1)
    nominator = 2 * (outputs * targets).sum()
    denominator = outputs.sum() + targets.sum()
    if denominator == 0:
        return 1
    return nominator / denominator


def compute_metric(image_id, pred_dir, thresh, pixel_thresh):
    pred = np.load(os.path.join(pred_dir, image_id + '.npy'))[0][0]
    pred = cv2.resize(pred, (1024, 1024))
    target = get_mask(masks_rle.loc[image_id].EncodedPixels)
    return dice(pred, target, thresh, pixel_thresh)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--folds_path', type=str, default='./filled_folds2')
    parser.add_argument('--pixel_thresh', type=int, default=500)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_dir = args.pred_dir

    pred_folds = [f for f in os.listdir(pred_dir) if f.startswith('val_')]
    pred_folds = sorted(pred_folds, key=lambda x: int(x.split('_')[1]))
    print(pred_folds)
    best_thresholds = []
    for fold_pred_path in pred_folds:
        image_ids = np.load(os.path.join(args.folds_path, 'fold_{}_test.npy'.format(
            int(fold_pred_path.split('_')[1]))), allow_pickle=True)

        threshes = np.arange(0.3, 0.6, 0.01)
        thresh_scores = []
        for thr in threshes:
            dice_scores = Parallel(n_jobs=12)(delayed(compute_metric)(id_, os.path.join(args.pred_dir, fold_pred_path),
                                                                      thr, args.pixel_thresh) for id_ in tqdm(image_ids))
            print(thr, np.mean(dice_scores))
            thresh_scores.append((thr, np.mean(dice_scores)))
        best_fold_thresh = max(thresh_scores, key=lambda x: x[1])[0]
        print(best_fold_thresh)
        best_thresholds.append(best_fold_thresh)

    with open(args.out, 'wb') as f:
        pickle.dump(best_thresholds, f)


if __name__ == '__main__':
    main()