import argparse
import os
import cv2
import pickle
import pandas as pd
import numpy as np

from scipy.ndimage import label
from mask_functions import mask2rle
from tqdm import tqdm

PIXEL_THRESH = 500


def get_masks(id_, segm_folds, thresholds):
    pred_masks = []
    for fold, thr in zip(segm_folds, thresholds):
        mask = np.load(os.path.join(fold, id_ + '.npy'))[0][0]
        mask = cv2.resize(mask, (1024, 1024))
        mask = mask - thr
        pred_masks.append(mask)
    mask = np.mean(pred_masks, axis=0)
    mask[mask >= 0.0] = 1
    mask[mask <= 0.0] = 0
    labeled_mask, _ = label(mask)
    new_mask = np.zeros_like(mask)
    for i in range(1, len(np.unique(labeled_mask))):
        cur_mask = (labeled_mask == i)
        if cur_mask.sum() > PIXEL_THRESH:
            new_mask[cur_mask] = 1
    return new_mask


def parse_argse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_path', required=True)
    parser.add_argument('--pred_ids', required=True)
    parser.add_argument('--preds_paths', type=str, nargs='+', required=True)
    parser.add_argument('--thresholds_paths', nargs='+', required=True)
    return parser.parse_args()


def get_pred_folds(pred_path):
    folds = [f for f in os.listdir(pred_path) if f.startswith('test_')]
    folds = sorted(folds, key=lambda x: int(x.split('_')[1]))
    return [os.path.join(pred_path, f) for f in folds]


def main():
    args = parse_argse()
    class_subm = pd.read_csv(args.cls_path)
    preds_ids = np.load(args.pred_ids, allow_pickle=True)
    segm_subm = pd.DataFrame(columns=class_subm.columns)

    for row in tqdm(class_subm.iterrows(), total=len(class_subm)):
        if row[1][0] in preds_ids:
            continue
        row[1][1] = '-1'
        segm_subm = segm_subm.append(row[1])

    segm_folds = []
    for pred_p in args.preds_paths:
        segm_folds.extend(get_pred_folds(pred_p))

    thresholds = []
    for thr_p in args.thresholds_paths:
        with open(thr_p, 'rb') as f:
            thresholds.extend(pickle.load(f))

    for id_ in tqdm(preds_ids):
        out_mask = get_masks(id_, segm_folds, thresholds)
        if out_mask.sum() == 0:
            segm_subm = segm_subm.append({'ImageId': id_, 'EncodedPixels': class_subm.loc[id_].EncodedPixels},
                                         ignore_index=True)
            continue

        rle = mask2rle((out_mask * 255).T, 1024, 1024)
        segm_subm = segm_subm.append({'ImageId': id_, 'EncodedPixels': rle}, ignore_index=True)
    segm_subm.to_csv('submissions.csv', index=False)


if __name__ == '__main__':
    main()