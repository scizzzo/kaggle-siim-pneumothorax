import os
import argparse
import pandas as pd
import numpy as np


CLS_THRESH = 0.7


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--out')
    return parser.parse_args()


def main():
    args = parse_args()
    dirs_preds = []
    for test_dir in args.pred_dirs:
        res_preds = []
        for file in os.listdir(test_dir):
            if 'test_' in file:
                df = pd.read_csv(os.path.join(test_dir, file))
                res_preds.append(df)

        mean_pred = np.sum([df.values[:, -2:] for df in res_preds], axis=0) / len(res_preds)
        pred_labels = (mean_pred[:, 1] > CLS_THRESH).astype(np.uint8)
        dirs_preds.append(pred_labels)

    dirs_preds = np.array(dirs_preds)
    realy_true = np.all(dirs_preds != 0, axis=0)
    realy_ids = df.id.values[realy_true]
    np.save(os.path.join(args.out, 'cls_ids.npy'), realy_ids)


if __name__ == '__main__':
    main()