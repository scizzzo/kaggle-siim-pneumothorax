import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segm_out', required=True)
    parser.add_argument('--class_out', required=True)
    parser.add_argument('--out')
    return parser.parse_args()


def main():
    args = parse_args()
    segm = pd.read_csv(args.segm_out)
    segm_ids = segm[segm.EncodedPixels != '-1'].ImageId.values

    class_ids = np.load(args.class_out, allow_pickle=True)
    merged = np.array(list(set(segm_ids.tolist() + class_ids.tolist())))
    np.save(args.out, merged)


if __name__ == '__main__':
    main()