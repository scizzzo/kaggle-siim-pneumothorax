import os
import pandas as pd
import numpy as np

train = pd.read_csv('./data/train-rle.csv')


def compute_area(rle):
    if rle == '-1':
        return 0
    pix_num = 0
    for pix in rle.split()[1::2]:
        pix_num += int(pix)
    return pix_num


train['area'] = train['EncodedPixels'].map(compute_area)
train_grouped = train[['ImageId', 'area']].groupby('ImageId').sum()
train_grouped_sorted = train_grouped.sort_values('area', ascending=False)

for num_folds in [5, 10]:
    os.makedirs(f'../folds{num_folds}')
    os.makedirs(f'../filled_folds{num_folds}')
    temp_folds = []
    for i in range(num_folds):
        temp_folds.append(train_grouped_sorted.index.values[i::num_folds])

    for i in range(num_folds):
        train_fold = np.concatenate([f for j, f in enumerate(temp_folds) if j != i])
        val_fold = temp_folds[i]
        np.save('./folds{}/fold_{}_train.npy'.format(num_folds, i), train_fold)
        np.save('./folds{}/fold_{}_test.npy'.format(num_folds, i), val_fold)

    cur_folds = f'../folds{num_folds}'
    for f in os.listdir(cur_folds):
        fold = np.load(os.path.join(cur_folds, f), allow_pickle=True)
        new_fold = []
        for id in fold:
            if train_grouped.loc[id].area > 0:
                new_fold.append(id)
        np.save(os.path.join(f'./filled_folds{num_folds}/', f), np.array(new_fold))