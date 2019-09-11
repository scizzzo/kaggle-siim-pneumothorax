import cv2
import os
import numpy as np
import pandas as pd
import random
import pydicom

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

from albumentations import ShiftScaleRotate, HorizontalFlip, RandomBrightness, RandomContrast, Blur, RandomGamma,\
    GaussNoise

from albumentations import Compose as Compose_alb

from constants import TRAIN_IMG_DIR, MASKS_PATH


def aug():
    return Compose_alb([
        HorizontalFlip(),
        ShiftScaleRotate(0.2, scale_limit=0.5, rotate_limit=10),
        RandomBrightness(),
        RandomContrast(p=0.2),
        Blur(p=0.2),
        RandomGamma(p=0.1),
        GaussNoise(p=0.05)
    ], p=0.95)


input_normalizer = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])


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


class CustomDataset(Dataset):
    def __init__(self,
                 data_dir,
                 stage,
                 fold_id,
                 data_prefix,
                 input_size):
        self.data_dir = data_dir
        self.stage = stage
        self.fold_id = fold_id
        self.data_prefix = data_prefix
        self.input_size = input_size
        self.segmentation_df = pd.read_csv(MASKS_PATH).set_index('ImageId')
        self._load_samples()

    def _load_samples(self):
        print('Load {} labels.'.format(self.stage))
        image_pathes = '{}_{}_{}.npy'.format(self.data_prefix, self.fold_id, self.stage)
        image_pathes = os.path.join(self.data_dir, image_pathes)
        image_pathes = np.load(image_pathes, allow_pickle=True)
        self.image_ids = image_pathes
        print('Image pathes are loaded.')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = self.image_ids[item]
        img_path = os.path.join(TRAIN_IMG_DIR, image_id + '.dcm')
        img_1ch = pydicom.read_file(img_path).pixel_array
        img = np.stack([img_1ch, img_1ch, img_1ch], axis=2)
        mask = get_mask(self.segmentation_df.loc[image_id]['EncodedPixels'])

        if self.stage == 'train':
            augmentation = aug()
            data = {'image': img, 'mask': mask}
            augmented = augmentation(**data)
            img, mask = augmented['image'], augmented['mask']

        img = cv2.resize(img, (self.input_size, self.input_size))
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        return input_normalizer(img), torch.from_numpy(mask).float().unsqueeze_(0)


class CustomTestDataset(Dataset):
    def __init__(self,
                 data_dir,
                 labels_path,
                 stage,
                 input_size):
        self.data_dir = data_dir
        self.stage = stage
        self.labels_path = labels_path
        self.input_size = input_size
        self._load_samples()

    def _load_samples(self):
        print('Load {} labels.'.format(self.stage))
        image_pathes = np.load(self.labels_path, allow_pickle=True)
        self.image_ids = image_pathes
        print('Image pathes are loaded.')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item):
        image_id = self.image_ids[item]
        img_path = os.path.join(self.data_dir, image_id + '.dcm')
        img_1ch = pydicom.read_file(img_path).pixel_array
        img = np.stack([img_1ch, img_1ch, img_1ch], axis=2)

        out_images = [img]
        out_images = [cv2.resize(im, (self.input_size, self.input_size)) for im in out_images]
        return [input_normalizer(img) for img in out_images], image_id


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        self.balanced_min = float('inf')
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        self.balanced_min = min([len(v) for v in self.dataset.values()])

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            image_id = dataset.image_ids[idx]
            mask = get_mask(dataset.segmentation_df.loc[image_id]['EncodedPixels'])
            if mask.sum() > 0:
                return 1
            else:
                return 0

    def __len__(self):
        return self.balanced_max * len(self.keys)
