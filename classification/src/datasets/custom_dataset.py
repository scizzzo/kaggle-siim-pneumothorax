import cv2
import os
import numpy as np
import pandas as pd
import random
import pydicom

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize

from albumentations import ShiftScaleRotate, HorizontalFlip, RandomBrightness, RandomContrast, RandomGamma, \
    GaussianBlur, ElasticTransform
from albumentations import Compose as Compose_alb

from constants import MASKS_PATH, TRAIN_IMG_DIR, TTA_CROPS_PERCENT, CROPS_PERCENT_RANGE


def aug():
    return Compose_alb([
        HorizontalFlip(),
        ShiftScaleRotate(rotate_limit=10, border_mode=0),
        RandomGamma(),
        RandomBrightness(),
        RandomContrast(),
        ElasticTransform(border_mode=0),
        GaussianBlur()
    ], p=1)


input_normalizer = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])


def random_crop_percent(img, percent_range=CROPS_PERCENT_RANGE):
    h, w = img.shape[:2]
    p_width = random.randint(percent_range[0], percent_range[1])
    p_height = random.randint(percent_range[0], percent_range[1])
    crop_width = round(w * p_width / 100.)
    crop_height = round(h * p_height / 100.)
    start_x = random.randint(0, w - crop_width)
    start_y = random.randint(0, h - crop_height)
    return img[start_y:start_y+crop_height, start_x:start_x+crop_width]


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
        if image_id in self.segmentation_df.index:
            mask = get_mask(self.segmentation_df.loc[image_id]['EncodedPixels'])
            label = int(mask.sum() > 0)
        else:
            label = 1

        if self.stage == 'train':
            if random.random() < 0.9:
                img = random_crop_percent(img)
            augmentation = aug()
            data = {'image': img}
            augmented = augmentation(**data)
            img = augmented['image']

        img = cv2.resize(img, (self.input_size, self.input_size))
        return input_normalizer(img), torch.tensor(label).long()


def create_crops(img, crop_percent=TTA_CROPS_PERCENT):
    h, w = img.shape[:2]
    crops = []
    if max(h, w) / min(h, w) > 5:
        if h > w:
            crop_height = round(h * crop_percent)
            crops.append(img[:crop_height, :])
            crops.append(img[-crop_height:, :])
        else:
            crop_width = round(w * crop_percent)
            crops.append(img[:, :crop_width])
            crops.append(img[:, -crop_width:])
    else:
        crop_height = round(h * crop_percent)
        crop_width = round(w * crop_percent)
        crops.append(img[:crop_height, :crop_width])
        crops.append(img[:crop_height, -crop_width:])
        crops.append(img[-crop_height:, -crop_width:])
        crops.append(img[-crop_height:, :crop_width])
    return crops


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
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

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