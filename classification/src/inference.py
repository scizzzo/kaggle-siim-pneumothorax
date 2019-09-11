import os
import argparse
import tqdm

import numpy as np
import pandas as pd
import torch

from datasets.custom_dataset import CustomTestDataset
from datasets import get_dataloader
from models import get_model
from constants import NUM_CLASSES
from utils.config import Config, INPUT_SIZE, MODEL_NAME, TRAIN_DIR
import utils.checkpoint


def inference(config, model, split, src_file, tta, output_filename=None):
    if split == 'test':
        data_path = '../data/dicom-images-test'
    else:
        data_path = '../data/dicom-images-train'

    dataset = CustomTestDataset(data_path, src_file, split, config[INPUT_SIZE])
    dataloader = get_dataloader(dataset, 1)

    model = model.cuda()
    model.eval()

    key_list = []
    probability_list = []

    with torch.no_grad():
        total_step = len(dataloader)
        for i, (images, id) in tqdm.tqdm(enumerate(dataloader), total=total_step):
            images = torch.cat(images, dim=0)
            images = images.cuda()

            logits = model(images)
            mean_logits = torch.mean(logits, dim=0, keepdim=True)
            probabilities = torch.softmax(mean_logits, dim=-1)
            probability_list.append(probabilities.cpu().numpy())

            key_list.extend(id)

        probabilities = np.concatenate(probability_list, axis=0)
        assert probabilities.shape[-1] == NUM_CLASSES

        records = []
        for id, probability in zip(key_list, probabilities):
            records.append(tuple([id] + ['{:.04f}'.format(p) for p in probability]))

        columns = ['id'] + ['P{:04d}'.format(l) for l in range(NUM_CLASSES)]

        df = pd.DataFrame.from_records(records, columns=columns)
        print('save {}'.format(output_filename))
        df.to_csv(output_filename, index=False)


def run(config, split, checkpoint_name, src_file, tta, output_filename):
    model = get_model(config[MODEL_NAME]).cuda()

    checkpoint = utils.checkpoint.get_checkpoint(config[TRAIN_DIR], name=checkpoint_name)
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    print('model loaded')
    inference(config, model, split, src_file, tta, output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        default=None, type=str)
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--config',
                        default=None, type=str)
    parser.add_argument('--checkpoint',
                        default=None, type=str)
    parser.add_argument('--tta', dest='tta', action='store_true')
    parser.set_defaults(tta=True)
    parser.add_argument('--split',
                        default='val', type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    print('inference', args)
    config = Config.from_json(args.config)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    run(config, args.split, args.checkpoint, args.src_file, args.tta, args.output)

    print('success!')


if __name__ == '__main__':
    main()
