import os
import argparse
import tqdm

import numpy as np
import torch

from datasets.custom_dataset import CustomTestDataset
from datasets import get_dataloader
from models import get_model
from utils.config import Config, INPUT_SIZE, MODEL_NAME, TRAIN_DIR
from utils.tta import TTAWrapper, fliplr_image2mask
import utils.checkpoint


def inference(config, model, split, src_file, output_path=None):
    if split == 'test':
        data_path = '../data/dicom-images-test'
    else:
        data_path = '../data/dicom-images-train'

    dataset = CustomTestDataset(data_path, src_file, split, config[INPUT_SIZE])
    dataloader = get_dataloader(dataset, 1)

    model = model.cuda()
    model.eval()
    model = TTAWrapper(model, fliplr_image2mask)

    with torch.no_grad():
        total_step = len(dataloader)
        for i, (images, id) in tqdm.tqdm(enumerate(dataloader), total=total_step):

            images = torch.cat(images, dim=0)
            images = images.cuda()

            merged_out = model(images)
            mean_logits = torch.mean(merged_out, dim=0, keepdim=True)

            np.save(os.path.join(output_path, id[0] + '.npy'), mean_logits.cpu().numpy())


def run(config, split, checkpoint_name, src_file, output_filename):
    model = get_model(config[MODEL_NAME]).cuda()

    checkpoint = utils.checkpoint.get_checkpoint(config[TRAIN_DIR], name=checkpoint_name)
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    print('model loaded')
    inference(config, model, split, src_file, output_filename)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        default=None, type=str)
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--config',
                        default=None, type=str)
    parser.add_argument('--checkpoint',
                        default=None, type=str)

    parser.add_argument('--split',
                        default='val', type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    args = parse_args()
    print('inference', args)
    config = Config.from_json(args.config)

    os.makedirs(args.output, exist_ok=True)

    run(config, args.split, args.checkpoint, args.src_file, args.output)

    print('success!')


if __name__ == '__main__':
    main()
