import argparse
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from tensorboardX import SummaryWriter

from constants import DATA_DIR
from datasets import get_dataloader
from datasets.custom_dataset import CustomDataset
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
import utils
import utils.checkpoint

from utils.config import Config, TRAIN_DIR, DATA_PREFIX, FOLD_ID, EPOCHS, MODEL_NAME, MODEL_PARAMS, OPTIM_NAME, \
    OPTIM_PARAMS, SCHEDULER_NAME, SCHEDULER_PARAMS, LOSS_NAME, LOSS_PARAMS, INPUT_SIZE, BATCH_SIZE


def inference(model, images):
    logits = model(images)
    if isinstance(logits, tuple):
        logits, aux_logits = logits
    else:
        aux_logits = None
    probabilities = F.sigmoid(logits)
    return logits, aux_logits, probabilities


def evaluate_single_epoch(model, dataloader, criterion, epoch, writer, postfix_dict):
    model.eval()

    with torch.no_grad():
        total_step = len(dataloader)

        probability_list = []
        label_list = []
        loss_list = []
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, (images, labels) in tbar:
            images = images.cuda()
            labels = labels.cuda()
            logits, aux_logits, probabilities = inference(model, images)

            loss = criterion(logits, labels)

            loss_list.append(loss.item())

            probability_list.extend(probabilities.cpu().numpy())
            label_list.extend(labels.cpu().numpy())

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format('val')
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

        log_dict = {}
        labels = np.array(label_list)
        probabilities = np.array(probability_list)

        predictions = np.argmax(probabilities, axis=1)

        log_dict['acc'] = (predictions == labels).sum() / len(labels)
        log_dict['loss'] = sum(loss_list) / len(loss_list)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('val/{}'.format(key), value, epoch)
            postfix_dict['val/{}'.format(key)] = value

        return log_dict['loss'], log_dict['acc']


def train_single_epoch(config, model, dataloader, criterion, optimizer,
                       epoch, writer, postfix_dict):
    model.train()

    total_step = len(dataloader)

    log_dict = {}
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, (images, labels) in tbar:
        images = images.cuda()
        labels = labels.cuda()
        logits, aux_logits, probabilities = inference(model, images)
        loss = criterion(logits, labels)

        log_dict['loss'] = loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        f_epoch = epoch + i / total_step

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        if i % 100 == 0:
            log_step = int(f_epoch * 10000)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)


def train(config, model, dataloaders, criterion, optimizer, scheduler, writer, start_epoch):
    num_epochs = config[EPOCHS]

    model = model.cuda()
    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'val/acc': 0.0,
                    'val/loss': 0.0}

    best_acc = 0.0
    # best_f1_mavg = 0.0
    for epoch in range(start_epoch, num_epochs):
        # train phase
        train_single_epoch(config, model, dataloaders['train'],
                           criterion, optimizer, epoch, writer, postfix_dict)

        # val phase
        val_loss, acc = evaluate_single_epoch(model, dataloaders['test'], criterion, epoch, writer, postfix_dict)

        if config[SCHEDULER_NAME] == 'reduce_lr_on_plateau':
            scheduler.step(acc)
        else:
            scheduler.step()

        utils.checkpoint.save_checkpoint(config[TRAIN_DIR], model, optimizer, epoch, 0, name='model')

        if optimizer.param_groups[0]['lr'] < 1e-6 / 2:
            print('too small lr')
            break
        if acc > best_acc:
            best_acc = acc
            utils.checkpoint.save_checkpoint(config[TRAIN_DIR], model, optimizer, epoch, 0, name='best_model')

    return {'f1': best_acc}


def run(config):
    model = get_model(config[MODEL_NAME], config[MODEL_PARAMS]).cuda()
    criterion = get_loss(config[LOSS_NAME], config[LOSS_PARAMS])
    optimizer = get_optimizer(config[OPTIM_NAME], model.parameters(), optimizer_params=config[OPTIM_PARAMS])

    last_epoch = -1
    scheduler = get_scheduler(config[SCHEDULER_NAME], optimizer, last_epoch, config[SCHEDULER_PARAMS])

    datasets = {stage: CustomDataset(DATA_DIR, stage, config[FOLD_ID], config[DATA_PREFIX], config[INPUT_SIZE])
                for stage in ['train', 'test']}

    dataloaders = {stage: get_dataloader(datasets[stage], config[BATCH_SIZE])
                   for stage in ['train', 'test']}

    writer = SummaryWriter(config[TRAIN_DIR])
    clip_grad_value_(model.parameters(), 2.0)
    train(config, model, dataloaders, criterion, optimizer, scheduler,
          writer, last_epoch+1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings('ignore')
    print('Pneumotrax Challenge.')
    args = parse_args()
    config = Config.from_json(args.config_path)
    utils.prepare_train_directories(config[TRAIN_DIR])
    run(config)

    print('success!')


if __name__ == '__main__':
    main()
