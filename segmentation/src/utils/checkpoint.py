from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import torch


def get_checkpoint(train_dir, name):
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    return os.path.join(checkpoint_dir, name)


def load_checkpoint(model, optimizer, checkpoint):
    print('load checkpoint from', checkpoint)
    checkpoint = torch.load(checkpoint)

    checkpoint_dict = checkpoint['state_dict']
    model.load_state_dict(checkpoint_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    step = checkpoint['step'] if 'step' in checkpoint else -1
    last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

    return last_epoch, step


def save_checkpoint(train_dir, model, optimizer, epoch, step, weights_dict=None, name=None):
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')

    if name:
        checkpoint_path = os.path.join(checkpoint_dir, '{}.pth'.format(name))
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch_{:04d}.pth'.format(epoch))

    if weights_dict is None:
        weights_dict = {
            'state_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
        }
    torch.save(weights_dict, checkpoint_path)
