import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def step(optimizer, last_epoch, step_size=80, gamma=0.1, **_):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_):
    if isinstance(milestones, str):
        milestones = eval(milestones)
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                    gamma=gamma, last_epoch=last_epoch)


def none(optimizer, last_epoch, **_):
    return lr_scheduler.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)


def reduce_lr_on_plateau(optimizer, last_epoch, mode='min', factor=0.5, patience=10,
                         threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, **_):
    return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                          threshold=threshold, threshold_mode=threshold_mode,
                                          cooldown=cooldown, min_lr=min_lr)


def cosine(optimizer, last_epoch, **_):
    return CosineAnnealingWarmRestarts(optimizer, 15, eta_min=1e-6, last_epoch=last_epoch)


def get_scheduler(scheduler_name, optimizer, last_epoch, scheduler_params=None):
    scheduler_params = scheduler_params or {}
    func = globals().get(scheduler_name)
    return func(optimizer, last_epoch, **scheduler_params)

