import torch.optim as optim

from .adamw import AdamW
from .radam import RAdam


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0,
         amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                      amsgrad=amsgrad)


def adamw(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0, **_):
    return AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)


def radam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0, **_):
    return RAdam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)


def sgd(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
    return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                     nesterov=nesterov)


def get_optimizer(optimizer_name, model_parameters, optimizer_params=None):
    optimizer_params = optimizer_params or {}
    f = globals().get(optimizer_name)
    return f(model_parameters, **optimizer_params)

