import torch.nn as nn


class DICELoss:
    eps = 1e-7

    def __init__(self, size_average):
        self.size_average = size_average

    def __call__(self, outputs, targets):
        batch_size = outputs.size(0)
        outputs = outputs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        nominator = (outputs * targets).sum(dim=1)
        denominator = outputs.sum(dim=1) + targets.sum(dim=1)

        if self.size_average:
            return ((2. * nominator + DICELoss.eps) / (denominator + DICELoss.eps)).mean()
        return (2. * nominator + DICELoss.eps) / (denominator + DICELoss.eps)


class BCEDICELoss:
    def __init__(self, loss_weights=None, size_average=True):
        loss_weights = loss_weights or {'bce': 0.5, 'dice': 0.5}
        self.bce_loss = nn.BCELoss(reduction='elementwise_mean' if size_average else 'none')
        self.dice_loss = DICELoss(size_average=size_average)
        self.loss_weights = loss_weights
        self.size_average = size_average

    def __call__(self, outputs, targets):
        if self.size_average:
            bce_loss = self.loss_weights['bce'] * self.bce_loss(outputs, targets)
        else:
            bce_loss = (self.loss_weights['bce'] * self.bce_loss(outputs, targets)).view(outputs.size(0), -1).mean()
        return bce_loss \
               + self.loss_weights['dice'] * (1 - self.dice_loss(outputs, targets))


def get_loss(loss_name, loss_params=None):
    loss_params = loss_params or {}
    f = globals().get(loss_name)
    return f(**loss_params)
