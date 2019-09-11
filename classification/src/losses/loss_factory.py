import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(**_):
    return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()


def get_loss(loss_name, loss_params=None):
    loss_params = loss_params or {}
    f = globals().get(loss_name)
    return f(**loss_params)
