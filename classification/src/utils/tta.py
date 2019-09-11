import torch.nn as nn

from functools import partial
from torch import Tensor


def torch_fliplr(x: Tensor):
    return x.flip(3)


def fliplr_image2label(model: nn.Module, image: Tensor) -> Tensor:
    output = model(image) + model(torch_fliplr(image))
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


class TTAWrapper(nn.Module):
    def __init__(self, model, tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, *input):
        return self.tta(self.model, *input)