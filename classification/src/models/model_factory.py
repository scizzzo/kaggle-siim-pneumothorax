import torch
import torch.nn as nn
import pretrainedmodels

from torchvision.models import resnet34, resnet50
from constants import NUM_CLASSES


def get_resnet34(num_classes=NUM_CLASSES,  **_):
    model = resnet34(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5, inplace=True), nn.Linear(in_features, num_classes))
    return model


def get_resnet50(num_classes=NUM_CLASSES,  **_):
    model = resnet50(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5, inplace=True), nn.Linear(in_features, num_classes))
    return model


def get_senet(model_name='se_resnext50', num_classes=NUM_CLASSES, **_):
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model


def get_se_resnext50(num_classes=NUM_CLASSES, **kwargs):
    return get_senet('se_resnext50_32x4d', num_classes=num_classes, **kwargs)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class SeResDp(nn.Module):
    def __init__(self, model_name, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        in_features = self.model.last_linear.in_features
        del self.model.last_linear
        self.pool = AdaptiveConcatPool2d()

        self.dp = nn.Dropout(0.5)
        self.linear = nn.Linear(in_features * 2, num_classes)

    def forward(self, x):
        x = self.model.features(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.linear(x)
        return x


def get_se_resnext50_dp(num_classes=NUM_CLASSES, **_):
    return SeResDp('se_resnext50_32x4d', num_classes=num_classes)


def get_se_resnext101_dp(num_classes=NUM_CLASSES, **_):
    return SeResDp('se_resnext101_32x4d', num_classes=num_classes)


def get_dpn98(num_classes=NUM_CLASSES, **_):
    model = pretrainedmodels.__dict__['dpn98'](num_classes=1000, pretrained='imagenet')
    in_ch = model.last_linear.in_channels
    del model.last_linear
    model.last_linear = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)
    return model


def get_model(model_name, model_params=None):
    print('model name:', model_name)
    f = globals().get('get_' + model_name)
    if model_params is None:
        return f()
    else:
        return f(**model_params)


if __name__ == '__main__':
    print('main')
    model = get_resnet34()
