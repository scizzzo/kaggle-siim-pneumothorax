from .albuunet import Resnet34_upsample, Resnet50_upsample, Seresnext50


def get_albunet_res34(pretrained=True):
    model = Resnet34_upsample(1)
    return model


def get_albunet_res50(pretrained=True):
    model = Resnet50_upsample(1)
    return model


def get_albunet_seres50(pretrained=True):
    model = Seresnext50(1)
    return model


def get_model(model_name, model_params=None):
    print('model name:', model_name)
    f = globals().get('get_' + model_name)
    if model_params is None:
        return f()
    else:
        return f(**model_params)

