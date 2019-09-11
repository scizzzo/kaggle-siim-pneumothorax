import torch.nn as nn

from .abstract_model import EncoderDecoder, Upscale, UnetDecoderBlock, ConvBottleneck


class Resnet(EncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name):
        if not hasattr(self, 'decoder_type'):
            self.decoder_type = Upscale.upsample_bilinear
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'need_center'):
            self.need_center = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck
        super().__init__(num_classes, num_channels, encoder_name)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu)
        elif layer == 1:
            return nn.Sequential(
                encoder.maxpool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_names(self):
        return ['conv1']


class Seresnet(EncoderDecoder):
    def __init__(self, num_classes, num_channels, encoder_name):
        if not hasattr(self, 'decoder_type'):
            self.decoder_type = Upscale.upsample_bilinear
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'need_center'):
            self.need_center = False
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck
        super().__init__(num_classes, num_channels, encoder_name)

    def get_encoder(self, encoder, layer):
        if layer == 0:
            return nn.Sequential(
                encoder.layer0.conv1,
                encoder.layer0.bn1,
                encoder.layer0.relu1)
        elif layer == 1:
            return nn.Sequential(
                encoder.layer0.pool,
                encoder.layer1)
        elif layer == 2:
            return encoder.layer2
        elif layer == 3:
            return encoder.layer3
        elif layer == 4:
            return encoder.layer4

    @property
    def first_layer_params_names(self):
        return ['conv1']


class Seresnext50(Seresnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        super().__init__(num_classes, num_channels, encoder_name='se_resnext50')


class Resnet34_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        super().__init__(num_classes, num_channels, encoder_name='resnet34')


class Resnet50_upsample(Resnet):
    def __init__(self, num_classes, num_channels=3):
        self.decoder_type = Upscale.upsample_bilinear
        super().__init__(num_classes, num_channels, encoder_name='resnet50')

