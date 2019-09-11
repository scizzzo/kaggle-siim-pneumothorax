import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torchvision.models import resnet34, resnet50
from pretrainedmodels import se_resnext50_32x4d


class Upscale:
    transposed_conv = 0
    upsample_bilinear = 1
    pixel_shuffle = 2


torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))

encoder_params = {
    'resnet34':
        {'filters': [64, 64, 128, 256, 512],
         'init_op': resnet34,
         'url': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'},
    'resnet50':
        {'filters': [64, 256, 512, 1024, 2048],
         'init_op': resnet50,
         'url': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'},
    'se_resnext50':
        {'filters': [64, 256, 512, 1024, 2048],
         'init_op': se_resnext50_32x4d,
         'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth'},
}


class BasicDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, conv_size=3, upscale=Upscale.transposed_conv):
        super().__init__()
        padding = 0
        if conv_size == 3:
            padding = 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, conv_size, padding=padding),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True)
        )

        last_conv_channels = middle_channels
        if upscale == Upscale.transposed_conv:
            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(middle_channels, middle_channels, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(middle_channels),
                nn.ReLU(inplace=True)
            )
        elif upscale == Upscale.upsample_bilinear:
            self.layer2 = nn.Upsample(scale_factor=2)
        else:
            self.layer2 = nn.PixelShuffle(upscale_factor=2)
            last_conv_channels = middle_channels // 4

        self.layer3 = nn.Sequential(
            nn.Conv2d(last_conv_channels, out_channels, conv_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, upscale=Upscale.upsample_bilinear):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = torch.nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()['conv1.weight'][:,:3,...] = pretrained_dict['conv1.weight'].data
            skip_layers = self.first_layer_params_names
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not any(k.startswith(s) for s in skip_layers)}
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_names(self):
        return ['conv1.conv']


def _get_layers_params(layers):
    return sum((list(l.parameters()) for l in layers), [])


class EncoderDecoder(AbstractModel):
    def __init__(self, num_classes, num_channels=3, encoder_name='resnet34'):
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

        self.filters = encoder_params[encoder_name]['filters']

        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        if self.need_center:
            self.center = self.get_center()

        self.bottlenecks = nn.ModuleList([self.bottleneck_type(f * 2, f) for f in reversed(self.filters[:-1])]) #todo init from type
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(1, len(self.filters))])

        if self.first_layer_stride_two:
            middle_filters = self.filters[0]
            self.last_upsample = self.decoder_block(self.filters[0], middle_filters, self.filters[0], upscale=self.decoder_type)

        self.final = self.make_final_classifier(self.filters[0], num_classes)

        self._initialize_weights()

        encoder = encoder_params[encoder_name]['init_op']()
        self.encoder_stages = nn.ModuleList([self.get_encoder(encoder, idx) for idx in range(len(self.filters))])
        if encoder_params[encoder_name]['url'] is not None:
            self.initialize_encoder(encoder, encoder_params[encoder_name]['url'], num_channels != 3)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        enc_results = []
        for stage in self.encoder_stages:
            x = stage(x)
            enc_results.append(x.clone())

        if self.need_center:
            x = self.center(x)

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        if self.first_layer_stride_two:
            x = self.last_upsample(x)

        f = self.final(x)

        return torch.sigmoid(f)

    def get_decoder(self, layer):
        return self.decoder_block(self.filters[layer], self.filters[layer], self.filters[max(layer - 1, 0)], self.decoder_type)

    def make_final_classifier(self, in_filters, num_classes):
        return nn.Sequential(
            # nn.Conv2d(in_filters // 2, in_filters // 2, 3, padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_filters, num_classes, 3, padding=1)
        )

    def get_encoder(self, encoder, layer):
        raise NotImplementedError

    @property
    def first_layer_params(self):
        return _get_layers_params([self.encoder_stages[0]])

    @property
    def first_layer_params_names(self):
        raise NotImplementedError

    @property
    def layers_except_first_params(self):
        layers = get_slice(self.encoder_stages, 1, -1) + [self.bottlenecks, self.decoder_stages, self.final]
        if self.need_center:
            layers += [self.center_pool, self.center]
        return _get_layers_params(layers)


def get_slice(features, start, end):
    if end == -1:
        end = len(features)
    return [features[i] for i in range(start, end)]


