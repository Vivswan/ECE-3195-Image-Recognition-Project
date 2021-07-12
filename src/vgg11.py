import re
from typing import List

import torch
import torch.nn as nn

from src.NNModule import NNModule

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_feature_layers(cfg, relu_class, batch_norm=False) -> nn.Sequential:
    layers: List[nn.Module] = []

    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=(1, 1))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), relu_class(inplace=True)]
            else:
                layers += [conv2d, relu_class(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


def make_fc_layer(dropout, relu_class, out_features):
    layers: List[nn.Module] = [
        nn.Linear(512 * 7 * 7, 4096),
        relu_class(inplace=True),
    ]
    if dropout:
        layers += [nn.Dropout()]

    layers += [
        nn.Linear(4096, 4096),
        relu_class(inplace=True),
    ]

    if dropout:
        layers += [nn.Dropout()]

    layers += [
        nn.Linear(4096, out_features),
    ]

    return nn.Sequential(*layers)


class VGG11Model(NNModule):
    # Modified Source code of torchvision.models.vgg
    def __init__(
            self,
            out_features: int,
            device: torch.device,
            log_dir: str,
            batch_norm: bool = False,
            leaky_relu: bool = False,
            init_class: str = "kaiming_normal_",
            dropout: bool = True,
    ):
        super().__init__(log_dir, device)
        if leaky_relu:
            relu_class = nn.LeakyReLU
        else:
            relu_class = nn.ReLU

        self.features = make_feature_layers(cfgs["A"], relu_class, batch_norm)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = make_fc_layer(dropout, relu_class, out_features)
        self._initialize_weights(init_class, leaky_relu)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, init_class: str, leaky_relu: bool) -> None:
        self.tb.add_text("init_class", re.sub("\n", "\n    ", "    " + str(init_class)))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init_class == "kaiming_normal_":
                    nn.init.kaiming_normal_(
                        m.weight,
                        mode='fan_out',
                        nonlinearity=('leaky_relu' if leaky_relu else 'relu')
                    )
                elif init_class == "xavier_uniform_":
                    nn.init.xavier_uniform_(
                        m.weight,
                        gain=nn.init.calculate_gain('leaky_relu' if leaky_relu else 'relu')
                    )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
