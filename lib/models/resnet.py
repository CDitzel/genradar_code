from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U

from .network_utils import *

# from .dynamicUnet import *


class ResNetMainBlock(nn.Sequential):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        out_expanded,
        kernel_size,
        stride,
        act,
        **kwargs,
    ):
        # print("ResNetMainBlock", kwargs)
        block1 = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            act=act,
            **kwargs,
        )
        block2 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_expanded,
            act=act,
            **kwargs,
        )
        block3 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_expanded,
            act=act,
            **kwargs,
        )
        block4 = ConvLayer(
            in_channels=out_channels,
            out_channels=out_expanded,
            act=act,
            **kwargs,
        )
        super().__init__(block1, block2, block3, block4)


class ResNetBlock(nn.Module):
    def __init__(
        self,
        main_block,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        act,
        **kwargs,
    ):
        super().__init__()
        # print("ResNetBlock", kwargs)
        self.DALLE_CONSTANT = 1 / 1
        out_expanded = out_channels * main_block.expansion
        # print('in res block', kwargs)

        self.blocks = main_block(
            in_channels=in_channels,
            out_channels=out_channels,
            out_expanded=out_expanded,
            kernel_size=kernel_size,
            stride=stride,
            act=act,
            **kwargs,
        )

        self.shortcut = (
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_expanded,
                kernel_size=1 if stride == 1 else 2,
                stride=stride,
                # norm=None,
                # stride=1,
                # act=None,
                # **kwargs,
            )
            if stride != 1 or in_channels != out_expanded
            else nn.Identity()
        )

        # self.act = get_activation(act)

    def forward(self, x):
        main = self.blocks(x) * self.DALLE_CONSTANT
        # main = self.act(main)
        s = self.shortcut(x)
        x = main + s
        # x = self.act(x)
        return x


class ResNetLayer(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, block, num_blocks, lazy_down, attn, **kwargs
    ):
        # print('in res layer', kwargs)

        # print('in', in_channels)
        # print('out', out_channels)
        # print(num_blocks)
        blocks = [
            ResNetBlock(
                main_block=block,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2 if in_channels != out_channels and lazy_down is None else 1,
                **kwargs,
            ),
            *[
                ResNetBlock(
                    main_block=block,
                    in_channels=out_channels * block.expansion,
                    out_channels=out_channels,
                    stride=1,
                    **kwargs,
                )
                for n in range(num_blocks - 1)
            ],
        ]
        if lazy_down is not None:
            blocks.append(lazy_down)
        if attn:
            blocks.append(AttnBlock(out_channels))
        super().__init__(*blocks)


# class ResNetEncoder(nn.ModuleList):
@instantiate_or_load
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_chn=1,
        # blocks_sizes=[1, 64, 128, 256, 512],
        # num_blocks=[1, 1, 1, 1],
        blocks_sizes=[1, 64],
        num_blocks=[2],
        block=ResNetMainBlock,
        lazy_down=False,
        attn=[False],
        **kwargs,
    ):
        super().__init__()
        # assert len(blocks_sizes) - 1 == len(num_blocks) == len(attn)
        assert len(blocks_sizes) == len(num_blocks) == len(attn)
        self.blocks_sizes = blocks_sizes
        self.num_blocks = num_blocks
        self.attn = attn
        self.lazy_down = lazy_down
        self.in_chn = in_chn
        # self.nLayers = len(blocks_sizes) - 1

        self.nLayers = len(blocks_sizes)
        self.gate = ConvLayer(
            in_channels=in_chn,
            out_channels=blocks_sizes[0],
            kernel_size=7,
            stride=1,
            padding=3) if 'order' in kwargs and kwargs['order'].startswith('a') else nn.Identity()

        self.layers = nn.ModuleList([])

        self.layers.append(ResNetLayer(
            in_channels=blocks_sizes[0] if not isinstance(self.gate, type(nn.Identity())) else in_chn,
            out_channels=blocks_sizes[0],
            block=block,
            num_blocks=num_blocks[0],
            lazy_down=None if not lazy_down else Downsample(blocks_sizes[0]),
            attn=attn[0],
            **kwargs))

        for in_channels, out_channels, num_blocks, attn in zip(
            blocks_sizes, blocks_sizes[1:], num_blocks[1:], attn[1:]
        ):
            self.layers.append(
                ResNetLayer(
                    in_channels=in_channels * block.expansion,
                    out_channels=out_channels,
                    block=block,
                    num_blocks=num_blocks,
                    lazy_down=None if not lazy_down else Downsample(out_channels),
                    attn=attn,
                    **kwargs,
                )
            )
    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        for layer in self.layers:
            yield layer

    def forward(self, x):
        x = self.gate(x)
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == '__main__':
    model = ResNetEncoder(
        in_chn=1,
        lazy_down=True,
        blocks_sizes=[64, 128, 256, 512],
        num_blocks=[2, 2, 2, 2],
        kernel_size=3,
        order='cnapdx',
        act='relu',
        attn=[False,False,False,False]
        ).cuda()

    print(model)
    from torchsummary import summary
    summary(model, (1, 256, 256))
"""

    @property
    def structure(self):
        return nn.Sequential(*self.layers)

class ResnetDecoder(nn.Module):
    def __init__(self, in_features, dim_embedding):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.lin_embedding1 = nn.Linear(in_features, dim_embedding)
        self.lin_embedding2 = nn.Linear(dim_embedding, dim_embedding)

    def forward(self, x):
        x = self.avg(x)
        x = torch.einsum("ijkl->ij", x)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.lin_embedding1(x)
        x = self.lin_embedding2(x)
        x = F.normalize(x, dim=1, p=2)
        # print(x.shape)
        return x


class ResNet(nn.Module):
    def __init__(self, dim_embedding, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(*args, **kwargs)
        self.decoder = ResnetDecoder(
            # self.encoder.layers[-1].blocks[-1].expanded_channels, dim_embedding
            kwargs['blocks_sizes'][-1], dim_embedding
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




def make_layer(inplanes, planes, block, n_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        # output size won't match input, so adjust residual
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )
    return nn.Sequential(
        block(inplanes, planes, stride, downsample),
        *[block(planes * block.expansion, planes) for _ in range(1, n_blocks)]
    )


def ResNetNew(block, layers, num_classes=1000):
    e = block.expansion

    resnet = nn.Sequential(
        Rearrange('b c h w -> b c h w', c=3, h=224, w=224),
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        make_layer(64,      64,  block, layers[0], stride=1),
        make_layer(64 * e,  128, block, layers[1], stride=2),
        make_layer(128 * e, 256, block, layers[2], stride=2),
        make_layer(256 * e, 512, block, layers[3], stride=2),
        # combined AvgPool and view in one averaging operation
        Reduce('b c h w -> b c', 'mean'),
        nn.Linear(512 * e, num_classes),
    )


    # initialization
    for m in resnet.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return resnet
"""
