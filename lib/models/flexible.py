from functools import partial, wraps
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U

from .network_utils import *
from .genradar import *
from lib import seed_everything

torch.set_printoptions(profile="full", linewidth=300, precision=10)


def is_sequence(s):
    return isinstance(s, (tuple, list))

def TamingBlock(in_chn, out_chn=None, dropout=0.0, reverse=False, **kwargs):
    out_chn = in_chn if out_chn is None else out_chn
    return nn.Sequential(
        ConvLayer(in_chn, out_chn, **kwargs),
        ConvLayer(out_chn, out_chn, dropout=dropout, **kwargs))

def OpenAIBlock(in_chn, out_chn, reverse=False, **kwargs):
    out_eff = out_chn // 4
    return nn.Sequential(
        ConvLayer(in_chn, out_eff, kernel_size=3 if not reverse else 1, **kwargs),
        *[ConvLayer(out_eff, out_eff, **kwargs)] * 2,
        ConvLayer(out_eff, out_chn, kernel_size=1 if not reverse else 3, **kwargs),
    )


class ResBlock(nn.Module):
    def __init__(self, block, in_chn, out_chn, block_gain, **kwargs):
        super().__init__()
        self.block_gain = block_gain
        self.shortcut = (
            ConvLayer(in_chn, out_chn, 1) if in_chn != out_chn else nn.Identity()
        )
        self.block = block(in_chn, out_chn, **kwargs)

    def __call__(self, x):
        return self.shortcut(x) + self.block(x) * self.block_gain


def layer(block, in_chn, out_chn, num_blocks, change_size, attn, block_gain, **kwargs):
    return nn.Sequential(
        OrderedDict(
            [
                *[
                    (
                        f"ResBlock: {g}",
                        ResBlock(
                            block,
                            in_chn if g == 0 else out_chn,
                            out_chn,
                            block_gain,
                            **kwargs,
                        ),
                    )
                    for g in range(num_blocks)
                ],
                (
                    "change_size",
                    change_size if change_size is not None else nn.Identity(),
                ),
                ('attention', AttnBlock(out_chn) if attn else nn.Identity())
            ]
        )
    )


@instantiate_or_load
class Encoder(nn.Sequential):
    def __init__(self, in_chn, layer_depths, num_blocks, attn, downs, block, **kwargs):
        self.in_chn = in_chn
        self.num_blocks = num_blocks
        self.attn = attn
        self.block = eval(block)
        self.kwargs = kwargs
        self.block_gain = 1 / sum(self.num_blocks)
        super().__init__(
            OrderedDict(
                [
                    (
                        "input",
                        nn.Conv2d(
                            in_chn,
                            layer_depths[0],
                            kernel_size=7,
                            stride=1,
                            padding=3,
                            # **kwargs,
                        ),
                    ),
                    *[
                        (
                            f"layer: {l}",
                            layer(
                                self.block,
                                layer_depths[l - 1 if l > 0 else l],
                                layer_depths[l],
                                num_blocks[l],
                                downsample(layer_depths[l], downs[l], **kwargs),
                                attn[l],
                                self.block_gain,
                                **kwargs,
                            ),
                        )
                        for l in range(len(layer_depths))
                    ],
                ]
            )
        )


@instantiate_or_load
class Decoder(nn.ModuleDict):
    def __init__(
        self, encoder, bottleneck=None, ups=["interp"] * 4, inp_size=256, **kwargs
    ):
        hooks, shapes_enc, x = analyze_encoder_structure(encoder, size=inp_size)
        print(
            f"Compressed input: {x.shape=} -> Seq length: {x.shape[-2] * x.shape[-1]}"
        )
        bottom = (
            nn.Identity("Missing bottleneck")
            if bottleneck is None
            else bottleneck.eval()
        )
        # bottom = nn.Sequential(nn.ReLU(), ConvLayer(1024, 8192, 1), ConvLayer(8192, 128, 1)).eval()
        x = bottom(x.detach())
        x = x[0] if isinstance(x, tuple) else x

        decoder = OrderedDict()
        for l, shape in enumerate(shapes_enc[::-1][:-1]):
            current_layer = layer(
                partial(encoder.block, reverse=True),
                x.shape[1],
                shape[1],
                encoder.num_blocks[::-1][l],
                upsample(shape[1], ups[l], **kwargs),
                attn=encoder.attn[::-1][l],
                block_gain=encoder.block_gain,
                **{**encoder.kwargs, **kwargs},
            ).eval()

            x = current_layer(x.detach())
            decoder[f"layer: {l}"] = current_layer
        decoder["output"] = ConvLayer(
            # shape[1], encoder.in_chn * 256, 1, act=encoder.kwargs['act'], order=encoder.kwargs['order']
            shape[1], encoder.in_chn, 1, act=encoder.kwargs['act'], order=encoder.kwargs['order']
            # **{**encoder.kwargs, **kwargs}
        )
        decoder = nn.Sequential(decoder)
        super().__init__({"encoder": encoder, "bottleneck": bottom, "decoder": decoder})

    def __call__(self, x):
        for n, segment in self.items():
            x = segment(x)
        return x


if __name__ == "__main__":
    # seed_everything(1337)
    model = Encoder(
        in_chn=1,
        layer_depths=[128, 256, 512, 1024],
        # block="OpenAIBlock",
        block="TamingBlock",
        num_blocks=[2, 2, 2, 2],
        attn=[False, False, False, False],
        downs=["max", "max", "max", None],
        act="relu",
        order="acnpdx",
        # order="cnapdx",
        # order="nadcpx",
        norm='GroupNorm',
        num_groups=32
    )
    # print(model)
    # print(model[1])
    # exit()
    dec = Decoder(model, ups=["interp", "interp", "interp", None])
    print(dec)
    # print(out.shape)
    # from torchsummary import summary
    # summary(dec, (1, 2, 2))
