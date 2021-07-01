from typing import List
from functools import partial
from enum import Enum
from typing import List, Tuple, Optional, Callable
import inspect, math
from collections import OrderedDict

# import attr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import torch.nn.init as init
import pandas as pd
import numpy as np

from lib import instantiate_or_load



class pResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels)
            # nn.LayerNorm(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class pEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(
            # nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.Conv2d(1, channels, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            pResidual(channels),
            pResidual(channels),
            # nn.Conv2d(channels, 256, 1)
        )

    def forward(self, x):
        return self.encoder(x)


class pDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.decoder = nn.Sequential(
            # nn.Conv2d(256, channels, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            pResidual(channels),
            pResidual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(channels),
            # nn.BatchNorm2d(channels),
            # nn.LayerNorm(channels),
            nn.ReLU(True),
            # nn.Conv2d(channels, 3 * 256, 1)

            # nn.Conv2d(channels, 6, 1)

            # nn.Conv2d(channels, 3, 1, bias=True)
            nn.Conv2d(channels, 1, 1, bias=False)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, n_in, n_out, n_layers, device, requires_grad):
        super().__init__()
        self.n_in: int = n_in
        self.n_out: int = n_out
        self.n_layers: int = n_layers
        self.device: torch.device = device
        self.requires_grad: bool = requires_grad

        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        make_conv     = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
        self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
            ('relu_1', nn.ReLU()),
            ('conv_1', make_conv(self.n_in,  self.n_hid, 3)),
            ('relu_2', nn.ReLU()),
            ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
            ('relu_3', nn.ReLU()),
            ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
            ('relu_4', nn.ReLU()),
            ('conv_4', make_conv(self.n_hid, self.n_out, 1)),]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)

class EncoderDalle(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_count:     int = 4
        self.n_hid:           int = 128
        self.n_blk_per_group: int = 2
        self.input_channels:  int = 3
        self.vocab_size:      int = 8192

        self.device:          torch.device = torch.device('cpu')
        self.requires_grad:       bool         = False
        self.use_mixed_precision: bool         = True

        blk_range  = range(self.n_blk_per_group)
        n_layers   = self.group_count * self.n_blk_per_group
        make_conv  = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
        make_blk   = partial(EncoderBlock, n_layers=n_layers, device=self.device,
                             requires_grad=self.requires_grad)

        self.blocks = nn.Sequential(OrderedDict([
            ('input', make_conv(self.input_channels, 1 * self.n_hid, 7)),
            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', make_blk(1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
            ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),
            ('group_2', nn.Sequential(OrderedDict([
		*[(f'block_{i + 1}', make_blk(1 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
		('pool', nn.MaxPool2d(kernel_size=2)),
	    ]))),
	    ('group_3', nn.Sequential(OrderedDict([
		*[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
		('pool', nn.MaxPool2d(kernel_size=2)),
	    ]))),
            ('group_4', nn.Sequential(OrderedDict([
		*[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
            ]))),
            ('output', nn.Sequential(OrderedDict([
		('relu', nn.ReLU()),
		('conv', make_conv(8 * self.n_hid, self.vocab_size, 1, use_float16=False)),
	    ]))),
	]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.input_channels:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
        if x.dtype != torch.float32:
            raise ValueError('input must have dtype torch.float32')
        return self.blocks(x)

class DecoderBlock(nn.Module):
    def __init__(self, n_in, n_out, n_layers, device=None, requires_grad=False):
        super().__init__()
        self.n_in: int = n_in
        self.n_out: int = n_out
        self.n_layers: int = n_layers
        self.device: torch.device = device
        self.requires_grad: bool = requires_grad

        self.n_hid = self.n_out // 4
        self.post_gain = 1 / (self.n_layers ** 2)

        make_conv     = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
        self.id_path  = make_conv(self.n_in, self.n_out, 1) if self.n_in != self.n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
            ('relu_1', nn.ReLU()),
            ('conv_1', make_conv(self.n_in,  self.n_hid, 1)),
            ('relu_2', nn.ReLU()),
            ('conv_2', make_conv(self.n_hid, self.n_hid, 3)),
            ('relu_3', nn.ReLU()),
            ('conv_3', make_conv(self.n_hid, self.n_hid, 3)),
            ('relu_4', nn.ReLU()),
            ('conv_4', make_conv(self.n_hid, self.n_out, 3)),]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)

class DecoderDalle(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_count:     int = 4
        self.n_init:          int = 128
        self.n_hid:           int = 128
        self.n_blk_per_group: int = 2
        self.output_channels: int = 3
        # self.output_channels: int = 6
        self.vocab_size:      int = 8192

        self.device:              torch.device = torch.device('cpu')
        self.requires_grad:       bool         = False
        self.use_mixed_precision: bool         = True

        blk_range  = range(self.n_blk_per_group)
        n_layers   = self.group_count * self.n_blk_per_group
        make_conv  = partial(Conv2d, device=self.device, requires_grad=self.requires_grad)
        make_blk   = partial(DecoderBlock, n_layers=n_layers, device=self.device,
                             requires_grad=self.requires_grad)

        self.blocks = nn.Sequential(OrderedDict([
	    ('input', make_conv(self.vocab_size, self.n_init, 1, use_float16=False)),
	    ('group_1', nn.Sequential(OrderedDict([
		*[(f'block_{i + 1}', make_blk(self.n_init if i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in blk_range],
		('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
	    ]))),
	    ('group_2', nn.Sequential(OrderedDict([
		*[(f'block_{i + 1}', make_blk(8 * self.n_hid if i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in blk_range],
		('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
	    ]))),
	    ('group_3', nn.Sequential(OrderedDict([
		*[(f'block_{i + 1}', make_blk(4 * self.n_hid if i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in blk_range],
		('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
	    ]))),
	    ('group_4', nn.Sequential(OrderedDict([
		*[(f'block_{i + 1}', make_blk(2 * self.n_hid if i == 0 else 1 * self.n_hid, 1 * self.n_hid)) for i in blk_range],
	    ]))),
	    ('output', nn.Sequential(OrderedDict([
		('relu', nn.ReLU()),
		# ('conv', make_conv(1 * self.n_hid, 2 * self.output_channels, 1)),
		('conv', make_conv(1 * self.n_hid,  self.output_channels, 1)),
	    ]))),
	]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 4:
            raise ValueError(f'input shape {x.shape} is not 4d')
        if x.shape[1] != self.vocab_size:
            raise ValueError(f'input has {x.shape[1]} channels but model built for {self.vocab_size}')
        if x.dtype != torch.float32:
            print(x.dtype)
            raise ValueError('input must have dtype torch.float32')
        return self.blocks(x)



@instantiate_or_load
class CosineAnnealing():
    def __init__(self, t0, t1, v0, v1, every):
        self.t0 = t0
        self.t1 = t1
        self.v0 = v0
        self.v1 = v1
        self.every = every if every <= t1 else t1
        self.num_called = 0
        self._v = v0
        if v0 == v1:
            print(f'-----Disabled Cosine Annealing-----')

    @property
    def v(self):
        return self._v

    def __call__(self, is_training):
        if (self.v0 != self.v1) and is_training:
            # self.num_called += 1
            if self.num_called % self.every == 0:
                # step = self.num_called // self.every
                # print('step', step)
                # caps at v1 when t1 is reached
                """ ramp from (t0, v0) -> (t1, v1) through a cosine schedule based on e \in [e0, e1] """
                # what fraction of the way through are we
                # alpha = max(0, min(1, (step - self.t0) / (self.t1 - self.t0)))
                alpha = max(0, min(1, (self.num_called - self.t0) / (self.t1 - self.t0)))
                alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
                self._v = alpha * self.v1 + (1 - alpha) * self.v0 # interpolate accordingly
            self.num_called += 1
        return self._v


@instantiate_or_load
class ExponentialAnnealing():
    # def __init__(self, start_temp=1.0, anneal_rate=1e-6, min_temp=0.5, every=1):
    def __init__(self, t0, t1, v0, v1, every):
        self.t0 = t0
        self.t1 = t1
        self._v = v0
        self.v0 = v0
        self.v1 = v1 # never falls below this temperature
        self.num_called = 0
        # after how many iterations annealing takes place
        self.every = every if every <= t1 else t1
        assert t0 != t1, 'division by zero in calculation of anneal_rate'
        self.anneal_rate = math.log(v0 / v1) / (t1 - t0)
        print(f'Anneal rate: {self.anneal_rate}')
        if self.anneal_rate == 0:
            print(f'-----Disabled Exponential Annealing-----')

    @property
    def temp(self):
        return self._v

    def __call__(self, is_training):
        if self.anneal_rate != 0 and is_training:
            # self.num_called += 1
            if self.num_called % self.every == 0:
                # step = self.num_called // self.every
                # print(f'Iteration: {self.num_called}: Old temperature... {self.temp:.6f}')
                # updated_temp = self.temp * math.exp(-self.anneal_rate * self.num_called)
                updated_temp = self.v0 * math.exp(-self.anneal_rate * self.num_called)
                # updated_temp = self.start_temp * math.exp(-self.anneal_rate * step)
                self._v = max(updated_temp, self.v1)
                # print(f'Iteration: {self.num_called}: New temperature... {self.temp:.6f}')
            self.num_called += 1
        return self._v


def simple_enc_dec(io_chn, dLatent, nLayers, nResBlocks, act, **kw):
    class ResBlock(nn.Module):
        def __init__(self, chan):
            super().__init__()
            self.net = nn.Sequential(
                # ConvLayer(chan, chan, 3, 1, 1, act=act, order='acnpdx'),
                # ConvLayer(chan, chan, 1, act=act, order='acnpdx'),

                ConvLayer(chan, chan, 3, act=act),
                ConvLayer(chan, chan, 3, act=act),
                ConvLayer(chan, chan, 1),
            )

        def __call__(self, x):
            return self.net(x) + x

    # enc_chans = [128, 256, 512, 1024]
    # dec_chans = enc_chans[::-1]
    # print(enc_chans)
    # print(dec_chans)
    # exit()

    enc_chans = [dLatent] * nLayers
    dec_chans = enc_chans[:]

    enc_chans = [io_chn, *enc_chans]
    dec_chans = [dec_chans[0], *dec_chans]

    enc_layers, dec_layers = [], []

    for i, ((enc_in, enc_out), (dec_in, dec_out)) in enumerate(zip(
            *(zip(b, b[1:]) for b in (enc_chans, dec_chans))), 1
    ):
        enc_layers.append(ConvLayer(enc_in, enc_out, 4, 2, 1, act=act if i < nLayers else None))
        dec_layers.append(ConvLayer(dec_in, dec_out, 4, 2, 1, act=act, **kw, transpose=True))

    # insert ResNet blocks after encoder and before decoder
    for _ in range(nResBlocks):
        enc_layers.append(ResBlock(enc_chans[-1]))
        dec_layers.insert(0, ResBlock(dec_chans[0]))

    dec_layers.append(ConvLayer(dec_chans[-1], io_chn, 1))
    return nn.Sequential(*enc_layers), nn.Sequential(*dec_layers)


norm_types = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
)

def transfrom_2d_to_3d(m):
    m_3d = list()
    for layer in m.modules():
        name = layer.__class__.__name__
        _2D_params = dict()
        if "Conv" in name:
            _2D_params['in_channels'] = layer.in_channels
            _2D_params['out_channels'] = layer.out_channels
            _2D_params['kernel_size'] = layer.kernel_size[0]
            _2D_params['stride'] = layer.stride[0]
            _2D_params['padding'] = layer.padding[0]
            m_3d.append(nn.Conv3d(**_2D_params))
        elif "Norm" in name:
            _2D_params['num_features'] = layer.num_features
            m_3d.append(nn.BatchNorm3d(**_2D_params))
        elif "AdaptiveAvgPool" in name:
            _2D_params['output_size'] = layer.output_size
            m_3d.append(nn.AdaptiveAvgPool3d(**_2D_params))
        elif "AdaptiveMaxPool" in name:
            _2D_params['output_size'] = layer.output_size
            m_3d.append(nn.AdaptiveMaxPool3d(**_2D_params))
        elif "MaxPool" in name:
            _2D_params['kernel_size'] = layer.kernel_size
            _2D_params['stride'] = layer.stride
            _2D_params['padding'] = layer.padding
            m_3d.append(nn.MaxPool3d(**_2D_params))
        elif "AvgPool" in name:
            _2D_params['kernel_size'] = layer.kernel_size
            _2D_params['stride'] = layer.stride
            _2D_params['padding'] = layer.padding
            m_3d.append(nn.AvgPool3d(**_2D_params))
        elif '2d' not in name:# act functions etc.
            m_3d.append(layer)
        else:
            raise RuntimeError(f'Conversion from {m} to 3d not implemented')
    return nn.Sequential(*m_3d)


class Hook:
    "Create a hook on `m` with `hook_func`."

    def __init__(self, m, hook_func, is_forward=True, detach=True, cpu=False):

        self.hook_func = hook_func
        self.detach = detach
        self.cpu = cpu

        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.stored, self.removed = None, False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        # if self.detach:
        # input = to_detach(input, cpu=self.cpu)
        # output = to_detach(output, cpu=self.cpu)
        self.stored = self.hook_func(module, input, output)
        # print("hook triggered")

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True
            # print("REMOVED")

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        # print("EXIT!!!!!!!!!!!")
        self.remove()


class Hooks:
    "Create several hooks on the modules in `ms` with `hook_func`."

    def __init__(self, ms, hook_func, is_forward=True, detach=True, cpu=False):
        self.hooks = [Hook(m, hook_func, is_forward, detach, cpu) for m in ms]

    def __getitem__(self, i):
        return self.hooks[i]

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def stored(self):
        return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks:
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        # print("EXIT!!!!!!!!!!!")
        self.remove()


def hook_outputs(modules, detach=True, cpu=False, grad=False):
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, cpu=cpu, is_forward=not grad)


def _hook_inner(m, i, o):
    return o if isinstance(o, torch.Tensor) else list(o)


def analyze_encoder_structure(m, size=(64, 64)):
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    hooks = hook_outputs(m, detach=False)
    encoder_output = dummy_eval(m, size=size)
    return hooks, [o.stored.shape for o in hooks], encoder_output


def dummy_eval(m, size=(64, 64)):
    "Evaluate `m` on a dummy input of a certain `size`"
    chn_in = in_channels(m)
    size = size if isinstance(size, (list, tuple)) else (size, size)
    x = torch.rand(1, chn_in, *size).requires_grad_(False)
    if next(m.parameters()).is_cuda:
        x = x.cuda()
    with torch.no_grad():
        return m.eval()(x)


def in_channels(m):
    "Return the shape of the first weight layer in `m`."
    for n, l in m.named_modules():
        if hasattr(l, "weight"):
            s = l.weight.shape[1]  # return first channel dim
            return s
        if isinstance(l, nn.ModuleDict):
            s = l['c'].weight.shape[1]
            return s
    raise Exception(f"No weight layer:{type(m)}")


def AdaptiveAvgPool(sz=1, ndim=2, **kw):
    "nn.AdaptiveAvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AdaptiveAvgPool{ndim}d")(sz)

def AdaptiveMaxPool(sz=1, ndim=2, **kw):
    "nn.AdaptiveAvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AdaptiveMaxPool{ndim}d")(sz)

def MaxPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False, **kw):
    "nn.MaxPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"MaxPool{ndim}d")(ks, stride=stride, padding=padding)

def AvgPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False, **kw):
    "nn.AvgPool layer for `ndim`"
    assert 1 <= ndim <= 3
    return getattr(nn, f"AvgPool{ndim}d")(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)

def _pool_func(prefix, **kwargs):
    # pool = getattr(nn, f"{prefix}{ndim}d")(**kwargs)
    return eval(prefix)(**kwargs)

# def get_pooling(pooling, *args, **kwargs):
    # print(pooling)
    # return dict([("adapt_avg_pool", _pool_func('AdaptiveAvgPool', *args, **kwargs)),
                 # ("adapt_max_pool", _pool_func('AdaptiveMaxPool', *args, **kwargs)),
                 # ("max_pool", _pool_func('MaxPool', *args, **kwargs)),
                 # ("avg_pool", _pool_func('AvgPool', *args, **kwargs))]).get(pooling)


def _norm_func(prefix, num_channels, ndim=2, num_groups=None, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    if prefix in ('BatchNorm', 'InstanceNorm'):
        # bn = getattr(nn, f"{prefix}{ndim}d")(out_channels, **kwargs)
        bn = getattr(nn, f"{prefix}{ndim}d")(num_channels)
    elif prefix in ('LayerNorm', 'GroupNorm'):
        # bn = getattr(nn, f"{prefix}")(**kwargs)
        bn = getattr(nn, f"{prefix}")(num_groups, num_channels)
    else:
        return nn.Identity()
    return bn


def get_norm(norm, *args, **kwargs):
    return _norm_func(norm, *args, **kwargs)
    # print('norm', norm)
    # return dict([("batch", _norm_func('BatchNorm', *args, **kwargs)),
                 # ("group", _norm_func('GroupNorm', *args, **kwargs)),
                 # ("layer", _norm_func('LayerNorm', *args, **kwargs)),
                 # ("instance", _norm_func('InstanceNorm', *args, **kwargs))]).get(norm, nn.Identity())
"""
def get_activation(activation, **kwargs):
    return dict([("relu", nn.ReLU(inplace=True)),
            ("leaky_relu", nn.LeakyReLU(kwargs.get('negative_slope', 0.01),
                                        kwargs.get('inplace', True))),
            ("tanh", nn.Tanh()),
            ("gelu", nn.GELU()),
            ("silu", nn.SiLU(inplace=True)),
            ("selu", nn.SELU(inplace=True))]).get(activation, nn.Identity())
"""

def get_activation(activation, **kwargs):
    return dict([("relu", nn.ReLU()),
                 ("leaky_relu", nn.LeakyReLU(kwargs.get('negative_slope', 0.01))),
                 ("tanh", nn.Tanh()),
                 ("gelu", nn.GELU()),
                 ("silu", nn.SiLU()),
                 ("selu", nn.SELU())]).get(activation, nn.Identity())

def get_conv(ndim=2, transpose=False):
    "Return the proper conv `ndim` module, potentially `transposed`."
    assert 1 <= ndim <= 3
    # return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')


def get_dropout(ndim=2, dropout=0.5, **kwargs):
    "Return the proper dropout `ndim` module."
    assert 1 <= ndim <= 3
    return getattr(nn, f'Dropout{ndim}d')(dropout)

"""
class Conv2d(nn.Module):
    def __init__(self, in_chn: int,
                 out_chn: int,
                 kw: int,
                 use_float16=True,
                 device='cpu',
                 requires_grad=False):
        super().__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.kw = kw
        self.use_float16 = use_float16
        self.device = device
        self.requires_grad = requires_grad

        w = torch.empty((self.out_chn, self.in_chn, self.kw, self.kw), dtype=torch.float32,
                        device=self.device, requires_grad=self.requires_grad)
        w.normal_(std=1 / math.sqrt(self.in_chn * self.kw ** 2))

        b = torch.zeros((self.out_chn,), dtype=torch.float32, device=self.device,
                        requires_grad=self.requires_grad)
        self.weight, self.bias = nn.Parameter(w), nn.Parameter(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if self.use_float16 and 'cuda' in self.w.device.type:
            # if x.dtype != torch.float16:
                # x = x.half()
            # w, b = self.w.half(), self.b.half()
        # else:
            # if x.dtype != torch.float32:
                # x = x.float()
        w, b = self.weight, self.bias
        return F.conv2d(x, w, b, padding=(self.kw - 1) // 2)
"""

class Conv2d(nn.Module):
    # cf. https://github.com/openai/DALL-E/blob/master/dall_e/utils.py
    def __init__(self,
                 in_chn: int,
                 out_chn: int,
                 kw: int,
                 bias: bool = True,
                 stride: int = 1,
                 padding: int = 0,
                 transpose=False,
                 **kwargs
    ):

        super().__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.kw = kw
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.transpose = transpose

        # w = torch.empty((out_chn, in_chn, kw, kw), dtype=torch.float32)\
                 # .normal_(std=1 / math.sqrt(in_chn * kw ** 2))
        # b = torch.zeros((out_chn,), dtype=torch.float32)
        # self.w, self.b = nn.Parameter(w), nn.Parameter(b)
        self.conv = nn.Conv2d(in_chn, out_chn, kw, padding=(self.kw - 1) // 2, bias=True)
        # nn.init.normal_(self.conv.weight, 1.0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if not self.transposed:
            # return F.conv2d(x, self.w, self.b, padding=(self.kw - 1) // 2)
        # else:
            # return F.conv_transpose2d(x, self.w, self.b, padding=(self.kw - 1) // 2)
        return self.conv(x)
# """

class ConvLayer(nn.Sequential):
# class ConvLayer(nn.ModuleDict):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        norm=None,
        act=None,
        pool=None,
        bias=None,
        ndim=2,
        dropout=None,
        norm_1st=True,
        transpose=False,
        xtra=None,
        order='cnapdx',
        init_conv=False,
        **kwargs,
    ):
        out_channels = out_channels if out_channels is not None else in_channels
        if padding is None:
            padding = (kernel_size - 1) // 2 if not transpose else 0
        # bn = norm in ('BatchNorm', 'BatchNormZero')
        # inn = norm in ('InstanceNorm', 'InstanceZeroNorm')
        if bias is None:
            # bias = not (bn or inn)
            bias = (norm != 'BatchNorm' and norm != 'InstanceNorm')
        bias = False # TODO: https://stats.stackexchange.com/questions/505027/can-i-use-the-mse-loss-function-along-with-a-sigmoid-activation-in-my-vae
            # enabling biases can cause collapsing to mean value of input images

        """
        conv = Conv2d(in_chn=in_channels,
                      out_chn=out_channels,
                      kw=kernel_size)
        """
        conv_func = get_conv(ndim, transpose=transpose)
        conv = conv_func(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            stride=stride,
            padding=padding,
            # **kwargs,
        )

        if init_conv:
            nn.init.normal_(conv.weight, std=1 / math.sqrt(in_channels * kernel_size ** 2))


        d = OrderedDict()
        for el in order:
            if el == 'c':
                d['c'] = conv
            if el == 'n' and norm is not None:
                if 'c' in d: # conv was first
                    d[el] = get_norm(norm, num_channels=out_channels, ndim=ndim, **kwargs)
                else:
                    d[el] = get_norm(norm, num_channels=in_channels, ndim=ndim, **kwargs)
            if el == 'a' and act is not None:
                d[el] = get_activation(act, **kwargs)
            if el == 'p' and pool is not None:
                d[el] = _pool_func(pool, ndim=ndim, **kwargs)
            if el == 'd' and dropout is not None:
                d[el] = get_dropout(ndim=ndim, dropout=dropout, **kwargs)
            if el == 'x' and xtra is not None:
                d[el] = xtra

        # {el: modules[el] for el in order if el in modules}

        super().__init__(d)
    # def forward(self, x):
        # for layer in self.values():
            # x = layer(x)
        # return x
        # super().__init__(*[modules[el] for el in order if el in modules])

class PixelShuffle_ICNR(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        scale=2,
        blur=False,
        norm='WeightNorm',
        act=None,
        **kwargs
    ):
        out_channels = out_channels if out_channels is not None else in_channels
        layers = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=out_channels * (scale ** 2),
                kernel_size=1,
                norm=norm,
                act=act,
                **kwargs
                # bias_std=0,
            ),
            nn.PixelShuffle(scale),
        ]
        layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))
        if blur:
            layers += [nn.ReplicationPad2d((1, 0, 1, 0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)

def downsample(in_chn, mode='conv', **kwargs):
    if mode == 'conv':
        return  ConvLayer(in_chn, in_chn, kernel_size=3, stride=2, padding=(1, 1), **kwargs)
    elif mode == 'avg':
        return  AvgPool(ks=2, stride=2)
    elif mode == 'max':
        return  MaxPool(ks=2, stride=2)
    else:
        return nn.Identity()

def upsample(in_chn, mode='interp', **kwargs):
    if mode == 'interp':
        return nn.Upsample(scale_factor=2, mode='nearest')
    elif mode == 'pixel_shuffle':
        return PixelShuffle_ICNR(in_chn, scale=2, **kwargs)
    elif mode == 'deconv':
        return ConvLayer(in_chn, in_chn, 4, stride=2, padding=1, transpose=True, **kwargs)
    else:
        return nn.Identity()

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm=None, act=None, scale_factor=2.0, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = ConvLayer(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              norm=norm,
                              act=act)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)


class InterpolateToOrig(nn.Module):
    "Merge a shortcut with the result of the module by adding them or concatenating them if `dense=True`."

    def __init__(self, mode="nearest"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        if x.orig.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(x, x.orig.shape[-2:], mode=self.mode)
        return x


class MergeLayer(nn.Module):
    "Merge a shortcut with the result of the module by adding them or concatenating them if `dense=True`."

    def __init__(self, dense = False):
        super().__init__()
        self.dense = dense

    def forward(self, x):
        return torch.cat([x, x.orig], dim=1) if self.dense else (x + x.orig)


# class SequentialEx(nn.Module):
class SequentialEx(nn.ModuleDict):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

    # def __init__(self, *layers):
    def __init__(self, segments):
        super().__init__()
        # self.layers = nn.Module(layers)
        self.segments = segments

    def forward(self, x):
        res = x
        for seg_name, seg in self.segments.items():
            res.orig = x
            nres = seg(res)
            print('Passed through segment:', seg_name)
            # We have to remove res.orig to avoid hanging refs and therefore memory leaks
            res.orig = None
            res = nres
        return res

    def __getitem__(self, seg):
        return self.segments[seg]

    def append(self, l):
        return self.segments.update(l)

    def extend(self, l):
        return self.layers.extend(l)

    def insert(self, i, l):
        return self.layers.insert(i, l)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        # h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_




def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net



def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni, nf, h, w = x.shape
    ni2 = int(ni / (scale ** 2))
    k = init(x.new_zeros([ni2, nf, h, w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale ** 2)
    return k.contiguous().view([nf, ni, h, w]).transpose(0, 1)




"""
def init_linear(m, act_func=None, init="auto", bias_std=0.01):
    if getattr(m, "bias", None) is not None and bias_std is not None:
        if bias_std != 0:
            normal_(m.bias, 0, bias_std)
        else:
            m.bias.data.zero_()
    if init == "auto":
        if act_func in (F.relu_, F.leaky_relu_):
            init = kaiming_uniform_
        else:
            init = getattr(act_func.__class__, "__default_init__", None)
        if init is None:
            init = getattr(act_func, "__default_init__", None)
    if init is not None:
        init(m.weight)

def init_default(m, func):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, "weight"):
            func(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
        # if hasattr(m, "bias") and hasattr(m.bias, "data"):
            init.constant_(m.bias.data, 0.0)
            # m.bias.data.fill_(0.0)
        print(m.__class__.__name__)

    return m


def apply_leaf(m, f):
    "Apply `f` to children of `m`."
    c = m.children()
    if isinstance(m, nn.Module):
        f(m)
    for l in c:
        apply_leaf(l, f)


def cond_init(m, func):
    "Apply `init_default` to `m` unless it's a batchnorm module"
    if (not isinstance(m, norm_types)) and requires_grad(m):
        init_default(m, func)


def apply_init(m, func=nn.init.kaiming_normal_):
    "Initialize all non-batchnorm layers of `m` with `func`."
    apply_leaf(m, partial(cond_init, func=func))
"""
