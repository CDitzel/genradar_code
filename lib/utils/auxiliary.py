import random
import sys
import glob
import os
import inspect
from subprocess import Popen, PIPE
import functools
import socket
import pickle
import math
import traceback
import warnings
import platform
import lmdb


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torchvision import utils as U
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torchvision.transforms.functional as tf
from dataflow.utils.utils import fix_rng_seed
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange

import lib
from .config_handler import *


def open_lmdb(lmdb_path):
    isdir = os.path.isdir(lmdb_path)
    # It's OK to use super large map_size on Linux, but not on other platforms
    # See: https://github.com/NVIDIA/DIGITS/issues/206
    map_size = 1099511627776 * 2 if platform.system() == "Linux" else 128 * 10 ** 6
    # need sync() at the end
    return lmdb.open(lmdb_path, subdir=isdir, map_size=map_size, readonly=False, meminit=False, map_async=True)


def put_or_grow(txn, key, value):
    # put data into lmdb, and doubling the size if full.
    # Ref: https://github.com/NVIDIA/DIGITS/pull/209/files
    try:
        txn.put(key, value)
        return txn
    except lmdb.MapFullError:
        pass
    txn.abort()
    curr_size = self._db.info()["map_size"]
    new_size = curr_size * 2
    logger.info("Doubling LMDB map_size to {new_size / 10 ** 9:.2f}GB")
    self._db.set_mapsize(new_size)
    txn = self._db.begin(write=True)
    return put_or_grow(txn, key, value)

def write_lmdb(df, lmdb_path):
    db = open_lmdb(lmdb_path)
    with get_tqdm(total=df.size) as pbar:
        # LMDB transaction is not exception-safe!
        # although it has a context manager interface
        txn = db.begin(write=True)
        for idx, dp in enumerate(self.ds):
            idr = f"{idx:08}r".encode("ascii")
            idc = f"{idx:08}c".encode("ascii")
            txn = put_or_grow(txn, idr, pickle.dumps(dp["rad"]))
            txn = put_or_grow(txn, idc, pickle.dumps(dp["cam"]))
            pbar.update()
            if (idx + 1) % self._write_frequency == 0:
                txn.commit()
                txn = self._db.begin(write=True)
        txn.commit()
        logger.info("Flushing database ...")
        self._db.sync()
        self._db.close()


def zero_grads(m):
    for param in m.parameters():
        param.grad = None

def weight(nll_loss, g_loss, last_layer=None):
    """ the less significant the critic is, i.e. for a decrease in g_loss,
    the overall loss becomes less sensitive to changes in the critics weights/params.
    So bump up the critics weight which is below ratio. """
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    # print('nll_grads', nll_grads.shape)
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    # print('g_grads', g_grads.shape)
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    return torch.clamp(d_weight, 0.0, 1e4).detach()


def clampWeights(m):
    if type(m) != nn.BatchNorm2d and type(m) != nn.Sequential:
        for p in m.parameters():
            p.data.clamp_(-opt.clip, opt.clip)


def top_k_ratio(logits, thres=0.5):
    # num_logits = logits.shape[-1]
    # k = max(int((1 - thres) * num_logits), 1)
    num_cam_logits = logits.shape[-1] // 2
    k = max(int((1 - thres) * num_cam_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    return probs.scatter_(1, ind, val)


def eval_decorator(fn):
    def inner(module, *args, **kwargs):
        was_training = module.training
        module.eval()
        out = fn(module, *args, **kwargs)
        module.train(was_training)
        return out
    return inner


def instantiate_or_load(cls):
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        state = None
        if "load_path" in kwargs and kwargs["load_path"] is not None:
            try:
                deserialized_ckpt = torch.load(kwargs["load_path"])
                state = deserialized_ckpt["model_state_dict"]
                path, stem = os.path.split(kwargs["load_path"])
                if 'compressor' in deserialized_ckpt['config']:
                    kwargs = deserialized_ckpt['config']["compressor"]["args"]
                else:
                    kwargs = deserialized_ckpt['config']["model"]["args"]
            except KeyError as e: # to be compatible with older ckpts with config.yaml
                print(f'Loading associated config from file rather than ckpt')
                kwargs, _, cfg_dict = load_config(os.path.join(path, "config.yaml"))
                # kwargs = kwargs["model"]["args"]
                kwargs = kwargs["compressor"]["args"]
            except Exception as e:
                print(f'Problem with loading {kwargs["load_path"]}: {e}')

        for k, v in kwargs.items():
            if isinstance(v, dict) and "name" in v:
                kwargs[k] = eval(f'lib.{v["name"]}')(**v["args"])
        try:
            c = cls(*args, **kwargs)
            if state is not None:
                print(
                    f"Loading {c.__class__.__name__}s state from file: {stem} "
                    + f"{c.load_state_dict(state, strict=True)}"
                )
            else:
                print(f'Instantiated {c.__class__.__name__}')
            return c
        except Exception as e:
            print(f'Problem with class {cls}: {e}')
            print(traceback.format_exc())

    return wrapper


make_grid = lambda s: U.make_grid(s, nrow=s.shape[0], normalize=True, padding=1, scale_each=False)

def layout(*samples, only_last=None, mod='cam'):
    if mod == 'rad':
        samples = [(torch.rot90(s, 1, [-2, -1])) for s in samples]
    return torch.cat([make_grid(s[-only_last:] if only_last is not None else s) for s in samples], dim=1)


def is_a(obj, cls):
    return isinstance(obj, cls)


def disable_feature_for_certain_model_paramters(model, feature):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.Embedding, torch.nn.InstanceNorm2d)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    # print(decay)
    # print(no_decay)
    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay

    left_overs = param_dict.keys() - union_params
    if left_overs:
        print(f'Putting {(left_overs)} into no_decay set')
    no_decay.update(param_dict.keys() - union_params)
    union_params = decay | no_decay
    # exit()
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], feature: 0.0},
    ]
    return optim_groups
    """
def disable_feature_for_certain_model_paramters(model, feature, remains='decay'):
    # separate  parameters to those that will and won't get weight decay
    decay, no_decay, others = set(), set(), set()
    whitelist = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
    blacklist = (nn.LayerNorm, nn.BatchNorm2d, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if p.requires_grad:
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                else:
                    others.add(fpn)

    # TODO: find smarter way to deal with custom params, i.e. nn.Parameters
    if remains == 'decay':
        decay = (others - decay).intersection(others - no_decay) | decay
    else:
        no_decay = (others - decay).intersection(others - no_decay) | no_decay
    print(decay)
    print(no_decay)
    print(others)
    print(f'---------Putting {others=} in set {remains}----------')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], feature: 0.0},
    ]

    return optim_groups
    """



class TensorboardLogger:
    def __init__(self, res_dir):
        self.tb = SummaryWriter(res_dir)

    def disp(self, name, phase, epo, image):
        self.tb.add_image(f"{name} {phase} lastBatch", image, epo)
        self.tb.flush()

    def disp_attn(self, name, phase, epo, image):
        self.tb.add_image(f"{name} {phase} attn", image, epo)
        self.tb.flush()

    def disps(self, name, images, epo):
        self.tb.add_images(f"{name} lastBatch", images, epo)
        self.tb.flush()

    def plot(self, name, phase, epo, **kw):
        for key, val in kw.items():
            if isinstance(val, float):
                # self.tb.add_scalar(f"{name} {key}/" + phase, val, epo)
                self.tb.add_scalar(f"{key}/" + phase, val, epo)
            else:
                self.tb.add_scalars(f"{key}/" + phase, val, epo)
                self.tb.flush()
        self.tb.flush()

    def hist(self, name, epo, model): # for weights and grad display
        if isinstance(model, torch.nn.Module):
            for key, val in model.named_parameters():
                if val.grad is not None:  # excl. mod not used in computation
                    key = key.replace(".", "/")
                    self.tb.add_histogram(f"{name} {key}", to_np(val), epo)
                    self.tb.add_histogram(f"{name} {key}/grad", to_np(val.grad), epo)
                    self.tb.flush()
        elif isinstance(model, nn.Parameter):
            self.tb.add_histogram(f"{name} {codebook}", to_np(model), epo)
            if model.grad is not None:  # excl. mod not used in computation
                self.tb.add_histogram(f"{name} {codebook}/grad", to_np(model.grad), epo)
            self.tb.flush()


    def dist(self, name, phase, epo, distribution, bins='auto'): # for codebook indices
        self.tb.add_histogram(f"{name}/{phase}/token_dist", to_np(distribution), epo, bins=bins, max_bins=8)

    def text(self, text_string, tag='cfg'):
        self.tb.add_text(tag, text_string.replace("  ", "--").replace("\n", "  \n"))

    def fig(self, tag, epo, figure):
        self.tb.add_figure(tag, figure, global_step=epo, close=True, walltime=None)

def scrape_files(top_level_path, suffix, recursive=True):
    path = Path(top_level_path)
    print(f"Searching for {suffix} files in", path)
    assert path.is_dir(), f"select a directory containing {suffix} files"
    if recursive:
        file_paths = sorted(path.glob(f"**/*.{suffix}"))
    else:
        file_paths = sorted(path.glob(f"*.{suffix}"))
    if not file_paths:
        raise RuntimeError(f"No {suffix} files found")
    return list(str(path.resolve()) for path in file_paths)


def showcase(model_name, rad, cam, **kw):
    rescaling_via_batch(cam)
    if model_name in ["SoftAttention", "Incitedattention"]:
        augCam, fp, tp, tn, fn = prepare_attentive_cam_view(cam, **kw)
    elif model_name == "RaCamNetImplDist":
        confused_images, fp, tp, tn, fn = color_confusion(cam, **kw)
        augCam = render_probabilities(confused_images, **kw)
    elif model_name == "LocalizedAttention":
        attention_upscaled = F.interpolate(
            kw["loc"],
            size=(cam.shape[2], cam.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        att_cam = superimpose_attention(attention_upscaled, cam, **kw)
        return U.make_grid(att_cam, padding=0, normalize=True)

    camH, camW = augCam.shape[-2:]

    if "rad_cam" in kw:
        fp_rad = rad[fp].cpu()
        fp_cam = augCam[fp]
        cam_rad = kw["cam_rad"].cpu()
        rad_cam = kw["rad_cam"].cpu()
        fp_cam_rad = cam_rad[fp]
        fp_rad_cam = rad_cam[fp]
        fp_cam = rearrange([fp_cam, fp_rad_cam], "n b c h w -> (b n) c h w")

        augCam = torch.cat(
            (fp_cam, augCam[fn].cpu(), augCam[tn].cpu(), augCam[tp].cpu()), 0
        )
    if rad.ndim > 4:  # cube
        if any("rad_cam", "cam_rad") in kw:
            fp_rad = rearrange([fp_rad, fp_cam_rad], "n b c h w d -> (b n) c h w d")
            rad = torch.cat((fp_rad, rad[fn].cpu(), rad[tn].cpu(), rad[tp].cpu()), 0)
            radH, radW, radD = rad.shape[-3:]
            rad = torch.rot90(rad, 1, [-3, -2])
            rA = rad.mean(-2)
            rD = rad.mean(-1)
            # rA = rad.max(-2)[0]
            # rD = rad.max(-1)[0]
        if camW < (radW + radD):  # overlay rA over rD on the right
            rD[..., -32:] = rA
        else:  # if there is enough space due to larger cam then just cat
            rD = torch.cat((rD, rA), -1)
            radW = rD.shape[-1]
    else:
        if "rad_cam" in kw:
            fp_rad = rearrange([fp_rad, fp_cam_rad], "n b c h w -> (b n) c h w")
            # rad = torch.cat((fp_rad, rad[~fp].cpu()), 0)
            rad = torch.cat((fp_rad, rad[fn].cpu(), rad[tn].cpu(), rad[tp].cpu()), 0)
            radH, radW = rad.shape[-2:]
            rad = torch.rot90(rad, 1, [-2, -1])
            rad = rad[:, :1]  # handling magnitude/phase channel, i.e. display only mag
            rD = rad.mean(-3).unsqueeze(1)
            # rD = rad.max(-3)[0].unsqueeze(1)
    padW = int(math.ceil((camW - radW) / 2))
    rD = tf.pad(rD, (padW, 1), fill=0.0)
    augRad = U.make_grid(
        rD, nrow=rad.shape[0], padding=0, scale_each=True, normalize=True
    )
    augCam = U.make_grid(
        augCam, nrow=augCam.shape[0], padding=0, scale_each=True, normalize=True
    )
    return torch.cat([augRad.cpu(), augCam.cpu()], 1)


def get_Host_name_IP():
    try:
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        # print("Hostname: ",host_name)
        # print("IP : ",host_ip)
    except:
        print("Unable to get Hostname and IP")
    else:
        return host_name, host_ip


class AvgMeter:
    """Computes and stores the average and current value"""

    # def __init__(self, scaled=False):
    def __init__(self):
        self.reset()

    def reset(self):
        self._avg = 0.0
        self.sum = 0.0
        self.count = 1e-19

    def update(self, val, weight=1):
        # self.sum += val * weight if self.scaled else val
        self.sum += val * weight
        self.count += weight
        self._avg = self.sum / self.count

    def __call__(self, val, weight=1.0):
        # self.sum += val * weight if self.scaled else val
        self.sum += val * weight
        self.count += weight
        self._avg = self.sum / self.count

    @property
    def avg(self):
        return self._avg


def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print("removed existing directory...")
        else:
            return
        os.makedirs(path)


def save_model(epo, model, optim, fp, best_loss_total, config=None):
    ckpt_files = glob.glob(fp +'/*.ckpt')
    for ckpt_file in ckpt_files:
        os.remove(ckpt_file)
    fp = f"{fp}epo{epo:03}__{best_loss_total=:.5f}"
    torch.save(
        {
            "epo": epo,
            "config": config if config is not None else {},
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
        },
        fp + ".ckpt",
    )

def save_model_reg(epo, model, optim, fp, loss_total, config=None):
    fp = f"{fp}epo{epo:03}__{loss_total=:.5f}"
    torch.save(
        {
            "epo": epo,
            "config": config if config is not None else {},
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
        },
        fp + ".ckpt_reg",
    )


def print_input(layer, input):
    for i in input:
        print(f"input: {layer.__name__}: {i.shape}")


def verbose_output(cls):
    """ turns decorated class into verbose torch.Module """

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        c = cls(*args, **kwargs)
        for name, layer in c.named_modules():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(
                    f"output: {layer.__name__}: {output.shape}"
                )
            )
        return c

    return wrapper


def init_weights(net, init_type="normal", init_gain=0.02, **kwargs):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and "Conv" in classname or "Linear" in classname:

            if init_type == "normal":
                init.normal_(m.weight, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight, gain=init_gain)
            elif init_type == "constant":
                init.constant_(m.weight, val=1.0)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, norm_types):  # "BatchNorm2d" in classname
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            # print('CLASSNAME', classname)
            init.normal_(m.weight, 1.0, init_gain)
            init.constant_(m.bias, 0.0)
            # init.constant_(m.weight, 1.0)
            # init.constant_(m.bias, 1.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)  # apply the initialization function <init_func>


def update_learning_rate(opt, lrs):
    """Update learning rates for all the networks; called at the end of every epoch"""
    old_lr = opt[0].param_groups[0]["lr"]
    # for scheduler in self.schedulers:
    if self.opt.lr_policy == "plateau":
        lrs.step(self.metric)
    else:
        lrs.step()
    return opt[0].param_groups[0]["lr"]


def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group["lr"] *= decay_factor


def check_init(m):
    """ Check initialization method of modules, set constant_(1.0) in init before """
    for k, c in m.named_parameters():
        check = c.bool().all()
        if check.item() is not True:
            raise Exception


# m.apply(check_init)


def lambda_lr_rule(epoch, max_epoch):
    lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs)
    return lr_l / float(opt.n_epochs_decay + 1)


def save_networks(self, save_path, epoch):
    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
        torch.save(net.module.cpu().state_dict(), save_path)
        net.cuda(self.gpu_ids[0])
    else:
        torch.save(net.cpu().state_dict(), save_path)


class GNoise(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f"(mean={self.std}, std={self.mean})"



""" In scaling, you’re changing the range of your data while in normalization
you’re mostly changing the shape of the distribution of your data.  Not every
distribution with the same center and the same width automatically has the
correct "bell" shape. Standardization shifts and scales a distribution; it does
not change its shape.  """

def rescaling_via_batch(t):
    return t.add_(-torch.min(t)).div_(torch.max(t) - torch.min(t) + 1e-10)


def standardization_via_batch(t):
    t_flattened = t.view(t.shape[0], t.shape[1], -1)
    mean = t_flattened.mean(2)
    mean = mean.sum(0) / t.shape[0]
    std = t_flattened.std(2).sum(0) / t.shape[0]
    # if t.shape[-1] == 640:  # make broadbasting work for cam
    if t.shape[1] == 3:  # make broadbasting work for cam, i.e. 3 channels
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    return t.add_(-mean).div_(std)


def ImageNet_normalization(t):
    chn_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    chn_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return t.add_(-chn_mean).div_(chn_std)


def plus_minus_one(t):
    if t.shape[1] == 3:
        sub_val = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        div_val = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    else:
        sub_val = torch.tensor([0.5]).view(-1, 1, 1)
        div_val = torch.tensor([0.5]).view(-1, 1, 1)
    return t.sub_(sub_val).div_(div_val)


def plus_minus_zeroPointFive(t):
    if t.shape[1] == 3:
        sub_val = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    else:
        sub_val = torch.tensor([0.5]).view(-1, 1, 1)
    return t.sub_(sub_val)


def rescaling_via_train_data_cam(t):
    chn_min = torch.tensor([0.0, 0.0, 0.0]).view(-1, 1, 1)
    chn_max = torch.tensor([255.0, 255.0, 255.0]).view(-1, 1, 1)
    return t.add_(-chn_min).div_(chn_max - chn_min)


def standardization_via_train_data_cam(t):
    chn_mean = torch.tensor([120.9292907715, 117.9040069580, 117.6787567139]).view(-1, 1, 1)
    chn_std = torch.tensor([60.7255439758, 62.0701942444, 66.4532699585]).view(-1, 1, 1)
    return t.add_(-chn_mean).div_(chn_std)


def rescaling_via_in_train_data_rad(t):  # depends if cube or rD
    chn_min = torch.tensor(-475.3615 if t.shape[-1] == 32 else -414.6096191406)
    chn_max = torch.tensor(-48.6883 if t.shape[-1] == 32 else -47.0893630981)
    return t.add_(-chn_min).div_(chn_max - chn_min)


def standardization_via_in_train_data_rad(t):  # depends if cube or rD
    chn_mean = torch.tensor(-256.5768 if t.shape[-1] == 32 else -230.5729370117)
    chn_std = torch.tensor(14.2264 if t.shape[-1] == 32 else 14.3406629562)
    return t.add_(-chn_mean).div_(chn_std)


# def rescaling_via_ex_train_data_rad(t):  # depends if cube or rD
    # chn_min = torch.tensor(-235.5952 if t.shape[-1] == 32 else -214.1508)
    # chn_max = torch.tensor(-22.1240 if t.shape[-1] == 32 else -19.0797)
    # return t.add_(-chn_min).div_(chn_max - chn_min)


# def standardization_via_ex_train_data_rad(t):  # depends if cube or rD
    # chn_mean = torch.tensor(-127.9469 if t.shape[-1] == 32 else -114.9936)
    # chn_std = torch.tensor(7.1288 if t.shape[-1] == 32 else 7.1543)
    # return t.add_(-chn_mean).div_(chn_std)


def requires_grad(m):
    "Check if the first parameter of `m` requires grad or not"
    ps = list(m.parameters())
    return ps[0].requires_grad if ps else False


def set_requires_grad(m, requires_grad):
    """Change if autograd should record operations on parameters in this module.
    This method sets the parameters’ requires_grad attributes in-place.
    This method is helpful for freezing part of the module for finetuning or training parts of a model individually (e.g., GAN training).
    """
    for param in m.parameters():
        param.requires_grad_(requires_grad)
    return m


def get_obj_size(obj):
    return f"Sample size in MB: {len(pickle.dumps(obj))/1024/1024:.3f}"


def seed_everything(seed):
    fix_rng_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.seed = seed
    torch.cuda.seed = seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #multi gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    #https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )


def calculate_conv_output_size(input, kernel_size, stride, padding):
    return (input - kernel_size + 2 * padding) / stride + 1


# from mayavi import mlab
def to_np(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    return x.detach().cpu().numpy()


# from tensorboardX.pytorch_graph import graph
def diagnose_cuda():
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
    print("Cached:", round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), "GB")


def visualize_image_batch_over_feature_maps(batch):
    im = batch.detach().cpu()[:, 0].unsqueeze(1)
    print("im shape", im.shape)
    grid = utils.make_grid(im, nrow=5, normalize=True, pad_value=1, padding=1)

    myobj = plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))

    for i, chn in enumerate(range(batch.shape[1])):
        print(i)
        im = batch.detach().cpu()[:, i].unsqueeze(1)
        grid = utils.make_grid(
            im,
            nrow=4,
            normalize=True,
            pad_value=1,
            padding=1,
        )
        myobj.set_data(grid.cpu().numpy().transpose((1, 2, 0)))
        plt.draw()
        plt.pause(0.5)


def visualize_radar_batch_over_feature_maps(batch):
    rd = batch.detach().cpu().numpy()
    print("rd shape", rd.shape)
    scale = rd[0]
    JNorm = rd[0, 5]
    src = mlab.pipeline.scalar_field(JNorm)
    print("ptp", rd.ptp())
    print("max", rd.max())
    print("min", rd.min())
    scale = JNorm
    # print(JNorm)
    s = mlab.pipeline.iso_surface(
        src,
        contours=[
            scale.max() - 0.5 * scale.ptp(),
            scale.min() + 0.5 * scale.ptp(),
        ],
        opacity=0.7,
    )

    @mlab.animate(delay=200, ui=False)
    def anim():
        # f = mlab.gcf()
        for i, chn in enumerate(range(batch.shape[1])):
            print(i)
            # f.scene.camera.azimuth(0.5)
            # f.scene.render()
            render = batch[5, chn].detach().cpu().numpy()
            print("render shape", render.shape)
            s.mlab_source.scalars = render
            yield

    mlab.axes(xlabel="Velocity", ylabel="range", zlabel="azimuth")
    a = anim()  # Starts the animation without a UI.
    mlab.show()


def visualize_rD_feature_maps(batch):
    print(batch.shape)
    rd = batch[:, 0, ..., batch.shape[-1] // 2].unsqueeze(1)
    rd = rd.detach().cpu()
    grid = utils.make_grid(rd, nrow=5, normalize=True, pad_value=1, padding=1)
    # print(grid.shape)
    myobj = plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))

    for i, chn in enumerate(range(batch.shape[1])):
        print(i)
        # print(batch.shape)
        rd = batch[:, i, ..., batch.shape[-1] // 2].unsqueeze(1)
        # print(rd.shape)
        rd = rd.detach().cpu()
        grid = utils.make_grid(
            rd,
            nrow=4,
            normalize=True,
            pad_value=1,
            padding=1,
        )
        myobj.set_data(grid.cpu().numpy().transpose((1, 2, 0)))
        plt.draw()
        plt.pause(0.5)


def visualize_rA_feature_maps(batch):
    print(batch.shape)
    rd = batch[:, 0, batch.shape[-3] // 2].unsqueeze(1)
    rd = rd.detach().cpu()
    grid = utils.make_grid(rd, nrow=5, normalize=True, pad_value=1, padding=1)
    # print(grid.shape)
    myobj = plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))

    for i, chn in enumerate(range(batch.shape[1])):
        print(i)
        # print(batch.shape)
        rd = batch[:, i, batch.shape[-3] // 2].unsqueeze(1)
        # print(rd.shape)
        rd = rd.detach().cpu()
        grid = utils.make_grid(
            rd,
            nrow=4,
            normalize=True,
            pad_value=1,
            padding=1,
        )
        myobj.set_data(grid.cpu().numpy().transpose((1, 2, 0)))
        plt.draw()
        plt.pause(0.5)


def visualize_radar_feature_vector(batch):
    print("teset")
    print("batch", batch.shape)
    rd = batch.squeeze(4)
    print("rd", rd.shape)
    rd = rd.detach().cpu()
    rd = torch.transpose(rd, 1, 2)[:, :, :200]
    print("rd", rd.shape)
    grid = utils.make_grid(rd, nrow=12, normalize=True, padding=0)
    print(grid.shape)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.draw()
    plt.pause(0.5)


def _nd_window(data, filter_function):
    """
    Performs an in-place windowing on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.

    Parameters
    ----------
    data : ndarray
           Input data to be windowed, modified in place.
    filter_function : 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.hamming
    """

    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [1] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size).reshape(filter_shape)
        # scale the window intensities to maintain image intensity
        np.power(window, (1.0 / data.ndim), out=window)
        data *= window

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and 'Layer' not in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def weights_init_positive(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and 'Layer' not in classname:
        nn.init.constant_(m.weight.data, 10)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant(m.weight.data, 10)
        nn.init.constant_(m.bias.data, 10)


"""
def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if "Conv" in classname:
        m.weight.data.normal_(0.0, 0.02)

    elif "BatchNorm" in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
"""


"""
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        xavier(m.bias.data)
"""


def check_manual_seed(seed):
    """If manual seed is not specified, choose a random one and communicate it to the user."""
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using manual seed: {seed}".format(seed=seed))


def random_seeding(seed_value):
    # for workers > 0, you need to set the seed in the worker_init_fn
    print("SEED:", seed_value)
    if seed_value is not None:
        # torch.initial_seed(seed_value)

        torch.manual_seed(
            seed_value
        )  # seed the RNG for all devices (both CPU and CUDA)
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # if cuda:
        # print('cuda stuff')
        torch.cuda.manual_seed_all(seed_value)

        # Pytorch is not reproducible between CPU and GPU, between different platforms, and different commits.
        # Also there are functions like interpolate that introduce non-deterministic behaviour

        # https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054/2
        torch.backends.cudnn.deterministic = True
        # allow the cuda backend to optimize your graph during its first execution.
        # However, be aware that if you change the network input/output tensor size the graph will be optimized
        # each time a change occurs. This can lead to very slow runtime and out of memory errors.
        # Only set this flag if your input and output have always the same shape.
        # Usually, this results in an improvement of about 20%.
        # cuDNN will use some heuristics at the beginning of your training to figure out which algorithm will be most performant for your model architecture and input.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        #     warnings.warn('You have chosen to seed training. '
        #                   'This will turn on the CUDNN deterministic setting, '
        #                   'which can slow down your training considerably! '
        #                   'You may see unexpected behavior when restarting '
        #                   'from checkpoints.')


# Clear cuda cache between training/testing
def empty_cuda_cache():
    torch.cuda.empty_cache()
    import gc

    gc.collect()


def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules:
            print(f"{n}: {num_params}")
        total_num_params += num_params
    print("-" * 50)
    print(f"Total number of parameters: {total_num_params:.5e}")


def render_probabilities(imgs, cor, lab, **kw):
    for i in range(imgs.shape[0]):
        img = tf.to_pil_image(imgs[i])
        # img = img.resize((3 * 640, 3 * 480), resample=Image.LANCZOS)
        img = img.resize((3 * imgs.shape[3], 3 * imgs.shape[2]), resample=Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("DejaVuSansMono.ttf", 40)
        assert cor[i] <= 1, "prediction is larger than 1!!!!!"
        draw.text((3, 3), str(cor[i].item())[:5], (0, 0, 0), font=font)
        draw.text((3, 50), str(lab[i].item())[:1], (0, 0, 0), font=font)
        img = img.resize((imgs.shape[3], imgs.shape[2]), resample=Image.LANCZOS)
        imgs[i] = tf.to_tensor(img)
    return imgs


def superimpose_attention(atts_upsc, cam, **kw):
    # https://github.com/kyuyeonpooh/objects-that-sound/blob/master/utils/heatmap.py
    # result_color = torch.empty(cam.shape[0], 3, cam.shape[2], cam.shape[3])
    # result_red = torch.empty(cam.shape[0], 3, cam.shape[2], cam.shape[3])
    result_jet = torch.empty(cam.shape[0], 3, cam.shape[2], cam.shape[3])
    # result = torch.empty(cam.shape[0], 3, cam.shape[2], cam.shape[3])

    # atts_upsc[:, :, 30:40, 30:40] = 0.99
    # atts_upsc[:, :, 50:60, 50:60] = 0.7
    # atts_upsc[:, :, 100:110, 100:110] = 0.1

    # atts_upsc = np.log10(atts_upsc + 1.0)
    # print(atts_upsc.min())
    # print(atts_upsc.max())
    # print(atts_upsc.shape)
    # exit()
    # atts_cpu_red = atts_upsc.detach().cpu()
    atts_cpu = to_np(atts_upsc)
    # cam_red = (cam - torch.min(cam)) / torch.max(cam)
    # cam_red_cpu = cam.cpu()
    cam_color = (cam - torch.min(cam)) / torch.max(cam)
    cam_color_cpu = cam_color.cpu()

    for i in range(cam_color_cpu.shape[0]):
        # att_chn_red = torch.cat( # first channel is red
        # (atts_cpu_red[i], torch.zeros(2, cam.shape[2], cam.shape[3]))
        # )
        # att_pil_red = tf.to_pil_image(att_chn_red)
        # img_pil_red = tf.to_pil_image(cam_red_cpu[i])
        # img_att_red = Image.blend(img_pil_red, att_pil_red, 0.5)
        # img_att_ten_red = tf.to_tensor(img_att_red)
        # result_red[i] = img_att_ten_red

        colorize = plt.get_cmap("jet")
        heatmap = colorize(atts_cpu[i], bytes=True)
        att_pil = Image.fromarray(heatmap[0, :, :, :3], mode="RGB")
        img_pil = tf.to_pil_image(cam_color_cpu[i])
        result_jet[i] = tf.to_tensor(Image.blend(img_pil, att_pil, 0.5))

        # hm = magnitude2heatmap(atts_cpu[i, 0], log=False)
        # att_chn = torch.from_numpy(np.ascontiguousarray(hm))
        # att_chn = (att_chn - torch.min(att_chn)) / torch.max(att_chn)
        # att_chn = att_chn.permute(2, 0, 1)
        # att_pil = tf.to_pil_image(att_chn)
        # img_pil = tf.to_pil_image(cam_color_cpu[i])
        # img_att = Image.blend(img_pil, att_pil, 0.5)
        # img_att_ten = tf.to_tensor(img_att)
        # result_color[i] = img_att_ten
    return result_jet


def color_confusion(cam, prd, lab, **kw):
    """
    This function colors a left upper square of the camera image
    according to the binary classification result
    """
    h_rect = 35
    w_rect = 42
    prd = prd.bool()
    lab = lab.bool()
    # true = prd.nonzero().squeeze()
    true = torch.nonzero(prd).squeeze()
    false = torch.nonzero(~prd).squeeze()

    tp = torch.zeros_like(prd)
    tn = torch.zeros_like(prd)
    fn = torch.zeros_like(prd)
    fp = torch.zeros_like(prd)

    tp[true[prd[true] == lab[true]]] = 1
    fp[true[prd[true] != lab[true]]] = 1
    tn[false[prd[false] == lab[false]]] = 1
    fn[false[prd[false] != lab[false]]] = 1

    # green
    cam[tp, 0, :h_rect, :w_rect] = 0.0
    cam[tp, 1, :h_rect, :w_rect] = 1.0
    cam[tp, 2, :h_rect, :w_rect] = 0.0

    # blue
    cam[tn, 0, :h_rect, :w_rect] = 0.0
    cam[tn, 1, :h_rect, :w_rect] = 0.0
    cam[tn, 2, :h_rect, :w_rect] = 1.0

    # yellow
    cam[fp, 0, :h_rect, :w_rect] = 244 / 255.0
    cam[fp, 1, :h_rect, :w_rect] = 232 / 255.0
    cam[fp, 2, :h_rect, :w_rect] = 144 / 255.0

    # red
    cam[fn, 0, :h_rect, :w_rect] = 1.0
    cam[fn, 1, :h_rect, :w_rect] = 0.0
    cam[fn, 2, :h_rect, :w_rect] = 0.0

    return cam, fp, tp, tn, fn


def align_radar_cube_for_display(rad):
    mlab.options.offscreen = True

    rad_cpu_np = rad[:, 0].cpu().numpy()

    iso_view = torch.empty(rad.shape[0], 480, 640, 3)
    rD_proj = torch.empty(rad.shape[0], 480, 640, 3)
    rA_proj = torch.empty(rad.shape[0], 480, 640, 3)

    mayavi.mlab.figure(size=(640, 480))

    src = mlab.pipeline.scalar_field(rad_cpu_np[0])
    scene = mlab.pipeline.iso_surface(
        src,
        contours=[
            rad_cpu_np[0].max() - 0.15 * rad_cpu_np[0].ptp(),
            rad_cpu_np[0].min() + 0.15 * rad_cpu_np[0].ptp(),
        ],
        opacity=0.75,
    )

    def_view = mlab.view()

    #  loop over sample in batch and produce views of the cube
    for i in range(rad.shape[0]):

        scene.mlab_source.scalars = rad_cpu_np[i]
        s = scene.mlab_source
        s.trait_set(
            contours=[
                rad_cpu_np[i].max() - 0.15 * rad_cpu_np[i].ptp(),
                rad_cpu_np[i].min() + 0.15 * rad_cpu_np[i].ptp(),
            ]
        )

        mlab.view(def_view[0], def_view[1])

        #  iso view
        mlab.gcf().scene.parallel_projection = True
        mlab.axes(
            ranges=[-2, 2, 0.5, 10, -90, 90],
            xlabel="velocity",
            ylabel="range",
            zlabel="azimuth",
            nb_labels=3,
            x_axis_visibility=True,
            y_axis_visibility=True,
            z_axis_visibility=True,
        )

        iso_view_array = mlab.screenshot(scene, antialiased=True)
        iso_view[i] = torch.from_numpy(np.ascontiguousarray(iso_view_array))

        #  rD view
        mlab.axes(
            ranges=[-2, 2, 0.5, 10, -90, 90],
            xlabel="velocity",
            ylabel="range",
            zlabel="",
            # z_axis_visibility=False,
            nb_labels=3,
        )
        mlab.gcf().scene.parallel_projection = True
        mlab.view(azimuth=0, elevation=0)

        rD_proj_array = mlab.screenshot(scene, antialiased=True)
        rD_proj[i] = torch.from_numpy(np.ascontiguousarray(rD_proj_array))

        #  rA view
        mlab.axes(
            ranges=[-2, 2, 0.5, 10, -90, 90],
            xlabel="",
            ylabel="range",
            zlabel="azimuth",
            nb_labels=3,
            # x_axis_visibility=False,
            # z_axis_visibility=True,
        )

        mlab.gcf().scene.parallel_projection = True
        mlab.view(azimuth=360, elevation=270)

        rA_proj_array = mlab.screenshot(scene, antialiased=True)
        rA_proj[i] = torch.from_numpy(np.ascontiguousarray(rA_proj_array))
        # rA_proj = torch.from_numpy(np.ascontiguousarray(rA_proj_array))

    iso_view = iso_view.permute(0, 3, 1, 2)
    rD_proj = rD_proj.permute(0, 3, 1, 2)
    rA_proj = rA_proj.permute(0, 3, 1, 2)

    return iso_view, rD_proj, rA_proj


def align_radar_2D_views_for_display(rad):

    rD_sum = torch.sum(rad, dim=4).cpu()
    rA_sum = torch.sum(rad, dim=2).cpu()

    rD_slice = rad[..., rad.shape[-1] // 2].cpu()
    rA_slice = rad[:, :, rad.shape[2] // 2].cpu()

    # print("rD shaped", rD_sum.shape)
    # print("ra shaped", rA_sum.shape)

    # for i in range(rad.shape[0]):
    #     # print("redux", rD_sum[i].shape)
    #     img = tf.to_pil_image(rD_sum[i])
    #     # img = img.resize((640, 480), resample=Image.LANCZOS)
    #     a = tf.to_tensor(img)
    #     # print(a.shape)
    #     rD_view[i] = a

    # return rD_sum.permute(0, 1, 3, 2), rA_sum
    return rD_slice.permute(0, 1, 3, 2), rA_slice


def prepare_attentive_cam_view(cam, loc, cor, lab, **kw):

    attention_upscaled = F.interpolate(
        loc, size=(cam.shape[2], cam.shape[3]), mode="bilinear", align_corners=True
    )

    # attentive_images, red, jet = superimpose_attention(attention_upscaled, cam, **kw)
    attentive_images = superimpose_attention(attention_upscaled, cam, **kw)

    lab = lab.cpu()
    confused_images, fp, tp, tn, fn = color_confusion(
        attentive_images, torch.round(cor), lab
    )

    annotated_images = render_probabilities(confused_images, cor, lab)

    return annotated_images, fp, tp, tn, fn


# return annotated_images, fp


def prepare_rad_cube_view(state):
    return align_radar_2D_views_for_display(state.rad)


# return align_radar_cube_for_display(state.rad)


# def superimpose_heatmap_cv2(images, interpolate, batch_size):
#     result = torch.empty(batch_size, 3, 480, 640)
#     for image in range(images.shape[0]):
#         localization_upscaled = (
#             interpolate[image].cpu().permute(1, 2, 0).numpy() * 255
#         ).astype("uint8")
#         img = images[image].cpu().permute(1, 2, 0).numpy().astype("uint8")
#         # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         heatmap = cv2.applyColorMap(localization_upscaled, cv2.COLORMAP_HOT)
#         img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
#         res = TF.to_tensor(img)
#         result[image] = res
#     plt.figure()
#     grid = utils.make_grid(result, nrow=5)
#     plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
#     plt.axis("off")
#     plt.ioff()
#     plt.show()

# FID
# https://github.com/larry-11/Self-Supervised-GAN/blob/master/utils/fid_score.py
# https://github.com/larry-11/Self-Supervised-GAN/blob/master/utils/cal_fid.py


def generate_latent_walk(self, number):
    if not os.path.exists("interpolated_images/"):
        os.makedirs("interpolated_images/")

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

            z_intp = Variable(z_intp)
            images = []
            alpha = 1.0 / float(number_int + 1)
            print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1 * alpha + z2 * (1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5)  # denormalize
            images.append(fake_im.view(self.C, 32, 32).data.cpu())

            grid = utils.make_grid(images, nrow=number_int)
            utils.save_image(
                grid,
                "interpolated_images/interpolated_{}.png".format(str(number).zfill(3)),
            )
            print("Saved interpolated images.")


def magnitude2heatmap(mag, log=True, scale=255.0):
    if log:
        mag = np.log10(mag + 1.0)
        mag *= scale
        mag[mag > 255] = 255
        mag = mag.astype(np.uint8)
        mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
        mag_color = mag_color[:, :, ::-1]
    return mag_color


class SimpleDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
            # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in (
                    "update",
                    "pop",
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
            super().__setattr__(name, value)
            super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super().pop(k, d)
