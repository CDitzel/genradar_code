import os
import glob
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from os.path import abspath
import logging
import shutil
import random
import argparse
import time
import sys
import math
from datetime import datetime
from collections import OrderedDict
from functools import partial
import pprint
import json
from pickle import dumps

from skimage import exposure
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import trange, tqdm
from tqdm.contrib import tenumerate
from ruamel.yaml import YAML
import numpy as np
import torch
from torch import optim, nn
from torch.optim import lr_scheduler as lr
from torchvision import transforms as T
from torch.nn import functional as F
import torchvision.transforms.functional as tf
from torchvision import utils as U
from torch.utils.data import DataLoader
from torch import linalg as LA

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib as mpl


torch.set_printoptions(profile="full", linewidth=500, precision=10)
pp = pprint.PrettyPrinter(indent=4)
plt.switch_backend("agg")
import seaborn as sns
import PIL
from einops import rearrange, reduce, repeat
# """
plt.rcParams.update(
    {"text.usetex": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]}
)
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Helvetica"],
    }
)
rc_fonts = {
    "text.usetex": True,
    'text.latex.preview': True,
    "font.size": 4,
    'axes.titlesize': 4,
    "axes.labelsize": 4,
    "legend.fontsize": 4,
    "xtick.labelsize": 4,
    "ytick.labelsize": 4,
    'figure.titlesize': 4,
    'mathtext.default': 'regular',
    'text.latex.preamble': [r"""\usepackage{bm}"""],
}
mpl.rcParams.update(rc_fonts)
# """

import lib
from lib import (
    InrasData,
    setup_logging,
    load_config,
    get_Host_name_IP,
    seed_everything,
    AvgMeter,
    save_model,
    TensorboardLogger,
    disable_feature_for_certain_model_paramters,
    weight,
    set_requires_grad,
    weights_init,
    requires_grad,
    layout,
    LogitLaplaceLoss,
    NormalLoss,
    print_num_params,
    to_np,
    EncoderDalle,
    DecoderDalle,
    zero_grads,
    write_lmdb,
    put_or_grow,
    open_lmdb,
    InceptionV3,
    calculate_frechet_distance
)

# torch.backends.cudnn.enabled = False
torch.set_printoptions(profile="full", linewidth=200, precision=20)
parser = argparse.ArgumentParser(description="Radar Camera Fusion")
parser.add_argument("-c", default="config.yaml")
parser.add_argument("-d", default="__oO__", help="description")
parser.add_argument("-resume", default=None, help="resume from checkpoint")
parser.add_argument("-dry", action="store_true", help="dry run")
args = parser.parse_args()

cfg, cfg_str, cfg_dict = load_config(args.c)

def main():
    if not args.dry:
        if args.resume is None:
            host, _ = get_Host_name_IP()
            tStart = str(datetime.now())[:16].replace(" ", "_")
            desc = f"{host}_{tStart}_{args.d}"
            res_dir = f"{cfg.exp.dst_dir}/{cfg.exp.name}/{desc}/"
        else:
            ckpt_path = args.resume
            state = torch.load(ckpt_path)
            model_state = state["model_state_dict"]
            missing, unexpected = model.load_state_dict(model_state, strict=True)
            res_dir, stem = os.path.split(ckpt_path)
            res_dir += "/"
        os.makedirs(res_dir, exist_ok=True)
        setup_logging(res_dir + "run.log", cfg.exp.log_level)
        # model_file = abspath(sys.modules[model.__class__.__module__].__file__)
        [shutil.copy(f, res_dir) for f in (__file__, args.c)]
        tb = TensorboardLogger(res_dir)
        tb.text(cfg_str)
        if cfg.exp.seed is not None:
            seed_everything(cfg.exp.seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    torch.autograd.set_detect_anomaly(cfg.exp.debug)
    torch.autograd.profiler.profile(cfg.exp.debug)
    torch.autograd.profiler.emit_nvtx(cfg.exp.debug)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.model.args['load_path'] is None:
        model = eval(f"lib.{cfg.model.name}")(**cfg.model.args).to(dev).eval()
        param_groups_gen = disable_feature_for_certain_model_paramters(
            model, "weight_decay"
        )
        optim_com = eval(f"optim.{cfg.optim_com.name}")(
            param_groups_gen, **cfg.optim_com.args
        )
        lrs_com = eval(f"lr.{cfg.lrs_com.name}")(optim_com, **cfg.lrs_com.args)

        loss = eval(f"lib.{cfg.loss.name}")(**cfg.loss.args)

    data = {
        "train": InrasData.loader(**cfg.train.args),
        "valid": InrasData.loader(**cfg.valid.args)
    }

    if cfg.model.args['load_path'] is not None:
        model_list = cfg.model.args['load_path']
        del data['train']
        embeddings = dict()
        variance_ratio = []
        dims = 2048
        fig = plt.figure(figsize=(2.5, 9.0))
        gs = fig.add_gridspec(4, hspace=0.5)
        axs = gs.subplots()

        width = 0.004
        # width = 1
        alpha = 0.5
        # alpha = 1
        labels = [64, 256, 1024]
        nb_bins = 256
        nb_bins_rad = 256
        count_r = np.zeros((3, nb_bins))
        count_g = np.zeros((3, nb_bins))
        count_b = np.zeros((3, nb_bins))
        cum_r = np.zeros((3, nb_bins))
        cum_g = np.zeros((3, nb_bins))
        cum_b = np.zeros((3, nb_bins))
        count_rad = np.zeros((3, nb_bins_rad))
        cum_rad = np.zeros((3, nb_bins_rad))

        orig_count_r = np.zeros((nb_bins))
        orig_count_g = np.zeros((nb_bins))
        orig_count_b = np.zeros((nb_bins))
        orig_cum_r = np.zeros((nb_bins))
        orig_cum_g = np.zeros((nb_bins))
        orig_cum_b = np.zeros((nb_bins))
        orig_count_rad = np.zeros((nb_bins_rad))
        orig_cum_rad = np.zeros((nb_bins_rad))

        # if 'cat_util' in cfg.exp.eval_mode:
            # fig = plt.figure(figsize=(2.5, 9.0))
            # ax = fig.add_subplot(2,2,1)
            # fig, ax = plt.subplots()
            # ax.set_xticks([])
            # ax.set_yticks([])


        for model_idx, saved_model in enumerate(model_list):
            print(f'Performing inference on {saved_model}')
            # print(cfg.model.args['load_path'])
            cfg.model.args['load_path'] = saved_model
            # print(cfg.model.args['load_path'])
            # exit()
            model = eval(f"lib.{cfg.model.name}")(**cfg.model.args).to(dev).eval()

            """Batch size of 1 for video rendering with png output and lmdb serialization"""
            # state = torch.load(cfg.model.args['load_path'])["config"]
            state = torch.load(saved_model)["config"]
            del state['meta']
            # print(json.dumps(state, indent=1))
            # exit()

            path, stem = os.path.split(saved_model)
            fp = path + '/' + 'eval'
            fp_perplx = path + '/' + 'eval/perplx'
            os.makedirs(fp, exist_ok=True)
            os.makedirs(fp_perplx, exist_ok=True)
            nTok = model.discretizer.nTokens
            nSamples = 0
            nBatches = 0
            max_dist_orig = 0
            max_dist_infer = 0

            cat_util = torch.zeros([256, nTok]).to(dev)
            pmf = torch.zeros([256, nTok]).to(dev)
            # pmf = torch.zeros([nTok, 16, 16]).to(dev)
            min_max_masses = torch.zeros([256, 2])

            print(f'K={nTok}')
            print(f'D={model.discretizer.embedding.weight.shape[1]}')
            print(f'depth={model.discretizer.pre_proj}')
            # exit()
            if 'pca' in cfg.exp.eval_mode:
                # ONLY enable 256depth models of K: 64, 256, 1024
                """
                emb_256 = model.discretizer.embedding.weight

                model_128_path = path.replace('d256','d128') + '/model.ckpt'
                cfg.model.args['load_path'] = model_128_path
                model = eval(f"lib.{cfg.model.name}")(**cfg.model.args).to(dev).eval()
                state = torch.load(model_128_path)["config"]
                # del state['meta']
                # print(json.dumps(state, indent=1))
                emb_128 = model.discretizer.embedding.weight

                model_64_path = path.replace('d256','d64') + '/model.ckpt'
                cfg.model.args['load_path'] = model_64_path
                model = eval(f"lib.{cfg.model.name}")(**cfg.model.args).to(dev).eval()
                state = torch.load(model_64_path)["config"]
                # del state['meta']
                # print(json.dumps(state, indent=1))
                emb_64 = model.discretizer.embedding.weight

                emb = torch.stack([emb_64, emb_128, emb_256])
                emb = to_np(emb)
                # print(emb[:, :2, :2])
                # exit()
                """
                num_pca = 16
                emb = model.discretizer.embedding.weight
                factor = 4
                # if emb.shape[1] == 256:
                    # emb = emb[:, 0::4]
                    # emb = emb[:, :64]

                # if emb.shape[1] == 1024:
                    # emb = emb[:, :64]
                    # emb = emb[:, 0::16]

                emb = to_np(rearrange(emb, "n (h d) -> h n d", h=factor))
                pcas = [PCA(svd_solver='full', n_components=num_pca).fit(emb[n]) for n in range(len(emb))]
                # pcas = [PCA(n_components=num_pca).fit(emb[n, ...]) for n in range(len(emb))]
                embeddings[saved_model] = {"emb": emb, "pcas": pcas}

                variance_ratio.extend([[saved_model, *pca.explained_variance_ratio_] for pca in embeddings[saved_model]["pcas"]])
                continue

            if 'norms' in cfg.exp.eval_mode:
                mse_concrete, mse_warm_gumbel, mse_argmax, mse_categorical, mse_cold_gumbel = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter()
                mse_concrete_runs, mse_warm_gumbel_runs, mse_argmax_runs, mse_categorical_runs,  mse_cold_gumbel_runs =  [], [], [], [], []
                fro_concrete, fro_warm_gumbel, fro_argmax, fro_categorical, fro_cold_gumbel= AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
                fro_concrete_runs, fro_warm_gumbel_runs, fro_argmax_runs, fro_categorical_runs, fro_cold_gumbel_runs= [], [], [], [], []
                psnr_concrete, psnr_warm_gumbel, psnr_argmax, psnr_categorical, psnr_cold_gumbel= AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
                psnr_concrete_runs, psnr_warm_gumbel_runs, psnr_argmax_runs, psnr_categorical_runs, psnr_cold_gumbel_runs= [], [], [], [], []

            if 'fid' in cfg.exp.eval_mode:
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
                inception = InceptionV3([block_idx], resize_input=False,
                                        normalize_input=True, use_fid_inception=False).to(dev).eval()

                fid_warm_gumbel_runs, fid_categorical_runs, fid_argmax_runs, fid_cold_gumbel_runs = [], [], [], []

            for epo in range(cfg.exp.eval_times):
                data_arr = torch.empty((data['valid'].size, dims))
                warm_gumbel_arr = torch.empty((data['valid'].size, dims))
                categorical_arr = torch.empty((data['valid'].size, dims))
                argmax_arr = torch.empty((data['valid'].size, dims))
                cold_gumbel_arr = torch.empty((data['valid'].size, dims))
                start_idx = 0

                for i, batch in enumerate(t := tqdm(data['valid'], leave=True)):
                    mod = batch[cfg.exp.mod].to(dev)
                    # print(mod.min())
                    # print(mod.max())
                    with torch.no_grad():
                        # _, rec, _, _ = model(mod, cfg.exp.soft_idx)
                        _, rec, _, _, _ = model(mod, cfg.exp.soft_idx)
                        (warm_gumbel_t, argmax_t, categorical_t, cold_gumbel_t) = model.discretize(mod, cfg.exp.soft_idx)

                        # argmax_t = rearrange(argmax_t, "b (h w)-> b (w h)", h=16)

                        if 'cat_util' in cfg.exp.eval_mode:
                            for i in range(argmax_t.shape[0]):
                                cat_util += F.one_hot(argmax_t[i], nTok)
                                pmf += rearrange(z[i], 'k h w -> (h w) k').softmax(dim= -1)
                                max_masses = torch.amax(z[i].softmax(0), dim=0).flatten()
                                min_masses = torch.amin(z[i].softmax(0), dim=0).flatten()
                                for c in range(len(max_masses)):
                                    if max_masses[c] > min_max_masses[c, 0]:
                                        min_max_masses[c, 0] = max_masses[c]
                                    if min_masses[c] > min_max_masses[c, 1]:
                                        min_max_masses[c, 1] = min_masses[c]

                        # rad_toks = torch.ones(256, 256).to(rad_toks)
                        # cam_toks = torch.ones(256, 256).to(cam_toks)
                        # for t in range(256):
                            # rad_toks[t] = t * torch.ones(256).to(rad_toks)
                            # cam_toks[t] = t * torch.ones(256).to(cam_toks)

                        categorical_bins = torch.bincount(categorical_t.view(-1), minlength=nTok).cpu()
                        warm_gumbel_t_rec, argmax_t_rec, categorical_t_rec, cold_gumbel_t_rec = (model.constitute(t) for t
                                                              in (warm_gumbel_t, argmax_t, categorical_t, cold_gumbel_t))

                        last_batch = layout(mod, argmax_t_rec,
                                            categorical_t_rec, warm_gumbel_t_rec,
                                            only_last=cfg.exp.show, mod=cfg.exp.mod)
                        dist_orig = np.abs(to_np(mod.max()) - to_np(mod.min()))
                        if dist_orig > max_dist_orig:
                            max_dist_orig = dist_orig

                        dist_infer = np.abs(to_np(categorical_t_rec.max()) - to_np(categorical_t_rec.min()))
                        if dist_infer > max_dist_infer:
                            max_dist_infer = dist_infer

                    if 'hist' in cfg.exp.eval_mode:
                        nSamples += categorical_t_rec.shape[0]
                        nBatches += 1
                        categorical_t_for_hist = to_np(categorical_t_rec)
                        mod_for_hist = to_np(mod)
                        # print(categorical_t_for_hist[0, 0, :10, :10])
                        if cfg.exp.mod == 'cam':
                            hist_r = np.histogram(categorical_t_for_hist[:, 0], bins=nb_bins)
                            hist_g = np.histogram(categorical_t_for_hist[:, 1], bins=nb_bins)
                            hist_b = np.histogram(categorical_t_for_hist[:, 2], bins=nb_bins)
                            count_r[model_idx] += hist_r[0]
                            count_g[model_idx] += hist_g[0]
                            count_b[model_idx] += hist_b[0]
                            img_cdf_r, bins = exposure.cumulative_distribution(categorical_t_for_hist[:, 0], nb_bins)
                            img_cdf_g, bins = exposure.cumulative_distribution(categorical_t_for_hist[:, 1], nb_bins)
                            img_cdf_b, bins = exposure.cumulative_distribution(categorical_t_for_hist[:, 2], nb_bins)
                            cum_r[model_idx] += img_cdf_r
                            cum_g[model_idx] += img_cdf_g
                            cum_b[model_idx] += img_cdf_b

                            if model_idx == 1:
                                orig_hist_r = np.histogram(mod_for_hist[:, 0], bins=nb_bins)
                                orig_hist_g = np.histogram(mod_for_hist[:, 1], bins=nb_bins)
                                orig_hist_b = np.histogram(mod_for_hist[:, 2], bins=nb_bins)
                                orig_count_r += orig_hist_r[0]
                                orig_count_g += orig_hist_g[0]
                                orig_count_b += orig_hist_b[0]
                                orig_img_cdf_r, bins = exposure.cumulative_distribution(mod_for_hist[:, 0], nb_bins)
                                orig_img_cdf_g, bins = exposure.cumulative_distribution(mod_for_hist[:, 1], nb_bins)
                                orig_img_cdf_b, bins = exposure.cumulative_distribution(mod_for_hist[:, 2], nb_bins)
                                orig_cum_r += orig_img_cdf_r
                                orig_cum_g += orig_img_cdf_g
                                orig_cum_b += orig_img_cdf_b
                        if cfg.exp.mod == 'rad':
                            hist_rad = np.histogram(categorical_t_for_hist, bins=nb_bins_rad)
                            count_rad[model_idx] += hist_rad[0]
                            img_cdf_rad, bins = exposure.cumulative_distribution(categorical_t_for_hist, nb_bins)
                            cum_rad[model_idx] += img_cdf_rad

                            if model_idx == 1:
                                orig_hist_rad = np.histogram(mod_for_hist, bins=nb_bins_rad)
                                orig_count_rad += orig_hist_rad[0]
                                orig_img_cdf_rad, bins = exposure.cumulative_distribution(mod_for_hist, nb_bins)
                                orig_cum_rad += orig_img_cdf_rad


                    if 'fid' in cfg.exp.eval_mode:
                        if cfg.exp.mod == 'cam':
                            # mod /= 0.5
                            mod_fid = mod
                            warm_gumbel_t_rec_fid = warm_gumbel_t_rec
                            categorical_t_rec_fid = categorical_t_rec
                            argmax_t_rec_fid = argmax_t_rec
                            cold_gumbel_t_rec_fid = cold_gumbel_t_rec
                            # print(mod_fid.min())
                            # print(mod_fid.max())
                            # print(categorical_t_rec_fid.min())
                            # print(categorical_t_rec_fid.max())
                            data_arr[start_idx:start_idx + mod.shape[0]] = inception(mod_fid)[0].squeeze(3).squeeze(2)
                            warm_gumbel_arr[start_idx:start_idx + warm_gumbel_t_rec.shape[0]] = inception(warm_gumbel_t_rec_fid)[0].squeeze(3).squeeze(2)
                            categorical_arr[start_idx:start_idx + categorical_t_rec.shape[0]] = inception(categorical_t_rec_fid)[0].squeeze(3).squeeze(2)
                            argmax_arr[start_idx:start_idx + argmax_t_rec.shape[0]] = inception(argmax_t_rec_fid)[0].squeeze(3).squeeze(2)
                            cold_gumbel_arr[start_idx:start_idx + cold_gumbel_t_rec.shape[0]] = inception(cold_gumbel_t_rec_fid)[0].squeeze(3).squeeze(2)
                        if cfg.exp.mod == 'rad':
                            mod_fid = repeat(mod, 'b c h w -> b (c copy) h w', copy=3)
                            warm_gumbel_t_rec_fid = repeat(warm_gumbel_t_rec, 'b c h w -> b (c copy) h w', copy=3)
                            categorical_t_rec_fid = repeat(categorical_t_rec, 'b c h w -> b (c copy) h w', copy=3)
                            argmax_t_rec_fid = repeat(argmax_t_rec, 'b c h w -> b (c copy) h w', copy=3)
                            cold_gumbel_t_rec_fid = repeat(argmax_t_rec, 'b c h w -> b (c copy) h w', copy=3)
                            data_arr[start_idx:start_idx + mod.shape[0]] = inception(mod_fid)[0].squeeze(3).squeeze(2)
                            warm_gumbel_arr[start_idx:start_idx + warm_gumbel_t_rec.shape[0]] = inception(warm_gumbel_t_rec_fid)[0].squeeze(3).squeeze(2)
                            categorical_arr[start_idx:start_idx + categorical_t_rec.shape[0]] = inception(categorical_t_rec_fid)[0].squeeze(3).squeeze(2)
                            argmax_arr[start_idx:start_idx + argmax_t_rec.shape[0]] = inception(argmax_t_rec_fid)[0].squeeze(3).squeeze(2)
                            cold_gumbel_arr[start_idx:start_idx + argmax_t_rec.shape[0]] = inception(cold_gumbel_t_rec_fid)[0].squeeze(3).squeeze(2)

                        start_idx = start_idx + mod.shape[0]

                    if 'plx' in cfg.exp.eval_mode:
                        plt.subplots(figsize=(2.56, 2.56))
                        # plt.plot(torch.arange(nTok), categorical_bins / categorical_bins.max(), color='b')
                        plt.bar(torch.arange(nTok), categorical_bins / categorical_bins.max(), color='b', width=10)
                        # plt.axis('off')
                        # plt.tick_params(axis='both', left='off', top='off', right='off',
                                    # bottom='off', labelleft='off', labeltop='off', labelright='off',
                                    # labelbottom='off')
                        plt.savefig(f'{fp_perplx}/{str(i).zfill(5)}.png', dpi=100 , pad_inches=0.0)
                        # bbox_inches="tight"
                        plt.close()
                        # continue

                    if 'norms' in cfg.exp.eval_mode:
                        diff_concrete = mod - rec
                        diff_warm_gumbel = mod - warm_gumbel_t_rec
                        diff_categorical = mod - categorical_t_rec
                        diff_argmax = mod - argmax_t_rec
                        diff_cold_gumbel = mod - cold_gumbel_t_rec

                        if cfg.exp.mod == 'rad':
                            fro_concrete.update(LA.norm(diff_concrete[0, 0], ord='fro').item())
                            fro_warm_gumbel.update(LA.norm(diff_warm_gumbel[0, 0], ord='fro').item())
                            fro_argmax.update(LA.norm(diff_argmax[0, 0], ord='fro').item())
                            fro_categorical.update(LA.norm(diff_categorical[0, 0], ord='fro').item())
                            fro_cold_gumbel.update(LA.norm(diff_cold_gumbel[0, 0], ord='fro').item())
                        else:
                            r_concrete = LA.norm(diff_concrete[0, 0], ord='fro').item()
                            g_concrete = LA.norm(diff_concrete[0, 1], ord='fro').item()
                            b_concrete = LA.norm(diff_concrete[0, 2], ord='fro').item()
                            fro_concrete.update((r_concrete + g_concrete + b_concrete) / 3.0)

                            r_warm_gumbel = LA.norm(diff_warm_gumbel[0, 0], ord='fro').item()
                            g_warm_gumbel = LA.norm(diff_warm_gumbel[0, 1], ord='fro').item()
                            b_warm_gumbel = LA.norm(diff_warm_gumbel[0, 2], ord='fro').item()
                            fro_warm_gumbel.update((r_warm_gumbel + g_warm_gumbel + b_warm_gumbel) / 3.0)

                            r_argmax = LA.norm(diff_argmax[0, 0], ord='fro').item()
                            g_argmax = LA.norm(diff_argmax[0, 1], ord='fro').item()
                            b_argmax = LA.norm(diff_argmax[0, 2], ord='fro').item()
                            fro_argmax.update((r_argmax + g_argmax + b_argmax) / 3.0)

                            r_categorical = LA.norm(diff_categorical[0, 0], ord='fro').item()
                            g_categorical = LA.norm(diff_categorical[0, 1], ord='fro').item()
                            b_categorical = LA.norm(diff_categorical[0, 2], ord='fro').item()
                            fro_categorical.update((r_categorical + g_categorical + b_categorical) / 3.0)

                            r_cold_gumbel = LA.norm(diff_cold_gumbel[0, 0], ord='fro').item()
                            g_cold_gumbel = LA.norm(diff_cold_gumbel[0, 1], ord='fro').item()
                            b_cold_gumbel = LA.norm(diff_cold_gumbel[0, 2], ord='fro').item()
                            fro_categorical.update((r_cold_gumbel + g_cold_gumbel + b_cold_gumbel) / 3.0)

                        mse_concrete_val = F.mse_loss(mod, rec, reduction='mean').item()
                        mse_warm_gumbel_val = F.mse_loss(mod, warm_gumbel_t_rec, reduction='mean').item()
                        mse_argmax_val = F.mse_loss(mod, argmax_t_rec, reduction='mean').item()
                        mse_categorical_val = F.mse_loss(mod, categorical_t_rec, reduction='mean').item()
                        mse_cold_gumbel_val = F.mse_loss(mod, cold_gumbel_t_rec, reduction='mean').item()

                        mse_concrete.update(mse_concrete_val)
                        mse_warm_gumbel.update(mse_warm_gumbel_val)
                        mse_argmax.update(mse_argmax_val)
                        mse_categorical.update(mse_categorical_val)
                        mse_cold_gumbel.update(mse_cold_gumbel_val)

                        psnr_concrete.update(torch.sum(10.0 * torch.log10(mod.max() ** 2 / (mse_concrete_val + 1e-10))))
                        psnr_warm_gumbel.update(torch.sum(10.0 * torch.log10(mod.max() ** 2 / (mse_warm_gumbel_val + 1e-10))))
                        psnr_argmax.update(torch.sum(10.0 * torch.log10(mod.max() ** 2 / (mse_argmax_val + 1e-10))))
                        psnr_categorical.update(torch.sum(10.0 * torch.log10(mod.max() ** 2 / (mse_categorical_val + 1e-10))))
                        psnr_cold_gumbel.update(torch.sum(10.0 * torch.log10(mod.max() ** 2 / (mse_cold_gumbel_val + 1e-10))))

                    if 'png' in cfg.exp.eval_mode:
                        U.save_image(last_batch, fp=f'{fp}/{str(i).zfill(5)}.png', normalize=True)

            if 'cat_util' in cfg.exp.eval_mode:
                fig = plt.figure(figsize=(cat_util.shape[1] / 100, cat_util.shape[1] / 100), dpi=100)
                ax = fig.add_subplot(2,2,1)
                ax.set_xlabel(f"Category $K$", fontsize=4)
                ax.set_ylabel(r'Latent variable $\bm{c}$', fontsize=4)
                plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(cat_util.cpu(), interpolation='none', cmap='bone')
                fig.savefig(f'{path}/cat_util_{nTok}.png', dpi=1000, pad_inches=0.0, bbox_inches="tight")

                fig = plt.figure(figsize=(cat_util.shape[1] / 100, cat_util.shape[1] / 100), dpi=100)
                ax = fig.add_subplot(2,2,1)
                ax.set_xlabel(f"Category $K$", fontsize=4)
                ax.set_ylabel(r'Latent variable $\bm{c}$', fontsize=4)
                plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(pmf.cpu(), interpolation='none', cmap='bone')
                fig.savefig(f'{path}/pmf_{nTok}.png', dpi=1000, pad_inches=0.0, bbox_inches="tight")

                fig = plt.figure(figsize=(cat_util.shape[1] / 100, cat_util.shape[1] / 100), dpi=100)
                ax = fig.add_subplot(2,2,1)
                ax.bar(torch.arange(256), min_max_masses[:, 0], color='b', width=0.5)

                fig.savefig(f'{path}/max_probs_{nTok}.png', dpi=1000, pad_inches=0.0, bbox_inches="tight")
                fig = plt.figure(figsize=(cat_util.shape[1] / 100, cat_util.shape[1] / 100), dpi=100)
                ax = fig.add_subplot(2,2,1)
                ax.bar(torch.arange(256), min_max_masses[:, 1], color='r', width=0.5)
                fig.savefig(f'{path}/min_probs_{nTok}.png', dpi=1000, pad_inches=0.0, bbox_inches="tight")
                # for spine in plt.gca().spines.values():
                    # spine.set_visible(False)
                    # fig.savefig(f'{path}/cat_util_{nTok}.png', dpi = 200, pad_inches=0.0, bbox_inches="tight")
                    # print(cat_util.shape)
                    # print(cat_util)
                    # exit()

            if 'hist' in cfg.exp.eval_mode:
                if cfg.exp.mod == 'cam':
                    cam_bins = hist_r[1]
                    axs[model_idx + 1].bar(cam_bins[:-1], count_r[model_idx] /
                                           count_r[model_idx].max(), color='r',
                                           alpha=alpha, width=width *
                                           (max_dist_infer / max_dist_orig))
                    p1, = axs[model_idx + 1].plot(cam_bins[:-1], cum_r[model_idx] /
                                            nBatches, 'r', alpha=alpha,)
                                            # label=f"$K={labels[model_idx]}$")
                    cam_bins = hist_g[1]
                    axs[model_idx + 1].bar(cam_bins[:-1], count_g[model_idx] /
                                           count_g[model_idx].max(), color='g',
                                           alpha=alpha, width=width *
                                           (max_dist_infer / max_dist_orig))
                    p2, = axs[model_idx + 1].plot(cam_bins[:-1], cum_g[model_idx] /
                                            nBatches, 'g', alpha=alpha,)
                                            # label=f"$K={labels[model_idx]}$")
                    cam_bins = hist_b[1]
                    axs[model_idx + 1].bar(cam_bins[:-1], count_b[model_idx] /
                                           count_b[model_idx].max(), color='b',
                                           alpha=alpha, width=width *
                                           (max_dist_infer / max_dist_orig))
                    p3, = axs[model_idx + 1].plot(cam_bins[:-1],
                                                  cum_b[model_idx] / nBatches, 'b',
                                                  alpha=alpha,)
                                                  # label=f"$K={labels[model_idx]}$")
                    axs[model_idx + 1].set_xlabel("Channel Intensity", fontsize=11)
                    axs[model_idx + 1].set_ylabel("Normalized Count", fontsize=11)
                    axs[model_idx + 1].tick_params(axis="x", labelsize=11)
                    axs[model_idx + 1].tick_params(axis="y", labelsize=11)
                    axs[model_idx + 1].set_xticks(np.linspace(-0.5, 0.5, 5))
                    axs[model_idx + 1].set_xticklabels([-1.0, -0.5, 0, 0.5, 1.0], fontsize=11)
                    # axs[model_idx + 1].legend(loc="upper left", frameon=False, prop={"size": 11})
                    axs[model_idx + 1].set_yticks([])
                    axs[model_idx + 1].grid(True, which="both", color="#999999", linestyle="--", alpha=0.25)
                    axs[model_idx + 1].text(0.8, 0.6, f"$K={labels[model_idx]}$",
                                            horizontalalignment="center", transform=axs[model_idx + 1].transAxes,
                                            fontsize=10,)

                    # original
                    if model_idx == 1:
                        cam_bins = orig_hist_r[1]
                        axs[0].bar(cam_bins[:-1], orig_count_r / orig_count_r.max(), color='r', alpha=alpha, width=width)
                        axs[0].plot(cam_bins[:-1], orig_cum_r / nBatches, 'r', alpha=alpha)
                        cam_bins = orig_hist_g[1]
                        axs[0].bar(cam_bins[:-1], orig_count_g / orig_count_g.max(), color='g', alpha=alpha, width=width)
                        axs[0].plot(cam_bins[:-1], orig_cum_g / nBatches, 'g', alpha=alpha)
                        cam_bins = orig_hist_b[1]
                        axs[0].bar(cam_bins[:-1], orig_count_b / orig_count_b.max(), color='b', alpha=alpha, width=width)
                        axs[0].plot(cam_bins[:-1], orig_cum_b / nBatches, 'b', alpha=alpha,)# label='Input')
                        axs[0].set_xlabel("Channel Intensity", fontsize=11)
                        axs[0].set_ylabel("Normalized Count", fontsize=11)
                        axs[0].tick_params(axis="x", labelsize=11)
                        axs[0].tick_params(axis="y", labelsize=11)
                        axs[0].set_xticks(np.linspace(-0.5, 0.5, 5))
                        axs[0].set_xticklabels([-1.0, -0.5, 0, 0.5, 1.0], fontsize=11)
                        # axs[0].legend(loc="upper left", frameon=False, prop={"size": 11})
                        axs[0].set_yticks([])
                        axs[0].grid(True, which="both", color="#999999", linestyle="--", alpha=0.25)
                        axs[0].text(0.8, 0.6, "Input", # 0.2 0.8
                                    horizontalalignment="center",
                                    transform=axs[0].transAxes, fontsize=10,)



                if cfg.exp.mod == 'rad':
                    rad_bins = hist_rad[1]
                    axs[model_idx + 1].bar(rad_bins[:-1], count_rad[model_idx] /
                                           count_rad[model_idx].max(), color='black', alpha=alpha,
                                           width=width * (max_dist_infer / max_dist_orig), align='edge')
                    axs[model_idx + 1].plot(rad_bins[:-1], cum_rad[model_idx] /
                                            nBatches, color='black',
                                            alpha=alpha,)
                                            # label=f"$K={labels[model_idx]}$")
                    axs[model_idx + 1].set_xlabel('Power [dB]', fontsize=11)
                    axs[model_idx + 1].set_ylabel('Normalized Count', fontsize=11)
                    axs[model_idx + 1].tick_params(axis="x", labelsize=11)
                    axs[model_idx + 1].tick_params(axis="y", labelsize=11)
                    axs[model_idx + 1].set_xticks(np.linspace(-1.0, 1.0, 5))
                    axs[model_idx + 1].set_xticklabels([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=11)
                    # axs[model_idx + 1].legend(loc="upper left", frameon=False, prop={"size": 11})
                    axs[model_idx + 1].set_yticks([])
                    axs[model_idx + 1].grid(True, which="both", color="#999999", linestyle="--", alpha=0.25)

                    axs[model_idx + 1].text(0.8, 0.6, f"$K={labels[model_idx]}$",
                                            horizontalalignment="center", transform=axs[model_idx + 1].transAxes,
                                            fontsize=10,)

                    if model_idx == 1:
                        rad_bins = orig_hist_rad[1]
                        axs[0].bar(rad_bins[:-1], orig_count_rad /
                                   orig_count_rad.max(), color='black',
                                   alpha=alpha, width=width, align='edge')
                        axs[0].plot(rad_bins[:-1], orig_cum_rad / nBatches,
                                    # color='black', label='$\mu=-0.038\,$dB \n $\sigma=\quad~ 0.107\,$dB', alpha=alpha)
                                    color='black', alpha=alpha, )
                        # label='Input',
                        axs[0].set_xlabel('Power [dB]', fontsize=11)
                        axs[0].set_ylabel('Normalized Count', fontsize=11)
                        axs[0].tick_params(axis="x", labelsize=11)
                        axs[0].tick_params(axis="y", labelsize=11)
                        # axs[0].legend(loc="upper left", frameon=False, prop={"size": 11})
                        axs[0].grid(True, which="both", color="#999999", linestyle="--", alpha=0.25)

                        axs[0].set_yticks([])
                        axs[0].text(0.8, 0.6, "Input",
                                    horizontalalignment="center",
                                    transform=axs[0].transAxes, fontsize=10,)
                        axs[0].set_xticks(np.linspace(-1.0, 1.0, 5))
                        axs[0].set_xticklabels([-1.0, -0.5, 0, 0.5, 1.0], fontsize=11)


                if 'fid' in cfg.exp.eval_mode:
                    print(data_arr.shape)
                    print('fdfdssd!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
                    data_arr = to_np(data_arr)
                    mu_dataset = np.mean(data_arr, axis=0)
                    sigma_dataset = np.cov(data_arr, rowvar=False)

                    warm_gumbel_arr = to_np(warm_gumbel_arr)
                    mu_warm_gumbel_infered = np.mean(warm_gumbel_arr, axis=0)
                    sigma_warm_gumbel_infered = np.cov(warm_gumbel_arr, rowvar=False)
                    fid_warm_gumbel_runs.append(calculate_frechet_distance(mu_warm_gumbel_infered, sigma_warm_gumbel_infered, mu_dataset, sigma_dataset))
                    categorical_arr = to_np(categorical_arr)
                    mu_categorical_infered = np.mean(categorical_arr, axis=0)
                    sigma_categorical_infered = np.cov(categorical_arr, rowvar=False)
                    fid_categorical_runs.append(calculate_frechet_distance(mu_categorical_infered, sigma_categorical_infered, mu_dataset, sigma_dataset))
                    argmax_arr = to_np(argmax_arr)
                    mu_argmax_infered = np.mean(argmax_arr, axis=0)
                    sigma_argmax_infered = np.cov(argmax_arr, rowvar=False)
                    fid_argmax_runs.append(calculate_frechet_distance(mu_argmax_infered, sigma_argmax_infered, mu_dataset, sigma_dataset))
                    cold_gumbel_arr = to_np(cold_gumbel_arr)
                    mu_cold_gumbel_infered = np.mean(cold_gumbel_arr, axis=0)
                    sigma_cold_gumbel_infered = np.cov(cold_gumbel_arr, rowvar=False)
                    fid_cold_gumbel_runs.append(calculate_frechet_distance(mu_cold_gumbel_infered, sigma_cold_gumbel_infered, mu_dataset, sigma_dataset))

                if 'norms' in cfg.exp.eval_mode:
                    fro_concrete_runs.append(fro_concrete.avg)
                    fro_warm_gumbel_runs.append(fro_warm_gumbel.avg)
                    fro_argmax_runs.append(fro_argmax.avg)
                    fro_categorical_runs.append(fro_categorical.avg)
                    fro_cold_gumbel_runs.append(fro_categorical.avg)

                    mse_concrete_runs.append(mse_concrete.avg)
                    mse_warm_gumbel_runs.append(mse_warm_gumbel.avg)
                    mse_argmax_runs.append(mse_argmax.avg)
                    mse_categorical_runs.append(mse_categorical.avg)
                    mse_cold_gumbel_runs.append(mse_categorical.avg)

                    psnr_concrete_runs.append(psnr_concrete.avg.cpu())
                    psnr_warm_gumbel_runs.append(psnr_warm_gumbel.avg.cpu())
                    psnr_argmax_runs.append(psnr_argmax.avg.cpu())
                    psnr_categorical_runs.append(psnr_categorical.avg.cpu())
                    psnr_cold_gumbel_runs.append(psnr_categorical.avg.cpu())

            if 'hist' in cfg.exp.eval_mode:
                if cfg.exp.mod == 'cam':
                    plt.savefig(f"{path}/cam_histogram_recon.pdf", dpi=100,) #bbox_inches="tight",
                if cfg.exp.mod == 'rad':
                    plt.savefig(f"{path}/rad_histogram_recon.pdf", dpi=100,) #bbox_inches="tight",

            if 'fid' in cfg.exp.eval_mode:
                fid_warm_gumbel_avg = np.mean(fid_warm_gumbel_runs)
                fid_warm_gumbel_std = np.std(fid_warm_gumbel_runs)
                fid_categorical_avg = np.mean(fid_categorical_runs)
                fid_categorical_std = np.std(fid_categorical_runs)
                fid_argmax_avg = np.mean(fid_argmax_runs)
                fid_argmax_std = np.std(fid_argmax_runs)
                fid_cold_gumbel_avg = np.mean(fid_cold_gumbel_runs)
                fid_cold_gumbel_std = np.std(fid_cold_gumbel_runs)

            if 'norms' in cfg.exp.eval_mode:
                fro_concrete_avg = np.mean(fro_concrete_runs)
                fro_concrete_std = np.std(fro_concrete_runs)
                fro_warm_gumbel_avg = np.mean(fro_warm_gumbel_runs)
                fro_warm_gumbel_std = np.std(fro_warm_gumbel_runs)
                fro_argmax_avg = np.mean(fro_argmax_runs)
                fro_argmax_std = np.std(fro_argmax_runs)
                fro_categorical_avg = np.mean(fro_categorical_runs)
                fro_categorical_std = np.std(fro_argmax_runs)
                fro_cold_gumbel_avg = np.mean(fro_categorical_runs)
                fro_cold_gumbel_std = np.std(fro_argmax_runs)

                mse_concrete_avg = np.mean(mse_concrete_runs)
                mse_concrete_std = np.std(mse_concrete_runs)
                mse_warm_gumbel_avg = np.mean(mse_warm_gumbel_runs)
                mse_warm_gumbel_std = np.std(mse_warm_gumbel_runs)
                mse_argmax_avg = np.mean(mse_argmax_runs)
                mse_argmax_std = np.std(mse_argmax_runs)
                mse_categorical_avg = np.mean(mse_categorical_runs)
                mse_categorical_std = np.std(mse_argmax_runs)
                mse_cold_gumbel_avg = np.mean(mse_categorical_runs)
                mse_cold_gumbel_std = np.std(mse_argmax_runs)

                psnr_concrete_avg = np.mean(psnr_concrete_runs)
                psnr_concrete_std = np.std(psnr_concrete_runs)
                psnr_warm_gumbel_avg = np.mean(psnr_warm_gumbel_runs)
                psnr_warm_gumbel_std = np.std(psnr_warm_gumbel_runs)
                psnr_argmax_avg = np.mean(psnr_argmax_runs)
                psnr_argmax_std = np.std(psnr_argmax_runs)
                psnr_categorical_avg = np.mean(psnr_categorical_runs)
                psnr_categorical_std = np.std(psnr_categorical_runs)
                psnr_cold_gumbel_avg = np.mean(psnr_categorical_runs)
                psnr_cold_gumbel_std = np.std(psnr_categorical_runs)



            with open(path + "/check_statistics", 'w') as f:
                if 'fid' in cfg.exp.eval_mode:
                    f.write(f"pytorch_{fid_argmax_avg=} {fid_argmax_std}\n")
                    f.write(f"check_pytorch_{fid_categorical_avg=} {fid_categorical_std}\n")
                    f.write(f"pytorch_{fid_categorical_avg=} {fid_categorical_std}\n")
                    f.write(f"pytorch_{fid_warm_gumbel_avg=} {fid_warm_gumbel_std}\n")
                    f.write(f"pytorch_{fid_cold_gumbel_avg=} {fid_cold_gumbel_std}\n\n")
                if 'norms' in cfg.exp.eval_mode:
                    f.write(f"{fro_argmax_avg=} {fro_argmax_std=}\n")
                    f.write(f"{fro_categorical_avg=} {fro_categorical_std=}\n")
                    f.write(f"{fro_concrete_avg=} {fro_concrete_std=}\n")
                    f.write(f"{fro_warm_gumbel_avg=} {fro_warm_gumbel_std=}\n")
                    f.write(f"{fro_cold_gumbel_avg=} {fro_cold_gumbel_std=}\n\n")

                    f.write(f"{mse_argmax_avg=} {mse_argmax_std=}\n")
                    f.write(f"{mse_categorical_avg=} {mse_categorical_std=}\n")
                    f.write(f"{mse_concrete_avg=} {mse_concrete_std=}\n")
                    f.write(f"{mse_warm_gumbel_avg=} {mse_warm_gumbel_std=}\n")
                    f.write(f"{mse_cold_gumbel_avg=} {mse_cold_gumbel_std=}\n\n")

                    f.write(f"{psnr_argmax_avg=} {psnr_argmax_std=}\n")
                    f.write(f"{psnr_categorical_avg=} {psnr_categorical_std=}\n")
                    f.write(f"{psnr_concrete_avg=} {psnr_concrete_std=}\n")
                    f.write(f"{psnr_warm_gumbel_avg=} {psnr_warm_gumbel_std=}\n")
                    f.write(f"{psnr_cold_gumbel_avg=} {psnr_cold_gumbel_std=}\n")
                if 'pca' in cfg.exp.eval_mode:
                    f.write(f"\n{variance_ratio=}\n")


        if 'pca' in cfg.exp.eval_mode:
            columns = ["Model", *[str(i + 1) for i in range(num_pca)]]
            variance_ratio = pd.DataFrame(variance_ratio, columns=columns)
            print(variance_ratio.shape)

            # print(variance_ratio)
            print(variance_ratio.sum(1))
            # exit()

            variance_ratio = variance_ratio.melt(id_vars="Model", var_name="Pricipal Component",
                                                 value_name="Explained Variance Ratio")
            print(variance_ratio)
            print(variance_ratio.shape)

            pca_plot = sns.lineplot(x="Pricipal Component", y="Explained Variance Ratio", hue="Model", data=variance_ratio)
            fig = pca_plot.get_figure()
            fig.axes[0].grid(True, which="both", color="#999999", linestyle="--", alpha=0.25)
            fig.axes[0].legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.9))
            fig.axes[0].legend(['K=64', 'K=256', 'K=1024'], frameon=False)

            # fig.axes[0].set_yscale('logit')
            print('Plot saved to ', fp)
            fig.savefig(fp + "/variance_ratio.png")

            num_rows = len(embeddings)
            num_columns = 8
            fig = plt.figure(figsize=(2 * num_columns, 2 * num_rows))
            fig.subplots_adjust(wspace=-0.1)
            for i, model in enumerate(embeddings.keys(), 1):
                norms = np.linalg.norm(embeddings[model]["emb"], axis=-1)
                norms /= norms.max()
                for j, pca in enumerate(embeddings[model]["pcas"], 1):
                    ax = fig.add_subplot(num_rows, num_columns, j + (i - 1) * num_columns, projection='3d')
                    components = pca.transform(embeddings[model]["emb"][j - 1, ...])
                    X = components[:, 0]
                    Y = components[:, 1]
                    Z = components[:, 2]
                    ax.scatter(X, Y, Z, c=norms[j-1, :], s=2)
                    if j == 1:
                        ax.text2D(-0.1, 0.25, model, transform=ax.transAxes, rotation="vertical", fontsize="large")

                    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

                    mid_x = (X.max()+X.min()) * 0.5
                    mid_y = (Y.max()+Y.min()) * 0.5
                    mid_z = (Z.max()+Z.min()) * 0.5
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)

                    ax.axes.xaxis.set_ticklabels([])
                    ax.axes.yaxis.set_ticklabels([])
                    ax.axes.zaxis.set_ticklabels([])

            fig.savefig(fp + "/codebooks.png")

        exit('Data written to file - inference complete')


    print(model)
    # from torchsummary import summary
    # summary(model, (1, 256, 256))
    # exit()

    best_loss_total = 1e5
    train_step = 0
    nTok = model.discretizer.nTokens
    for epo in range(1, cfg.exp.max_epoch + 1):
        for phase in data:
            model.train(phase == 'train')
            loss_total, loss_kl, loss_rec = AvgMeter(), AvgMeter(), AvgMeter()
            soft_cluster_use, hard_cluster_use, cold_cluster_use = AvgMeter(), AvgMeter(), AvgMeter()
            soft_perplexity, hard_perplexity, cold_perplexity = AvgMeter(), AvgMeter(), AvgMeter()
            loss_soft_t_valid, loss_hard_t_valid, loss_cold_t_valid = AvgMeter(), AvgMeter(), AvgMeter()
            neg_ent_loss = AvgMeter()
            kl_weight, temperature = AvgMeter(), AvgMeter()
            soft_bins, hard_bins, cold_bins = torch.zeros(nTok), torch.zeros(nTok), torch.zeros(nTok)
            num_iters_so_far = len(data[phase]) * (epo - 1)
            for i, batch in enumerate(t := tqdm(data[phase], leave=False)):
                mod = batch[cfg.exp.mod].to(dev)
                # print(mod.min())
                # print(mod.max())
                # exit()
                batch_size = mod.shape[0]
                num_pixel = mod.numel() // batch_size
                train_step = num_iters_so_far + i if model.training else train_step
                with torch.set_grad_enabled(phase == 'train'):
                    kl_loss, rec, perplx, clus_use, neg_ent = model(mod, cfg.exp.soft_idx)
                    rec_loss = F.mse_loss(mod, rec)
                    # rec_loss = loss(mod, rec)
                    total_loss = kl_loss / num_pixel + rec_loss
                    # total_loss = kl_loss / 1.0 + rec_loss
                    # elbo = (KL - logp) / N
                    # bpd = elbo / np.log(2)
                    if phase == "train":
                        zero_grads(model)
                        total_loss.backward()
                        optim_com.step()

                    neg_ent_loss(neg_ent.item())
                    loss_total(total_loss.item(), batch_size)
                    loss_rec(rec_loss.item(), batch_size)
                    loss_kl(kl_loss.item(), batch_size)
                    kl_weight(model.discretizer.kl_weight.v)
                    temperature(model.discretizer.temperature.temp)
                    soft_perplexity(perplx[0].item())
                    hard_perplexity(perplx[1].item())
                    cold_perplexity(perplx[2].item())
                    soft_cluster_use(clus_use[0].item())
                    hard_cluster_use(clus_use[1].item())
                    cold_cluster_use(clus_use[2].item())
                    if phase == "valid":
                        soft_t, hard_t, cold_t, froz_t = model.discretize(mod, cfg.exp.soft_idx)
                        soft_bins += torch.bincount(soft_t.view(-1), minlength=nTok).cpu()
                        hard_bins += torch.bincount(hard_t.view(-1), minlength=nTok).cpu()
                        cold_bins += torch.bincount(cold_t.view(-1), minlength=nTok).cpu()
                        soft_t_rec, hard_t_rec, cold_t_rec = (model.constitute(t) for t in (soft_t, hard_t, cold_t))

                        loss_soft_t_valid(F.mse_loss(mod, soft_t_rec).item())
                        loss_hard_t_valid(F.mse_loss(mod, hard_t_rec).item())
                        loss_cold_t_valid(F.mse_loss(mod, cold_t_rec).item())

                t.set_description( f"{phase} epo: {epo} step {train_step} total_loss: {total_loss:.10f}")
            logging.info(f"{phase} {epo=}: {loss_total.avg=:.10f} {loss_rec.avg=:.10f} {loss_kl.avg=:.8f}")

            if not args.dry:
                tb.plot(cfg.exp.name, phase, epo, Negative_Entropy=neg_ent_loss.avg)
                tb.plot(cfg.exp.name, phase, epo, Absolute_error=loss_total.avg)
                tb.plot(cfg.exp.name, phase, epo, KL_error=loss_kl.avg)
                tb.plot(cfg.exp.name, phase, epo, LL_error=loss_rec.avg)
                tb.plot(cfg.exp.name, phase, epo, KL_weight=kl_weight.avg)
                tb.plot(cfg.exp.name, phase, epo, Temperature=temperature.avg)
                tb.plot(cfg.exp.name, phase, epo, SoftPerplexity=soft_perplexity.avg)
                tb.plot(cfg.exp.name, phase, epo, HardPerplexity=hard_perplexity.avg)
                tb.plot(cfg.exp.name, phase, epo, ColdPerplexity=cold_perplexity.avg)
                tb.plot(cfg.exp.name, phase, epo, SoftCluster_use=soft_cluster_use.avg)
                tb.plot(cfg.exp.name, phase, epo, HardCluster_use=hard_cluster_use.avg)
                tb.plot(cfg.exp.name, phase, epo, ColdCluster_use=cold_cluster_use.avg)
                if phase == 'train':
                    last_batch = layout(mod, rec, only_last=cfg.exp.show, mod=cfg.exp.mod)
                    tb.disp(cfg.exp.name, phase, epo, last_batch)

        lrs_com.step(loss_total.avg)
        logging.info(f'Num bad epochs: {lrs_com.num_bad_epochs}')
        logging.info(f'Best: {lrs_com.best}')
        if not args.dry and loss_total.avg < best_loss_total * (1 - 1e-4):
            best_loss_total= loss_total.avg
            logging.info(f"New lowest loss reached in {epo=}... {best_loss_total}")
            if cfg.exp.save_model:
                save_model(epo, model, optim_com, res_dir, best_loss_total, cfg_dict)
        if epo % cfg.exp.log_interval == 0 and not args.dry:
            with torch.no_grad():

                tb.dist(cfg.exp.name, phase, epo, distribution=soft_t,
                        bins=model.discretizer.nTokens)
                tb.hist(cfg.exp.name, epo, model.discretizer.embedding)

                fig = plt.figure()
                plt.bar(torch.arange(len(soft_bins)), 20 * torch.log10(soft_bins) ** 2, color='g', alpha=.5,)
                tb.fig(cfg.exp.name + "/log_soft_token", epo, plt.gcf())
                plt.bar(torch.arange(len(hard_bins)), 20 * torch.log10(hard_bins) ** 2, color='r', alpha=.5,)
                tb.fig(cfg.exp.name + "/log_hard_token", epo, plt.gcf())
                plt.bar(torch.arange(len(cold_bins)), 20 * torch.log10(cold_bins) ** 2, color='b', alpha=.5,)
                tb.fig(cfg.exp.name + "/log_cold_token", epo, plt.gcf())

                last_batch = layout(mod, soft_t_rec, hard_t_rec, cold_t_rec,
                                    rec, only_last=cfg.exp.show, mod=cfg.exp.mod)
                tb.disp(cfg.exp.name, phase, epo, last_batch)

                tb.plot(cfg.exp.name, phase, epo, SoftTokenReconError=loss_soft_t_valid.avg)
                tb.plot(cfg.exp.name, phase, epo, HardTokenReconError=loss_hard_t_valid.avg)
                tb.plot(cfg.exp.name, phase, epo, ColdTokenReconError=loss_cold_t_valid.avg)


if __name__ == "__main__":
    main()
