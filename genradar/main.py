import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from os.path import abspath
from os import path
import logging
import shutil
import random
import argparse
import time
import sys
from datetime import datetime
from collections import OrderedDict
from functools import partial

from tqdm import trange, tqdm
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
from torch.nn.utils import clip_grad_norm_
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange, reduce, repeat
from matplotlib import pyplot as plt
import seaborn as sns
import PIL
import math
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib as mpl

import lib
from lib import (
    InrasData,
    setup_logging,
    load_config,
    get_Host_name_IP,
    seed_everything,
    AvgMeter,
    save_model,
    save_model_reg,
    TensorboardLogger,
    layout,
    disable_feature_for_certain_model_paramters,
    requires_grad,
    to_np,
    zero_grads,
    InceptionV3,
    calculate_frechet_distance
)

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

torch.set_printoptions(profile="full", linewidth=200, precision=10)
parser = argparse.ArgumentParser(description="Radar Camera Fusion")
parser.add_argument("-c", default="config.yaml")
parser.add_argument("-d", default="__oO__", help="description")
parser.add_argument("-resume", default=None, help="resume from checkpoint")
parser.add_argument("-dry", action='store_true', help="dry run")
args = parser.parse_args()
cfg, cfg_str, cfg_dict = load_config(args.c)


def color_bar(batch):
    v_border_left = torch.arange(0, 8)
    v_border_right = torch.arange(batch.shape[-1] - 1, batch.shape[-1] - 9, -1)
    vert = torch.linspace(256, batch.shape[-1] - 256- 1, steps=batch.shape[-1] // 256 -1).long()
    vert = torch.sort(torch.cat([vert - 1, vert, vert + 1]))[0].long()

    red_border = torch.tensor([1031, 1032, 1033, 1034, 1287, 1288, 1289, 1290])
    red_color = torch.FloatTensor([0.8, 0, 0]).unsqueeze(1).unsqueeze(2).to(batch)
    green_border = (0, 1, 256, 257)
    green_color = torch.FloatTensor([0, 0.8, 0]).unsqueeze(1).unsqueeze(2).to(batch)
    blue_border = (774, 775, 1030, 1031)
    blue_color = torch.FloatTensor([0, 0, 0.8]).unsqueeze(1).unsqueeze(2).to(batch)
    yellow_border = (516, 517, 772, 773, 1290, 1291, 1546, 1547)
    yellow_color = torch.FloatTensor([0.8, 0.8, 0]).unsqueeze(1).unsqueeze(2).to(batch)
    purple_border = (258, 259, 514, 515)
    purple_color = torch.FloatTensor([0.625, 0.125, 0.9375]).unsqueeze(1).unsqueeze(2).to(batch)

    batch[:, red_border] = red_color
    batch[:, red_border[0]:red_border[7], v_border_left] = red_color
    batch[:, red_border[0]:red_border[7], v_border_right] = red_color
    # batch[:, red_border[0]:red_border[7], vert] = red_color

    batch[:, blue_border[0]:blue_border[3], v_border_left] = blue_color
    batch[:, blue_border[0]:blue_border[3], v_border_right] = blue_color
    # batch[:, blue_border[0]:blue_border[3], vert] = blue_color

    batch[:, yellow_border[0]:yellow_border[3], v_border_left] = yellow_color
    # batch[:, yellow_border[0]:yellow_border[3], vert] = yellow_color
    batch[:, yellow_border[4]:yellow_border[7], v_border_left] = yellow_color
    # batch[:, yellow_border[4]:yellow_border[7], vert] = yellow_color

    batch[:, green_border[0]:green_border[3], v_border_right] = green_color
    batch[:, green_border[0]:green_border[3], v_border_left] = green_color
    # batch[:, green_border[0]:green_border[3], vert] = green_color

    batch[:, yellow_border[0]:yellow_border[3], v_border_right] = yellow_color
    # batch[:, yellow_border[0]:yellow_border[3], vert] = yellow_color
    batch[:, yellow_border[4]:yellow_border[7], v_border_right] = yellow_color
    # batch[:, yellow_border[4]:yellow_border[7], vert] = yellow_color

    batch[:, purple_border[0]:purple_border[3], v_border_left] = purple_color
    batch[:, purple_border[0]:purple_border[3], v_border_right] = purple_color
    # batch[:, purple_border[0]:purple_border[3], vert] = purple_color
    return batch

def color_bar_cam_prd(batch):
    v_border_left = torch.arange(0, 8)
    v_border_right = torch.arange(batch.shape[-1] - 1, batch.shape[-1] - 9, -1)

    green_border = (0, 1, 256, 257)
    green_color = torch.FloatTensor([0, 0.8, 0]).unsqueeze(1).unsqueeze(2).to(batch)

    red_border = torch.tensor([516, 517, 518, 519, 770, 771, 772, 773])
    red_color = torch.FloatTensor([0.8, 0, 0]).unsqueeze(1).unsqueeze(2).to(batch)

    blue_border = (258, 259, 514, 515)
    blue_color = torch.FloatTensor([0, 0, 0.8]).unsqueeze(1).unsqueeze(2).to(batch)

    batch[:, green_border[0]:green_border[3], v_border_right] = green_color
    batch[:, green_border[0]:green_border[3], v_border_left] = green_color

    batch[:, red_border] = red_color
    batch[:, red_border[0]:red_border[7], v_border_left] = red_color
    batch[:, red_border[0]:red_border[7], v_border_right] = red_color

    batch[:, blue_border[0]:blue_border[3], v_border_left] = blue_color
    batch[:, blue_border[0]:blue_border[3], v_border_right] = blue_color
    return batch


def showcase(model, rad, cam, epo, phase, tb, loss, res_dir, rad_faithfullness, cam_faithfullness):
    *samples, correct_constitutes, sim, rad_faith, cam_faith, cam_stepwise = model.synthesize(rad, cam, cfg.exp.temp, cfg.exp.selection_thres,
                                                                                              cfg.exp.num_samples, cfg.exp.rad_synth_s,
                                                                                              cfg.exp.cam_synth_s, cfg.exp.noise_cond,
                                                                                              cam_stepwise=cfg.exp.cam_stepwise)


    # if cfg.model.args['load_path'] is not None:
        # better choose batch_size = 1 for this
        # if True:
        # fig = plt.figure()
        # ax = fig.add_subplot(2,2,1)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # trafo_layer = model.transformer.layers
        # att_maps = torch.cat([layer.attn.att_maps for layer in trafo_layer], dim=0)
        # att_maps = rearrange(att_maps, 'l head h w -> (l h ) (head w)').cpu()
        # ax.imshow(att_maps, interpolation='none', cmap='bone')
        # fig.savefig(f'{res_dir}/attn_{epo=}.png', dpi = 2000, pad_inches=0.0, bbox_inches="tight")

    rad = torch.rot90(rad, 1, [-2, -1])
    last_batch = layout(rad, *samples[:2], cam, *samples[2:], only_last=cfg.exp.show)
    tb.disp(cfg.exp.name + ' inference', phase, epo, last_batch)
    tb.plot(cfg.exp.name, phase, epo, Absolute_error=loss.avg)
    tb.plot(cfg.exp.name, phase, epo, Rad_Faithfullness=rad_faithfullness.avg) # avg over entire valid set
    tb.plot(cfg.exp.name, phase, epo, Cam_Faithfullness=cam_faithfullness.avg)# avg over entire valid set
    tb.plot(cfg.exp.name, phase, epo, Rad_Faithfullness_Synth=rad_faith) # only over last batch/epoch
    tb.plot(cfg.exp.name, phase, epo, Cam_Faithfullness_Synth=cam_faith) # only over last batch/epoch
    tb.plot(cfg.exp.name, phase, epo, cosine_sim_synth=sim.item())
    tb.plot(cfg.exp.name, phase, epo, cor_tokens=correct_constitutes.item())

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
            model_state = state['model_state_dict']
            missing, unexpected = model.load_state_dict(model_state, strict=True)
            res_dir, _ = os.path.split(ckpt_path)
            res_dir += '/'
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
        param_groups = disable_feature_for_certain_model_paramters(model, 'weight_decay')
        optim = eval(f"optim.{cfg.optim.name}")(param_groups, **cfg.optim.args)
        # optim = eval(f"optim.{cfg.optim.name}")(model.parameters(), **cfg.optim.args)
        lrs = eval(f"lr.{cfg.lrs.name}")(optim, **cfg.lrs.args)
        loss = eval(f"nn.{cfg.loss.name}")(**cfg.loss.args)

    # from torchsummary import summary
    # summary(model, (1, 256, 256))

    data = {
        "train": InrasData.loader(**cfg.train.args),
        "valid": InrasData.loader(**cfg.valid.args)
    }

    # stuff for stiched patch visualization
    # cam_toks = torch.ones(64, 3, 256, 256).to(dev)
    # for t in range(64):
        # cam_t = t * torch.ones(256).int().to(dev).unsqueeze(0)
        # cam_toks[t] = model.rad_vae.constitute(cam_t)

    # cam_toks =  model.rad_vae.constitute(torch.arange(256).int().to(dev).unsqueeze(0))
    # U.save_image(cam_toks, nrow=int(math.sqrt(cam_toks.shape[0])),
                 # fp=f'{res_dir}/rad_arange_256.png', normalize=True, scale_each=True)
    # exit()
    if cfg.model.args['load_path'] is not None:
        """Batch size of 1 for video rendering with png output and lmdb serialization"""
        model_list = cfg.model.args['load_path']

        dims = 2048

        for model_idx, saved_model in enumerate(model_list):
            print(f'Performing inference on {saved_model}')
            cfg.model.args['load_path'] = saved_model
            model = eval(f"lib.{cfg.model.name}")(**cfg.model.args).to(dev).eval()
            state = torch.load(saved_model)["config"]
            del state['meta']
            path, stem = os.path.split(saved_model)
            samples_seen = 0
            # fp = path + '/' + 'eval'
            # fp = path + '/' + 'roundabout'
            fp = path + '/' + 'max'
            os.makedirs(fp, exist_ok=True)
            if 'cam_prd_two_rows' in cfg.exp.eval_mode:
                frames = 2
                frame_start = 0
                second_list = range(0, frames)
                cam_vid = torch.empty([frames, 3, 256, 256]).to(dev)
                rad_vid = torch.empty([frames, 3, 256, 256]).to(dev)
                prd_vid = torch.empty([frames, 3, 256, 256]).to(dev)

            if 'fid' in cfg.exp.eval_mode:
                start_idx = 0
                data_arr = torch.zeros((data['valid'].size, dims))
                pred_arr = torch.zeros((data['valid'].size, dims))
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
                inception = InceptionV3([block_idx], resize_input=False,
                                        normalize_input=False, use_fid_inception=False).to(dev).eval()

            for i, batch in enumerate(t := tqdm(data['valid'], leave=True)):
                rad, cam = batch['rad'].to(dev), batch['cam'].to(dev)
                batch_size = cam.shape[0]
                sample_step = samples_seen + cam.shape[0]
                with torch.no_grad():
                    if 'fid' in cfg.exp.eval_mode:
                        _, _, _, _, _, _, _, _, cam_modes = model(rad, cam,
                                                                  cfg.exp.rad_train_s, cfg.exp.cam_train_s,
                                                                  noise_cond=cfg.exp.noise_cond, last_iter= False)

                        prd = model.cam_vae.constitute(cam_modes)
                        data_arr[start_idx:start_idx + cam.shape[0]] = inception(cam)[0].squeeze(3).squeeze(2)
                        pred_arr[start_idx:start_idx + prd.shape[0]] = inception(prd)[0].squeeze(3).squeeze(2)
                        start_idx = start_idx + cam.shape[0]

                if 'syn' in cfg.exp.eval_mode or 'cam_prd_two_rows' or 'reveal' or 'attn' in cfg.exp.eval_mode:
                    rad_prd, rad_dec, cam_prd, cam_dec, _, _, _, _, cam_stepwise = model.synthesize(rad, cam,
                                                                          cfg.exp.temp, cfg.exp.selection_thres,
                                                                          cfg.exp.num_samples, cfg.exp.rad_synth_s,
                                                                          cfg.exp.cam_synth_s, cfg.exp.noise_cond,
                                                                          cam_stepwise=cfg.exp.cam_stepwise)
                    if 'attn' in cfg.exp.eval_mode:
                        fig = plt.figure()
                        ax = fig.add_subplot(2,2,1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        trafo_layer = model.transformer.layers
                        att_maps = torch.cat([layer.attn.att_maps for layer in trafo_layer], dim=0)
                        att_maps = rearrange(att_maps, 'l head h w -> (l h ) (head w)').cpu()
                        ax.imshow(att_maps, interpolation='none', cmap='binary')

                    fig.savefig(f'{path}/attn_{i=}.png', dpi = 2000, pad_inches=0.0, bbox_inches="tight")
                    exit()
                    if 'reveal' in cfg.exp.eval_mode:
                        rows = range(16, 256, 32)
                        for batch_sample in range(cam.shape[0]):
                            cam_gt = cam[batch_sample].cpu()
                            rad_gt = rad[batch_sample].cpu()
                            cam_steps = cam_stepwise[batch_sample, rows]
                            rad_gt = repeat(torch.rot90(rad_gt, 1, [-2, -1]), 'c h w -> (c copy) h w', copy=3)
                            cam_steps = torch.cat([rad_gt.unsqueeze(0), cam_steps, cam_gt.unsqueeze(0)], dim=0)
                            U.save_image(cam_steps,
                                         nrow=5,
                                         fp=f'{path}/{str(batch_sample+1).zfill(3)}_cam_stepwise.png', normalize=True,
                                         scale_each=True)
                        exit()

                    if 'syn' in cfg.exp.eval_mode:
                        batch = layout(torch.rot90(rad, 1, [-2, -1]), rad_prd, rad_dec, cam, cam_prd, cam_dec)
                        batch = color_bar(batch)
                        U.save_image(batch,
                                     fp=f'{fp}/sample_{str(i).zfill(4)}_{str(samples_seen+1).zfill(4)}_{str(sample_step).zfill(4)}.png',
                                     normalize=True, scale_each=True)
                        samples_seen = sample_step

                    if 'cam_prd_two_rows' in cfg.exp.eval_mode: # batch size 20 for 2s apart time sequence
                        offset = 0
                        if i >= offset:
                            cam_vid[i - offset] = cam[frame_start]
                            rad_vid[i - offset] = rad[frame_start]
                            prd_vid[i - offset] = cam_prd[frame_start]
                        if i == frames + offset - 1:
                            break
            if 'cam_prd_two_rows' in cfg.exp.eval_mode:

                # batch = layout(torch.rot90(rad_vid, 1, [-2, -1]), cam_vid, prd_vid)
                batch = layout(torch.rot90(rad_vid, 1, [-2, -1]), cam_vid, prd_vid)
                batch = color_bar_cam_prd(batch)
                # U.save_image(batch, fp=f'{fp}/k=thld0_tau1_k256_10s.png', normalize=True, scale_each=True)

                batch = U.make_grid(batch, normalize=True, scale_each=True)
                print(batch.shape)

                # for i in range(cam_vid.shape[0]):
                # img = tf.to_pil_image(batch)
                # img = img.resize((3 * batch.shape[2], 3 * batch.shape[1]), resample=Image.LANCZOS)
                # draw = ImageDraw.Draw(img)
                # font = ImageFont.truetype("DejaVuSansMono.ttf", 80)
                # font = ImageFont.truetype("Helvetica.ttf", 60)
                # draw.text((30, 800), f't=0s', (0, 0, 0), font=font)
                # img = img.resize((batch.shape[2], batch.shape[1]), resample=Image.LANCZOS)
                # batch = tf.to_tensor(img)
                # print(batch.shape)
                # U.save_image(batch, fp=f'{path}/test.png', normalize=True, scale_each=True)

                fig = plt.figure(figsize=(batch.shape[1] / 100, batch.shape[2] / 100), dpi=100, frameon=False)
                ax = fig.add_subplot(2,2,1)

                # mpl.rcParams['savefig.pad_inches'] = 0
                ax.set_xlabel(f"$t=0 \qquad \qquad \quad t=2 \qquad  $", fontsize=4)
                ax.set_ylabel(r'Latent variable $\bm{c} \qquad\qquad\qquad\qquad$', fontsize=4)
                # plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
                ax.set_xticks([])
                ax.set_yticks([])
                batch = rearrange(batch, 'c h w -> h w c')
                ax.imshow(batch.cpu(), interpolation='none', cmap='bone')
                fig.savefig(f'{path}/test.png', dpi=1000, pad_inches=0.0, bbox_inches="tight")


            if 'fid' in cfg.exp.eval_mode:
                data_arr = to_np(data_arr)
                mu_data = np.mean(data_arr, axis=0)
                sigma_data = np.cov(data_arr, rowvar=False)

                pred_arr = to_np(pred_arr)
                mu_pred = np.mean(pred_arr, axis=0)
                sigma_pred = np.cov(pred_arr, rowvar=False)
                fid_mode = calculate_frechet_distance(mu_pred, sigma_pred, mu_data, sigma_data)
                with open(path + "/fid_score", 'w') as f:
                    f.write(f"pytorch_{fid_mode=}")
        exit('Data written to file - inference complete')




    best_loss_total = 1e5
    for epo in range(1, cfg.exp.max_epoch + 1):
        for phase in data:
            num_batches = len(data[phase])
            model.train(phase == 'train')
            loss_total, cam_faithfullness, rad_faithfullness, similarity = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
            d=f"{phase} epo: {epo}"
            fp = f"{res_dir}/{desc}__epo{epo:03}"
            for it, batch in enumerate(tqdm(data[phase], leave=False, desc=d), start=1):
                rad, cam = batch['rad'].to(dev), batch['cam'].to(dev)
                with torch.set_grad_enabled(phase == 'train'):
                    loss, rad_faith, cam_faith, rad_p, cam_p, rad_dec, cam_dec, sim, _= model(rad, cam, cfg.exp.rad_train_s, cfg.exp.cam_train_s, noise_cond=cfg.exp.noise_cond, last_iter=True if it == num_batches else False)

                    # logits, target = model(rad, cam, cfg.exp.sampling)
                    # total_loss = loss(logits, target)
                    if phase == 'train':
                        (loss / cfg.exp.collect_grad).backward() # scale gradients
                        # TODO: Not sure if clip grads before or after above scaling
                        # clip_grad_norm_(param_groups, cfg.exp.max_norm) # cut norm of params
                        # clip_grad_norm_(param_groups[0]['params'], cfg.exp.max_norm) # cut norm of params
                        # clip_grad_norm_(param_groups[1]['params'], cfg.exp.max_norm) # cut norm of params
                        if it % cfg.exp.collect_grad == 0 or it == len(data[phase]):
                            optim.step() # every collect_grad iteration or when last batch
                            zero_grads(model)
                    rad_faithfullness(rad_faith)
                    cam_faithfullness(cam_faith)

                loss_total(loss.item(), rad.shape[0])
                similarity(sim.item())
            logging.info(f"{phase} {epo}: {loss_total.avg=:.5f}")
            last_batch = layout(torch.rot90(rad, 1, [-2, -1]),
                                torch.rot90(rad_p, 1, [-2, -1]),
                                torch.rot90(rad_dec, 1, [-2, -1]), cam, cam_p,
                                cam_dec, only_last=cfg.exp.show)
            tb.disp(cfg.exp.name + ' fitting', phase, epo, last_batch)
            tb.plot(cfg.exp.name, phase, epo, cosine_sim=similarity.avg)

            if epo % cfg.exp.log_interval == 0 and not args.dry: # always the last batch
                showcase(model, rad, cam, epo, phase, tb, loss_total, res_dir, rad_faithfullness, cam_faithfullness)
        lrs.step(loss_total.avg)
        if not args.dry and loss_total.avg < best_loss_total * (1 - 1e-4):
            best_loss_total = loss_total.avg
            logging.info(f"New lowest loss reached in {epo=}: {best_loss_total:.5f}")
            if cfg.exp.save_model:
                save_model(epo, model, optim, res_dir, best_loss_total, cfg_dict)
        if not args.dry and epo % 10 == 0:
            current_loss_total = loss_total.avg
            logging.info(f"Interval saving in {epo=}: {current_loss_total:.5f}")
            if cfg.exp.save_model:
                save_model_reg(epo, model, optim, res_dir, current_loss_total, cfg_dict)

if __name__ == '__main__':
    main()
