import argparse
import time
from datetime import datetime
import sys
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pywt
from numpy import fft
import torch.nn.functional as F
from PIL import Image
from torchvision import utils as U
from tqdm import trange, tqdm
import logging
from skimage import exposure
from scipy import stats

# from mayavi import mlab
from lib import load_config, to_np
from lib import InrasData
from lib import wpt, draw_1d_wp_basis
from lib import get_obj_size, setup_logging
from lib import InceptionV3, get_activations, calculate_frechet_distance, torch_cov


def pure():
    timing_list = []
    for e in range(1):
        start = time.time()
        print("epoch", e)
        aligned = non_aligned = 0
        # print("size of dataloader", sizeof(df))
        # if e == 1:
        for batch_idx, dp in enumerate(tqdm(df)):
            # pass
            # print("BATCH IDX", batch_idx)
            # print("rad shape", dp["rad"].shape)
            # if both_mod:
                # print("cam rad shape", dp["cam_rad"].shape)
            cam = dp["cam"].view(dp["cam"].shape[0], 3, -1)
            rad = dp["rad"].view(dp["rad"].shape[0], 1, -1)
            bsze = dp["rad"].shape[0]
            """
            # print("cam shape", cam.shape)
            print("cam min", cam.min())
            print("cam max", cam.max())

            meanCam = cam.mean(2).sum(0).unsqueeze(1) / bsze
            stdCam = cam.std(2).sum(0).unsqueeze(1) / bsze

            print('meanCam', meanCam)
            print('stdCam', stdCam)

            # cam = (cam - meanCam) / stdCam
            print('After Sclaing')
            print("cam min", cam.min())
            print("cam max", cam.max())

            meanCam = cam.mean(2).sum(0) / cam.shape[0]
            stdCam = cam.std(2).sum(0)/ cam.shape[0]
            print('meanCam', meanCam)
            print('stdCam', stdCam)


            print("rad min", rad.min())
            print("rad max", rad.max())

            meanRad = rad.mean(2).sum(0).unsqueeze(1) / bsze
            stdRad = rad.std(2).sum(0).unsqueeze(1) / bsze

            print('meanRad', meanRad)
            print('stdRad', stdRad)

            # rad = (rad - meanRad) / stdRad
            print('After Sclaing')
            print("rad min", rad.min())
            print("rad max", rad.max())


            meanRad = rad.mean(2).sum(0) / rad.shape[0]
            stdRad = rad.std(2).sum(0)/ rad.shape[0]
            print('meanRad', meanRad)
            print('stdRad', stdRad)
            # print("rad type", dp["rad"].dtype)
            # print(get_obj_size(dp["rad"]))
            # print(get_obj_size(dp["rad"]) + get_obj_size(dp["cam_rad"]))

            # rad = dp["rad"]
            # print(rad[:, 0, 0].shape)
            # print(rad[:, 1, 0].shape)
            # rad1 = rad[:, 0, 0]
            # rad2 = rad[:, 1, 0]
            # equal = torch.all(rad1.eq(rad2))
            # print(equal)
            # if equal:
            # aligned += 1
            # else:
            # non_aligned += 1

            # time.sleep(0.05)
            # continue
            # exit()
            # dp['cam'].cuda()
            # dp['rad'].cuda()
            # dp['lab'].cuda()
            """

        stop = time.time()
        print("aligned", aligned)
        print("non aligned", non_aligned)
        timing_list.append(stop - start)
        print("Duration", stop - start)
    print(timing_list)


def frechet_inception_distance():
    dev = 'cuda'
    dims = 2048
    mod = 'rad'

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], resize_input=False, normalize_input=False, use_fid_inception=True).to(dev)
    # print(model)

    print("Calculating inception parameters for dataset")
    act = get_activations(df.size, df, mod, model, dims=dims, device=dev)
    act = to_np(act)
    mu_dataset = np.mean(act, axis=0)
    sigma_dataset = np.cov(act, rowvar=False)

    print('check sum act', np.sum(act)) #5379666
    print('check sum mu', np.sum(mu_dataset)) #748
    print('check sum sigma', np.sum(sigma_dataset)) #27796

    print("Calculating inception parameters for generated data")
    # cfg.viz.args["src_dir"] = cfg.viz.args['cmp_dir']
    # df_infered = InrasData.loader(**cfg.viz.args)
    mod = 'cam'

    act = get_activations(df.size, df, mod, model, dims=dims, device=dev)
    act = to_np(act)
    mu_infered = np.mean(act, axis=0)
    sigma_infered = np.cov(act, rowvar=False)
    # print(sigma_infered.dtype)

    fid_value = calculate_frechet_distance(mu_dataset, sigma_dataset, mu_infered, sigma_dataset)
    print(fid_value)
    exit()
    with open(f"{cfg.viz.args['dst_dir']}/fid_mu_sigma", 'w') as f:
        f.write(f"{mu_dataset=}\n{sigma_dataset=}\n")
        f.write(f"{mu_infered=}\n{sigma_infered=}\n")
        f.write(f"{fid_value=}\n")


def save_to_disk(): # CAUTION resize cam imgg to 256x256, disable bufferqueu, batch
    for i, batch in enumerate(bar := tqdm(df)):
        rad, cam, bsze = batch["rad"], batch["cam"], batch["rad"].shape[0]
        # print(rad.shape)
        # print(cam.shape)
        # exit()
        if rad.ndim > 4:  # cube
            rad = torch.rot90(rad, 1, [-3, -2])
            rA = rad.mean(-2)
            rD = rad.mean(-1)
            rD[..., -32:] = rA
        else:
            rad = torch.rot90(rad, 1, [-2, -1])
            rad = rad[:, :1]
            rD = rad.mean(-3).unsqueeze(1)
        # for j in range(bsze):
            # augRad = U.make_grid(rD[j], normalize=True)
            # augCam = U.make_grid(cam[j], normalize=True)
            # sample = torch.cat([augRad, augCam], 1)
            # U.save_image(
                # sample,
                # fp=cfg.viz.args["dst_dir"] + f"/{str(i*bsze+j).zfill(5)}.png",
                # nrow=20,
                # padding=0,
            # )
        augRad = U.make_grid(rD, nrow=10, normalize=True)
        augCam = U.make_grid(cam, nrow=10, normalize=True)
        sample = torch.cat([augRad, augCam], 2)
        U.save_image(
            sample,
            fp=cfg.viz.args["dst_dir"] + f"/{str(i).zfill(5)}.png",
            # nrow=10,
            padding=0,
        )

def survey_statistics():
    nSamples = 0

    meanRad =  0.
    stdRad = 0.
    varRad = 0.
    unitVarRad = 0.
    minRad = float("Inf")
    maxRad = -float("Inf")

    meanCam = 0.
    stdCam = 0.
    varCam = 0.
    unitVarCam = 0.
    # minCam = [float("Inf")] * 3
    # maxCam = [-float("Inf")] * 3
    minCam = torch.tensor([float("Inf")] * 3)
    maxCam = torch.tensor([-float("Inf")] * 3)

    for batch in tqdm(df, leave=True):
        bsze = batch["rad"].shape[0]
        rad = batch["rad"].view(bsze, -1)
        cam = batch["cam"].view(bsze, 3, -1)

        # Rad: channel-wise mean and std and summation over batch
        meanRad += rad.mean(1).sum(0)
        stdRad += rad.std(1).sum(0)
        varRad += rad.var(1).sum(0)
        unitVarRad += (rad / -414.6096191406).var(1).sum(0)

        # min/max without dim returns global min/max
        current_min_rad = rad.min()
        if current_min_rad < minRad:
            minRad = current_min_rad

        current_max_rad = rad.max()
        if current_max_rad > maxRad:
            maxRad = current_max_rad

        # Cam: channel-wise mean and std and summation over batch
        meanCam += cam.mean(2).sum(0)
        stdCam += cam.std(2).sum(0)
        varCam += cam.var(2).sum(0)
        unitVarCam += (cam / 255.0).var(2).sum(0)

        # print(unitVarCam)
        # c = cam / 255.0
        # print(c.min())
        # print(c.max())
        # exit()

        # amin just returns values and not indices
        current_min_cam_chn = cam.amin(dim=(0, 2)) # min over batch and entries for every channel
        minCam = torch.minimum(current_min_cam_chn, minCam) # returns channel-wise min
        current_max_cam_chn = cam.amax(dim=(0, 2)) # max over batch and entries for every channel
        maxCam = torch.maximum(current_max_cam_chn, maxCam) # returns channel-wise max

        # for i in range(current_min_cam.shape[1]):
            # current_min_cam_chn = cam.amin(dim=(0, 2))[0][:, i]
            # if current_min_cam_chn[:, i] < minCam[i]:
                # minCam[i] = current_min_cam[:, i]

        # current_max_cam = cam.amax(dim=2)[0]
        # for i in range(current_max_cam.shape[1]):
            # if current_max_cam[:, i] > maxCam[i]:
                # maxCam[i] = current_max_cam[:, i]

        nSamples += bsze
        # break

    meanRad /= nSamples
    stdRad /= nSamples
    varRad /= nSamples
    unitVarRad /= nSamples

    meanCam /= nSamples
    stdCam /= nSamples
    varCam /= nSamples
    unitVarCam /= nSamples

    _, stem = os.path.split(cfg.viz.args['src_dir'])
    with open(f"{cfg.viz.args['dst_dir']}/{stem}.statistics_may_batched_1k_var_valid_scaled", 'w') as f:
        f.write(f"{meanRad=}\n{stdRad=}{varRad=}\n{unitVarRad=}\n{minRad=}\n{maxRad=}\n")
        f.write(f"\n{meanCam=}\n{stdCam=}\n{varCam=}\n{unitVarCam=}\n{minCam=}\n{maxCam=}\n")


def histogram():

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
    # matplotlib.rcParams['text.usetex'] = True


    nSamples = 0
    nBatches = 0
    nb_bins = 256
    nb_bins_rad = 256
    # nb_bins = 50
    # nb_bins_rad = 50
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)

    cum_r = np.zeros(nb_bins)
    cum_g = np.zeros(nb_bins)
    cum_b = np.zeros(nb_bins)

    count_rad = np.zeros(nb_bins_rad)
    cum_rad = np.zeros(nb_bins_rad)

    for batch in tqdm(df, leave=True):
        # rad = batch["rad"].view(bsze, -1).numpy()
        # cam = batch["cam"].view(bsze, 3, -1).numpy()
        rad = batch["rad"].numpy()
        cam = batch["cam"].numpy()
        nSamples += rad.shape[0]
        nBatches += 1
        # print(rad.min())
        # print(rad.max())
        # exit()
        # print('cam min', cam.min())
        # print('cam max', cam.max())
        # print('rad min', rad.min())
        # print('rad max', rad.max())

        # t_flattened = batch["cam"].view(batch["cam"].shape[0], batch["cam"].shape[1], -1)
        # mean = t_flattened.mean(2)
        # mean = mean.sum(0) / batch["cam"].shape[0]
        # std = t_flattened.std(2).sum(0) / batch["cam"].shape[0]
        # print('cam mean', mean)
        # print('cam std', std)
        # t_flattened = batch["rad"].view(batch["rad"].shape[0], batch["rad"].shape[1], -1)
        # mean = t_flattened.mean(2)
        # mean = mean.sum(0) / batch["rad"].shape[0]
        # std = t_flattened.std(2).sum(0) / batch["rad"].shape[0]
        # print('rad mean', mean)
        # print('rad std', std)


        hist_r = np.histogram(cam[:, 0], bins=nb_bins)
        hist_g = np.histogram(cam[:, 1], bins=nb_bins)
        hist_b = np.histogram(cam[:, 2], bins=nb_bins)

        count_r += hist_r[0]
        count_g += hist_g[0]
        count_b += hist_b[0]

        img_cdf_r, bins = exposure.cumulative_distribution(cam[:, 0], nb_bins)
        img_cdf_g, bins = exposure.cumulative_distribution(cam[:, 1], nb_bins)
        img_cdf_b, bins = exposure.cumulative_distribution(cam[:, 2], nb_bins)

        cum_r += img_cdf_r
        cum_g += img_cdf_g
        cum_b += img_cdf_b

        hist_rad = np.histogram(rad, bins=nb_bins_rad) #, range=[0, 255])
        count_rad += hist_rad[0]
        img_cdf_rad, bins = exposure.cumulative_distribution(rad, nb_bins)
        cum_rad += img_cdf_rad

    cam_bins = hist_r[1]
    rad_bins = hist_rad[1]
    # print(count_r)
    # print(cam_bins)

    # fig = plt.figure()
    width = 0.005
    width = 1
    alpha = 0.5

    fig, ax = plt.subplots()
    plt.bar(cam_bins[:-1], count_r / count_r.max(), color='r', alpha=alpha, width=width)
    # plt.plot(cam_bins[:-1], cum_r / nBatches, 'r', label='$\mu=120.92$\quad $\sigma=60.72$') #train
    plt.plot(cam_bins[:-1], cum_r / nBatches, 'r', label='$\mu=115.19$\quad $\sigma=58.78$')
    # plt.plot(cam_bins[:-1], cum_r / nBatches, 'r', label='$\mu=-0.096$\quad $\sigma=0.451$')#scaled

    plt.bar(cam_bins[:-1], count_g / count_g.max(), color='g', alpha=alpha, width=width)
    # plt.plot(cam_bins[:-1], cum_g / nBatches, 'g', label='$\mu=117.90$\quad $\sigma=62.07$') #train
    plt.plot(cam_bins[:-1], cum_g / nBatches, 'g', label='$\mu=114.30$\quad $\sigma=60.01$')
    # plt.plot(cam_bins[:-1], cum_g / nBatches, 'g', label='$\mu=-0.103$\quad $\sigma=0.460$')#scaled

    plt.bar(cam_bins[:-1], count_b / count_b.max(), color='b', alpha=alpha, width=width)
    # plt.plot(cam_bins[:-1], cum_b / nBatches, 'b', label='$\mu=117.67$\quad $\sigma=66.45$') #train
    plt.plot(cam_bins[:-1], cum_b / nBatches, 'b', label='$\mu=114.87$\quad $\sigma=64.50$')
    # plt.plot(cam_bins[:-1], cum_b / nBatches, 'b', label='$\mu=-0.099$\quad $\sigma=0.497$') #scaled

    plt.grid(True, which="both", color="#999999", linestyle="--", alpha=0.25)

    # plt.xlabel('Pixel Intensity', fontsize=15)
    plt.xlabel('Channel Intensity', fontsize=15)
    plt.ylabel('Normalized Count', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    legend = ax.legend(loc="upper left", frameon=False, prop={'size': 12}) #, bbox_to_anchor=(1.0, 0.9))
    # ax.set_yticks([])
    plt.savefig(f"{cfg.viz.args['dst_dir']}/cam_histogram_scaled.pdf")

    fig, ax = plt.subplots()
    plt.bar(rad_bins[:-1], count_rad / count_rad.max(), color='black', alpha=alpha, width=width)
    # plt.plot(rad_bins[:-1], cum_rad / nBatches, 'black', label='$\mu=-230.57\,$dB \n $\sigma=\quad~ 14.34\,$dB') #train
    plt.plot(rad_bins[:-1], cum_rad / nBatches, 'black', label='$\mu=-230.18\,$dB \n $\sigma=\quad~ 14.33\,$dB')
    # plt.plot(rad_bins[:-1], cum_rad / nBatches, 'black', label='$\mu=-0.038\,$dB \n $\sigma=\quad~ 0.107\,$dB')#scaled
    plt.grid(True, which="both", color="#999999", linestyle="--", alpha=0.25)
    # plt.xlabel('Pixel Intensity', fontsize=15)
    plt.xlabel('Power [dB]', fontsize=15)
    plt.ylabel('Normalized Count', fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    legend = ax.legend(loc="upper left", frameon=False, prop={'size': 12}) #, bbox_to_anchor=(1.0, 0.9))
    plt.savefig(f"{cfg.viz.args['dst_dir']}/rad_histogram_scaled.pdf")


def WPT():
    data_len = 512
    wave = pywt.Wavelet("db2")
    # wave = pywt.Wavelet("db1")
    mode = "periodization"
    J = pywt.dwt_max_level(data_len=data_len, filter_len=wave)
    # J = 2
    # J = 5
    idx = 0
    print("decom level", J)
    p3d = (12, 0)
    xfm = wpt.WPTForward(J=J, wave=wave, mode=mode).cuda()
    win = np.hanning(512)
    start = time.time()
    fig, axes = plt.subplots(1, 1, figsize=[8, 8])
    for i, seq in enumerate(df):
        if_signal = seq["rad"][:, :data_len, 0]
        ft_signal = np.fft.fft2(if_signal)
        ft_signal = np.fft.fftshift(ft_signal, axes=0)
        plt.imshow(20 * np.log10(np.abs(ft_signal)))
        plt.draw()
        plt.pause(1e-5)
        plt.clf()
        continue
        chirp = torch.from_numpy(if_signal).unsqueeze(0).unsqueeze(0).float().cuda()
        # chirp = F.pad(chirp, p3d, "constant", 0)
        res = xfm(chirp)
        xfm.compute_best_basis()
        print("Best basis computation finished")
        label_levels = 0  # how many levels to explicitly label on the plots
        # print(xfm._best_tree[0][0])
        # for node in xfm._best_tree:
        # if len(node[0]) < J:
        # print(len(node[0]))
        # raise Exception
        print("seq:", i, end="\r")
        # continue
        # continue
        # print("len best tree: ", len(xfm._best_tree))
        # for node in xfm._best_tree:
        # print("Best basis: ", node[0])
        ret = draw_1d_wp_basis(
            keys=xfm._best_tree,
            ax=axes,
            label_levels=label_levels,
            shape=[512, 512],
        )
        # fig.show()
        fig.tight_layout()
        axes.set_xlabel("Time")
        axes.set_ylabel("Frequency")
        # plt.gca().set_axis_off()
        # plt.margins(0, 0)
        plt.setp(axes.spines.values(), color="w")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(f"/home/ditzel/convert/wpt/{str(idx).zfill(5)}.png")
        plt.pause(1e-3)
        plt.cla()
        idx += 1
    stop = time.time()
    print("Duration", stop - start)


def DWT():
    xfm = DWTForward(J=9, wave="db1", mode="zero").cuda()

    start = time.time()
    if_signal = next(iter(df))["rad"]
    if_signal = if_signal[if_signal.shape[0] // 2, ..., if_signal.shape[-1] // 2]
    plt.plot(if_signal)
    idx = 0
    for i, seq in enumerate(df):
        print("seq:", i, "-" * 10)
        if_signal = seq["rad"]
        print(if_signal.shape)
        for j, ant in enumerate(range(if_signal.shape[-1])):
            print("ant", j)
            # for chirp in range(if_signal.shape[0]):
            idx += 1

            chirp = (
                torch.from_numpy(if_signal[..., ant])
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .cuda()
            )
            res = xfm(chirp)
            # print("DWT SHAPE", res[0].shape)
            # print("DWT SHAPE", res[1][0].shape)
            # print("DWT SHAPE", res[1][1].shape)
            # print("DWT SHAPE", res[1][2].shape)
            # print('antenna', ant)
            # print('chirp', chirp)
            # print(if_signal[chirp, :, ant])
            # print('data:', if_signal[chirp, :, ant])
            # plt.plot(if_signal[chirp, :, ant])
            # plt.draw()
            # plt.pause(0.01)
            # plt.clf()
        # exit()
    print(idx)
    stop = time.time()
    print("Duration", stop - start)


def rawImage():
    # for i, seq in enumerate(df):
    #     print("seq:", i, "-" * 10)
    #     print(seq["rad"].shape)
    #     print(seq["rad"][..., 0].shape)
    #     a = seq["rad"][..., 0]
    #     a_min = np.min(a)
    #     a_max = np.max(a)
    #     a_scaled = 255 * (a - a_min) / (a_max - a_min)
    #     plt.imshow(a, cmap=plt.cm.gray)
    #     plt.show()
    #     plt.pause(0.01)
    #     plt.cla()

    # img = next(iter(df))["rad"]
    myobj = plt.imshow(np.ones([512, 512]))
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    # fig, ax = plt.subplots()
    for i, ds in enumerate(df):
        print("img", i)
        img = ds["rad"][..., 0:, 0]
        myobj.set_data(img)
        plt.gca().set_axis_off()
        a = img
        print(a.shape)
        a_min = np.min(a)
        a_max = np.max(a)
        a_scaled = 255 * (a - a_min) / (a_max - a_min)
        plt.imshow(a, cmap=plt.cm.gray)
        plt.draw()
        plt.savefig(f"/home/ditzel/raw{i}.png", bbox_inches="tight", pad_inches=0)
        plt.pause(0.05)
        plt.clf()


def IF_signal():
    nIni = 2
    start = time.time()
    if_signal = next(iter(df))["rad"]
    print(if_signal.shape)
    if_signal = if_signal[if_signal.shape[0] // 2, ..., if_signal.shape[-1] // 2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.set_ylabel("LSB [V]")
    ax.set_xlabel("Sample duration [s]")

    xticks = np.linspace(0, len(if_signal), 10)
    offset = nIni * 51.2e-6 / (len(if_signal) - nIni)
    print("offset", offset)

    xticklabels = [
        format(label, ".6f")
        for label in (tick for tick in np.linspace(offset, 51.2e-6, 10))
        # format(label, ".6f")
        # for label in (tick for tick in np.arange(0, 51.2e-6, 2.56e-6))
    ]
    print(xticklabels)
    ax.set_xticks(xticks)  # [0:-1:velBinStep])
    ax.set_xticklabels(
        xticklabels,
        fontsize=10,
    )

    # plt.show()
    # exit()
    # plt.axis([None, None, -0.01, 0.01])

    idx = 0
    for i, seq in enumerate(df):
        print("seq:", i, "-" * 10)
        if_signal = seq["rad"]
        print(if_signal.shape)
        for ant in range(1):
            print("ant:", ant)
            for chirp in range(1):
                print("chirp: ", chirp)
                data = if_signal[128, :, 0]
                print("shape", data.shape)
                # data = np.pad(data, (16, 16), "constant", constant_values=(0, 0),)
                ax.plot(data[nIni:])
                ax.set_xticks(xticks)  # [0:-1:velBinStep])
                ax.set_xticklabels(
                    xticklabels,
                    fontsize=10,
                )
                ax.set_ylabel("LSB [V]")
                ax.set_xlabel("Sample duration [s]")
                ax.set_title(f"Frame {idx}")
                plt.grid(True)
                # ax.axis("equal")
                plt.axis([None, None, -1500, 1500])

                # plt.margins(0, 0)
                # plt.subplots_adjust(
                # top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                # )
                plt.savefig(f"/home/ditzel/convert/ifs/{str(idx).zfill(5)}.png")

                fig.show()
                plt.pause(0.01)
                plt.cla()
                idx += 1
                print(idx)
            # exit()
        # exit()
    print(idx)
    stop = time.time()
    print("Duration", stop - start)


def camera():
    img = next(iter(df))["cam"]

    myobj = plt.imshow(np.ones([480, 640, 3]))
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    # fig, ax = plt.subplots()
    for _ in range(1):
        for i, ds in enumerate(df):
            print("img", i)
            img = ds["cam"]
            print("img shape", img.shape)
            myobj.set_data(img)
            # comment below lines in for video frame saving without white borders
            plt.gca().set_axis_off()
            plt.margins(0, 0)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.savefig(
                f"/home/ditzel/radar/data/data_images/cam_{str(i).zfill(3)}.png"
            )
            # exit()
            plt.draw()
            plt.pause(0.01)
            # if i == 10:
            # break


def range_profile():
    print("Showing Range Profile")
    start = time.time()
    first_seq = next(iter(df))["rad"]
    s = first_seq[first_seq.shape[0] // 2, ..., first_seq.shape[-1] // 2]
    print(s.shape)
    JdB = 20 * np.log10(np.abs(s))
    # JdB = s
    # JMax = np.amax(JdB)
    # JNorm = JdB - JMax
    # JNorm[JNorm < -20] = -20
    JNorm = JdB
    plt.plot(JNorm)
    for i, seq in enumerate(df):
        print("seq:", i)
        s = seq["rad"]
        s = s[s.shape[0] // 2, ..., s.shape[-1] // 2]
        JdB = 20 * np.log10(np.abs(s))
        JNorm = JdB
        plt.plot(JNorm)
        plt.draw()
        plt.pause(0.01)
        plt.clf()
    stop = time.time()
    print("Duration", stop - start)


def rD_map():
    start = time.time()
    first_seq = next(iter(df))["rad"]

    if len(first_seq.shape) == 2:

        print(first_seq.shape)
    else:
        first_seq = first_seq[..., first_seq.shape[-1] // 2]
    first_seq = np.swapaxes(first_seq, 0, 1)
    first_seq = np.flipud(first_seq)

    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # _, (ax1, ax2) = plt.subplots(1)
    fig, ax = plt.subplots()
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    # ax.set_axis_off()
    # ax.axis("off")
    # ax.set_ylabel("Range [m]")

    # rangeBinStep = int(np.around(240 / (25 - 1)))
    # ax.set_xticks(np.arange(-236, 236))
    # ax.set_xticklabels(
    # [format(label, ".1f") for label in range_grid_excerpt[0:-1:rangeBinStep]],
    # fontsize=6,
    # )
    # start, end = ax.get_xlim()
    # ax.xaxis.set_ticks(np.arange(start, end, 1))

    ax.set_title("range-Doppler map")
    # ax.set_xlabel("Velocity [m/s]")
    # ax.set_ylabel("Range [m]")
    # plt.xticks(np.arange(-5, 5, 1.0))

    # JMax = np.amax(JdB)
    # JNorm = JdB - JMax
    # JNorm[JNorm < -20] = -20
    # JNorm = s
    # myobj = plt.imshow(JNorm)
    myobj = plt.imshow(first_seq)
    for i, seq in enumerate(df):
        print("seq:", i)
        first_seq = seq["rad"]
        # print("data:", seq["rad"])
        if len(first_seq.shape) != 2:
            first_seq = first_seq[..., s.shape[-1] // 2]
        first_seq = np.swapaxes(first_seq, 0, 1)
        first_seq = np.flipud(first_seq)

        # JMax = np.amax(JdB)
        # JNorm = JdB - JMax
        # JNorm[JNorm < -20] = -20
        myobj.set_data(first_seq)
        plt.savefig(
            f"/home/ditzel/radar/data/data_images/rD{str(i).zfill(3)}.png",
            bbox_inches="tight",
        )
        # plt.draw()

        plt.tight_layout()
        plt.pause(0.1)


def rA_map():
    first_seq = next(iter(df))["rad"]
    s = first_seq[first_seq.shape[0] // 2]
    # JdB = 20 * np.log10(s)
    s = np.flipud(s)
    JMax = np.amax(s)
    JNorm = s - JMax
    JNorm[JNorm < -20] = -20

    fig, ax = plt.subplots()
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    # JNorm = JdB
    ax.set_title("range-Azimuth map")

    myobj = plt.imshow(JNorm)
    for i, seq in enumerate(df):
        print("seq:", i)
        s = seq["rad"]
        s = s[s.shape[0] // 2]
        s = np.flipud(s)

        # print(s.shape)
        # JdB = 20 * np.log10(s)
        # JMax = np.amax(s)
        # JNorm = s - JMax
        # JNorm[JNorm < -20] = -20
        # JNorm = JdB
        # myobj.set_data(JNorm)

        myobj.set_data(s)
        plt.savefig(
            f"/home/ditzel/convert/rD{str(i).zfill(3)}.png",
            bbox_inches="tight",
        )
        plt.tight_layout()
        plt.draw()
        # plt.imshow(JNorm)
        plt.pause(0.1)


def radar_cube():
    first_seq = next(iter(df))["rad"]
    # JdB = 20 * np.log10(first_seq)
    # JMax = np.amax(JdB)
    # JNorm = JdB - JMax
    mlab.figure(size=(1024, 768), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    # JNorm[JNorm < -20] = -20
    JNorm = first_seq
    src = mlab.pipeline.scalar_field(JNorm)
    s = mlab.pipeline.iso_surface(
        src,
        contours=[
            JNorm.max() - 0.2 * JNorm.ptp(),
            JNorm.min() + 0.35 * JNorm.ptp(),
        ],
        opacity=0.75,
    )
    mlab.gcf().scene.parallel_projection = True

    @mlab.animate(delay=100, ui=False)
    def anim():
        f = mlab.gcf()
        # f.background = (0.0, 0.0, 0.0)

        for i, seq in enumerate(df):
            print(i)
            f.scene.camera.azimuth(0.2)
            f.scene.render()
            # data = 20 * np.log10(seq["rad"])
            s.mlab_source.scalars = seq["rad"]
            mlab.savefig(
                f"/home/ditzel/convert/cube{str(i).zfill(3)}.png",
                figure=f,
                magnification="auto",
            )
            yield

    mlab.axes(
        xlabel="Velocity",
        ylabel="range",
        zlabel="azimuth",
        ranges=[-5, 5, 1, 25, -90, 90],
    )
    a = anim()  # Starts the animation without a UI.
    mlab.show()


def rD_map_and_cam():
    def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
        dst = Image.new(
            "RGB", (im1.width + im2.width, max(im1.height, im2.height)), color
        )
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    start = time.time()
    first_seq = next(iter(df))
    rad, cam, _ = first_seq.values()

    if len(rad.shape) != 2:
        rad = rad[..., rad.shape[-1] // 2]
    rad = np.swapaxes(rad, 0, 1)
    rad = np.flipud(rad)

    cam = Image.fromarray(cam)
    rad = Image.fromarray(rad)
    get_concat_h_blank(cam, rad).save(
        "/home/ditzel/radar/data/data_images_static/rD_cam{str(1).zfill(3)}.png",
    )

    # exit()

    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # _, (ax1, ax2) = plt.subplots(1)
    fig, ax = plt.subplots()
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    # ax.set_axis_off()
    # ax.axis("off")
    # ax.set_ylabel("Range [m]")

    # rangeBinStep = int(np.around(240 / (25 - 1)))
    # ax.set_xticks(np.arange(-236, 236))
    # ax.set_xticklabels(
    # [format(label, ".1f") for label in range_grid_excerpt[0:-1:rangeBinStep]],
    # fontsize=6,
    # )
    # start, end = ax.get_xlim()
    # ax.xaxis.set_ticks(np.arange(start, end, 1))

    ax.set_title("range-Doppler map")
    # ax.set_xlabel("Velocity [m/s]")
    # ax.set_ylabel("Range [m]")
    # plt.xticks(np.arange(-5, 5, 1.0))

    # JMax = np.amax(JdB)
    # JNorm = JdB - JMax
    # JNorm[JNorm < -20] = -20
    # JNorm = s
    # myobj = plt.imshow(JNorm)

    first_seq = next(iter(df))["cam"]
    myobj = plt.imshow(first_seq)

    for i, seq in enumerate(df):
        print("seq:", i)
        first_seq = seq["cam"]
        # print("data:", seq["rad"])
        # if len(first_seq.shape) != 2:
        # first_seq = first_seq[..., s.shape[-1] // 2]
        # first_seq = np.swapaxes(first_seq, 0, 1)
        # first_seq = np.flipud(first_seq)

        # JMax = np.amax(JdB)
        # JNorm = JdB - JMax
        # JNorm[JNorm < -20] = -20
        myobj.set_data(first_seq)
        plt.savefig(
            f"/home/ditzel/radar/data/data_images_static/cam{str(i).zfill(3)}.png",
            bbox_inches="tight",
        )
        # plt.draw()

        plt.tight_layout()
        plt.pause(0.1)


def sizeof(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)):
        return size + sum(map(sizeof, obj))
    return size


# cfg, _ = load_config("/home/ditzel/radar/config.yaml")
cfg, _, _ = load_config("/home/ditzel/rad/lib/config.yaml")
# print(cfg.viz.args)
df = InrasData.loader(**cfg.viz.args)
print("Sequenzen Laenge:", len(df))
for dsp_mode in cfg.viz.display_modes:
    eval(dsp_mode)()
