import itertools
import datetime
import queue
import operator
import math
import threading
import random
import logging
import os
import sys
import copy
import lmdb
import h5py
import platform
import pprint
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict
from enum import Enum
from time import perf_counter
import pickle
from functools import partial
import six

from einops import rearrange
import torch
from torchvision.transforms import functional as tf
from torchvision import transforms as T

import numpy as np
from numpy.fft import fft, rfft, fftshift
from scipy import constants
from scipy.fftpack.helper import next_fast_len as nfl
from pathlib import Path
from typing import Dict

# import logging

import dataflow as D
from dataflow.utils.utils import get_tqdm, fix_rng_seed
from dataflow.utils import logger

# from .auxiliary import rescaling_via_batch, standardization_via_batch, rescaling_via_train_data_cam, standardization_via_train_data_cam, rescaling_via_ex_train_data_rad, rescaling_via_in_train_data_rad, standardization_via_in_train_data_rad, standardization_via_ex_train_data_rad, ImageNet_normalization
from .auxiliary import *

def scrape_files(top_level_path, suffix, recursive=True):
    path = Path(top_level_path)
    logging.debug(f"Searching for {suffix} files in", path)
    assert path.is_dir(), f"no {suffix} files found at {top_level_path}"
    if recursive:
        file_paths = sorted(path.glob(f"**/*.{suffix}"))
    else:
        file_paths = sorted(path.glob(f"*.{suffix}"))
    if not file_paths:
        raise RuntimeError(f"No {suffix} files found")
    return list(str(path.resolve()) for path in file_paths)


class InrasHDF5Source(D.DataFlow):
    def __init__(self, hdf5_path, params, recursive=True):

        for k, v in params.items():
            setattr(self, k, v)

        self._recursive = recursive
        self.data_files = dict()
        self.nSeq_skipped = 0
        # self._logger = logging.getLogger(self.__class__.__name__)

        if os.path.isfile(hdf5_path):
            self._hdf5_path = hdf5_path
            hdf5_files = [hdf5_path]
        elif os.path.isdir(hdf5_path):
            self._hdf5_path = hdf5_path
            hdf5_files = scrape_files(self._hdf5_path, "h5", self._recursive)
        else:
            raise NotImplementedError("file type not supported")

        # for path in self.data_paths:
        for p in hdf5_files:
            file = h5py.File(p, "r", swmr=True, libver="latest")
            if not file.keys():
                print(f"CAUTION! Skipping {p} due to empty keys: {file.keys()}")
            else:
                rec = self._fetch_params(file)
                assert rec["cam"]["nPkt_img"] == 16, "one cam sample every 16 pkts"
                self.data_files[p] = rec
                logging.info(
                    f'Cube extend: {rec["rad"]["vMaxIdx"] - rec["rad"]["vMinIdx"]} x {rec["rad"]["rMaxIdx"] - rec["rad"]["rMinIdx"]} x {rec["rad"]["aPad"]}'
                )

                logging.info(f'Recording features {rec["rad"]["nChn_rec"]} RX')
        logging.info(
            f" {self.nSeq_skipped} sequences discarded due to missings camera frames in a total of {len(self.data_files)} hdf5 files"
        )

    def __len__(self):
        return sum(rec["rad"]["nSeq_rec"] for rec in self.data_files.values())

    def _allocate_cube(self, rad):
        return np.empty(
            [
                rad["nChi_seq"] // rad["nTdm"],
                rad["N"],
                rad["nChn_rec"] * rad["nTdm"],
            ],
            "int16",
        )

    def _fill_cube(self, seq, rec):
        seq_srt = seq * rec["rad"]["nChi_seq"]
        seq_end = (seq + 1) * rec["rad"]["nChi_seq"]
        for chn, chn_ds in rec["sChn_rec"].items():
            self.cube[..., chn] = chn_ds[seq_srt : seq_end : rec["rad"]["nTdm"]]
            self.cube[..., chn + rec["rad"]["nChn_rec"]] = chn_ds[
                seq_srt + 1 : seq_end : rec["rad"]["nTdm"]
            ]
        return self.cube

    def _rearrange_cam_img(self, seq, rec):
        seq_srt = seq * rec["cam"]["nCol_img"] * rec["cam"]["nChn_img"]
        seq_end = (seq + 1) * rec["cam"]["nCol_img"] * rec["cam"]["nChn_img"]
        sCam_seq = rec["sCam_rec"][seq_srt:seq_end].reshape(
            rec["cam"]["nRow_img"],
            rec["cam"]["nCol_img"],
            rec["cam"]["nChn_img"],
        )[..., (2, 1, 0)]
        return sCam_seq

    def __iter__(self):
        print("Total length:", self.__len__())
        for p, rec in self.data_files.items():
            try:
                print("Converting file:", p)

                self.cube = self._allocate_cube(rec["rad"])
                for seq_idx in range(rec["rad"]["nSeq_rec"]):
                    # print(p)
                    # print(seq_idx)
                    rec["rad"]["seq_idx"] = rec["cam"]["seq_idx"] = seq_idx
                    rec["rad"]["orig_path"] = rec["cam"]["orig_path"] = p
                    rec["rad"]["data"] = self._fill_cube(seq_idx, rec)
                    rec["cam"]["data"] = self._rearrange_cam_img(seq_idx, rec)

                    yield {"rad": rec["rad"], "cam": rec["cam"]}
            except Exception as e:
                print(
                    f"PROBLEM while constructing raw cube from file {p}, skipping file"
                )
                continue

    def _fetch_params(self, file):
        rec = dict(file.attrs.items())
        # range parameter
        rad = dict()
        cam = dict()

        rec["sChn_rec"] = {
            int(ds_name[3:]) - 1: file[ds_name]
            for ds_name in file.keys()
            if ds_name.startswith("Chn") and "Time" not in ds_name
        }

        rad["nChn_rec"] = rec["nChn_rec"].item()
        rad["nTdm"] = rec["nTdm"].item()
        rad["nChi_seq"] = rec["nChi_seq"].item()
        rad["nSeq_rec"] = rec["nSeq_rec"].item()
        rad["fAdc"] = rec["fAdc"].item()
        rad["N"] = rec["N"].item()
        rad["fs"] = rec["fs"].item()
        rad["f0"] = rec["f0"].item()
        rad["fs_eff"] = rec["fs_eff"].item()
        rad["tSam"] = rec["tSam"].item()
        rad["tCri"] = rec["tCri"].item()
        rad["tDem"] = rec["tDem"].item()
        rad["tCpi"] = rec["tCpi"].item()
        rad["tIdl"] = rec["tIdl"].item()
        rad["tSeq"] = rec["tSeq"].item()
        rad["fbw"] = rec["fbw"].item()
        rad["rMax"] = rec["rMax"].item()
        rad["rMax_eff"] = rec["rMax_eff"].item()
        rad["rMax_lpf"] = rec["rMax_lpf"].item()
        rad["vMax"] = rec["vMax"].item()
        rad["dv"] = rec["dv"].item()
        rad["fD_max"] = rec["fD_max"].item()
        rad["df"] = rec["df"].item()
        rad["dr"] = rec["dr"].item()
        rad["fc"] = rec["fc"].item()
        rad["fStrt"] = rec["fStrt"].item()
        rad["fStop"] = rec["fStop"].item()
        rad["fu_sca"] = rec["fu_sca"].item()
        rad["la_f0"] = rec["la_f0"].item()
        rad["la_fc"] = rec["la_fc"].item()
        rad["aMax_rad"] = rec["aMax_rad"].item()
        rad["da_0_rad"] = rec["da_0_rad"].item()
        rad["da_max_rad"] = rec["da_max_rad"].item()
        rad["TxPos"] = rec["TxPos"]
        rad["RxPos"] = rec["RxPos"]
        rad["nChi_pkt"] = rec["nChi_pkt"].item()
        rad["nSam_pkt"] = rec["nSam_pkt"].item()
        rad["nSam_rec"] = rec["nSam_rec"].item()
        rad["nPkt_seq"] = rec["nPkt_seq"].item()
        rad["nPkt_rec"] = rec["nPkt_rec"].item()

        rad["nIni"] = self.nIni
        rad["rPad"] = nfl(self.rPad if self.rPad > rad["N"] else rad["N"])
        # rad["rPad"] = self.rPad

        logging.debug(f'{rad["rMax"]=}')
        logging.debug(f'{rad["rPad"]=}')
        rad["df_fft"] = rad["fs"] / rad["rPad"]
        logging.debug(f'{rad["df"]=}')
        logging.debug(f'{rad["df_fft"]=}')

        rad["dr_fft"] = constants.c * rad["tSam"] / (2 * rad["fbw"]) * rad["df_fft"]
        logging.debug(f'{rad["dr"]=}')
        logging.debug(f'{rad["dr_fft"]=}')
        rad["rGrid"] = np.arange(rad["rPad"] // 2 + 1) * rad["dr_fft"]
        logging.debug(f'{rad["rGrid"]=}')
        rad["rMinIdx"] = np.argmin(abs(rad["rGrid"] - self.rMin))
        logging.debug(f'{rad["rMinIdx"]=}')
        rad["rMaxIdx"] = np.argmin(abs(rad["rGrid"] - self.rMax))
        logging.debug(f'{rad["rMaxIdx"]=}')

        # rad["rMinIdx"] = int(self.rMin / rad["dr_fft"] + 0.5) - 1
        # rad["rMaxIdx"] = (
        # int(self.rMax / rad["dr_fft"] + 0.5) + 1
        # if self.rMax < rad["rMax"]
        # else len(rad["rGrid"]) - 1
        # )
        rad["rGridEx"] = rad["rGrid"][rad["rMinIdx"] : rad["rMaxIdx"]]
        logging.debug(f'{rad["rGridEx"]=}')
        rad["rWin"] = np.hanning(rad["N"] - self.nIni)
        rad["rSca_win"] = sum(rad["rWin"])

        # velocity parameter
        rad["dfD"] = rad["fD_max"] / rad["nChi_seq"]
        rad["vPad"] = nfl(
            self.vPad
            if self.vPad > rad["nChi_seq"] // rad["nTdm"]
            else rad["nChi_seq"] // rad["nTdm"]
        )
        rad["dfD_fft"] = rad["fD_max"] / rad["vPad"]
        rad["dv"] = constants.c / rad["f0"] * rad["dfD"] * rad["nTdm"]
        # nTdm is neglected since the vel discretization is calculated
        # via the vel-FFT bins
        rad["dv_fft"] = constants.c / rad["f0"] * rad["dfD_fft"]
        rad["vGrid"] = np.arange(-rad["vPad"] // 2, rad["vPad"] // 2) * rad["dv_fft"]
        rad["vMinIdx"] = np.argmin(abs(rad["vGrid"] - self.vMin))
        rad["vMaxIdx"] = np.argmin(abs(rad["vGrid"] - self.vMax))
        assert (
            self.vMax <= rec["vMax"]
        ), "desired velocity exceeds sensor parameterization"
        rad["vGridEx"] = rad["vGrid"][rad["vMinIdx"] : rad["vMaxIdx"]]
        rad["vWin"] = np.hanning(rad["nChi_seq"] // rad["nTdm"])
        rad["vSca_win"] = sum(rad["vWin"])

        # azimuth parameter
        rad["aPad"] = nfl(
            self.aPad
            if self.aPad > rad["nChn_rec"] * rad["nTdm"]
            else rad["nChn_rec"] * rad["nTdm"]
        )
        rad["antIdx"] = [
            chn  # discard only the overlapping antenna element, i.e. the 17th element at index 16
            for chn in range(0, rad["nChn_rec"] * rad["nTdm"])
            if chn != rec["nChn_rec"].item()
        ]
        rec["reCalData"] = np.asarray(
            [np.frombuffer(x) for x in rec["CalRe"]]
        ).flatten()
        rec["imCalData"] = np.asarray(
            [np.frombuffer(x) for x in rec["CalIm"]]
        ).flatten()
        rad["calData"] = (rec["reCalData"] + 1j * rec["imCalData"])[rad["antIdx"]]
        rad["aWin"] = np.hanning(len(rad["antIdx"]))
        rad["aSca_win"] = sum(rad["aWin"])

        # camera parameter
        try:
            # Get the varying camera name in RadServe
            for ds_name in file.keys():
                if not ds_name.startswith("Chn") and "Time" not in ds_name:
                    cam_name = ds_name
            cam["nRow_img"] = rec[cam_name + "_Rows"].item()
            cam["nCol_img"] = rec[cam_name + "_Cols"].item()
            cam["nPkt_img"] = rec[cam_name + "_Rate"].item()
            cam["nChn_img"] = rec[cam_name + "_Channels"].item()
            rec["sCam_rec"] = file[cam_name]
            cam["nImg_rec"] = rec["sCam_rec"].shape[0] // (
                cam["nCol_img"] * cam["nChn_img"]
            )
            if cam["nImg_rec"] != rad["nSeq_rec"]:
                rec["nSeq_rec"] = cam["nImg_rec"]
                rad["nSeq_rec"] = cam["nImg_rec"]
                self.nSeq_skipped += 1
                # print(f"Camera samples lost in recording of file")
        except Exception as e:
            logging.error(e, "No camera found")
            self.cam = False
        rec["rad"] = rad
        rec["cam"] = cam
        # print(rec)
        return rec


class RangeProfiler(D.ProxyDataFlow):
    def __init__(self, ds):
        super().__init__(ds)

    def __iter__(self):
        for dp in self.ds:

            # print(dp['rad']['data'].shape)

            # sIF_seq = dp["rad"]["data"] - np.mean(
                # dp["rad"]["data"], axis=0, dtype=np.float32, keepdims=True
            # )

            sIF_seq = dp["rad"]["data"]

            sIF_seq = sIF_seq[:, dp["rad"]["nIni"] :]

            # range FFT (# FFTs = nChi_seq / nTdm * nChn_rec * nTdm)
            sRP_seq = rfft(
                sIF_seq * dp["rad"]["rWin"][:, np.newaxis],
                n=dp["rad"]["rPad"],
                axis=1,
            )

            # cutting out region of interest and compensating the gain
            sRP_seq = (
                sRP_seq[:, dp["rad"]["rMinIdx"] : dp["rad"]["rMaxIdx"]]
                * dp["rad"]["fu_sca"]
                / dp["rad"]["rSca_win"]
            )
            dp["rad"]["data"] = sRP_seq
            yield dp


class RangeDoppler(D.ProxyDataFlow):
    def __init__(self, ds):
        super().__init__(ds)

    def __iter__(self):
        for dp in self.ds:
            # Doppler FFT (# FFTs = nR_cells * nChn_rec * nTdm)
            sRD_seq = fft(
                dp["rad"]["data"] * dp["rad"]["vWin"][:, np.newaxis, np.newaxis],
                n=dp["rad"]["vPad"],
                axis=0,
            )

            # Shifting the zero frequency bin to the center and scaling
            sRD_seq = (
                fftshift(sRD_seq, axes=0)[dp["rad"]["vMinIdx"] : dp["rad"]["vMaxIdx"]]
                / dp["rad"]["vSca_win"]
            )
            dp["rad"]["data"] = sRD_seq.astype(np.complex64)
            yield dp


class Beamformer(D.ProxyDataFlow):
    def __init__(self, ds):
        super().__init__(ds)

    def __iter__(self):
        for dp in self.ds:
            # Angle FFT
            # sRDA_seq = (
            #     fft(
            #         dp["rad"]["data"],
            #         * dp["rad"]["aWin"],
            #         n=dp["rad"]["aPad"],
            #         axis=2,
            #     )
            #     / dp["rad"]["aSca_win"]
            # )
            sRDA_seq = (
                fft(
                    dp["rad"]["data"][..., dp["rad"]["antIdx"]]
                    * dp["rad"]["calData"]
                    * dp["rad"]["aWin"],
                    n=dp["rad"]["aPad"],
                    axis=2,
                )
                / dp["rad"]["aSca_win"]
            )
            sRDA_seq = fftshift(sRDA_seq, axes=2)

            dp["rad"]["data"] = sRDA_seq.astype(np.complex64)
            yield dp


class IF_signal(D.ProxyDataFlow):
    """Vanilla IF-signal processing
    Only does the following elementary steps
    - mean substraction across chirps
    - discarding first nIni samples of each chirp
    - multiply fu_Sca coefficient elementwise
    - calibrate the respective antenna data
    """

    def __init__(self, ds):
        super().__init__(ds)

    def __iter__(self):
        for dp in self.ds:
            # subtract mean across chirps of one sequence i.e. remove static objects
            sIF_seq = dp["rad"]["data"] - np.mean(
                dp["rad"]["data"], axis=0, dtype=np.float32, keepdims=True
            )

            dp["rad"]["data"] = sIF_seq
            # obj = sIF_seq[..., 0]
            # print('Sample shape:', obj.shape)
            yield dp


class Compressor(D.ProxyDataFlow):
    def __init__(self, ds):
        super(Decompressor, self).__init__(ds)

    def __iter__(self):
        for pt in self.ds:
            # dump data to byte array
            data = D.utils.serialize.dumps(pt)
            # compress binary data with blosc
            data = numcodecs.blosc.compress(data, "blosclz".encode("ascii"), 3)
            yield data


class Decompressor(D.ProxyDataFlow):
    def __init__(self, ds):
        super(Decompressor, self).__init__(ds)

    def __iter__(self):
        for pt in self.ds:
            data = numcodecs.blosc.decompress(pt)
            data = D.utils.serialize.loads(data)
            yield data


# convert numpy arrays to torch tensors
class Torching(D.ProxyDataFlow):
    def __init__(self, ds):
        super().__init__(ds)

    def __iter__(self):
        for dp in self.ds:
            yield {k: torch.as_tensor(v).float() for k, v in dp.items()}


class Transforming(D.ProxyDataFlow):
    # def __init__(self, ds, transform=None):
    def __init__(self, ds, trafos):
        super().__init__(ds)
        # print(trafos)
        self.trafos = {mod: [eval(t) for t in ts or []] for mod, ts in trafos.items()}

    def __iter__(self):
        for dp in self.ds:
            for mod, trafos in self.trafos.items():
                for trafo in trafos:
                    if isinstance(trafo, T.transforms.RandomTransforms):
                        dp[mod] = torch.stack([trafo(sample) for sample in dp[mod]])
                    else:
                        dp[mod] = trafo(dp[mod])
            yield dp

"""
def __iter__(self):
        for dp in self.ds:
            for mod, trafos in self.trafos.items():
                for trafo in trafos:
                    for k in dp.keys():
                        if k.endswith(mod):
                            l = list()
                            for i in range(len(dp[k])):

for sample_idx in len(dp[k]): # batch dim

print(sample.shape)

print(

                                l.append(
# # # dp[k][i].shape)trafo(dp[k][i]))
                            dp[k] = torch.stack(l)
                            # dp[k] = trafo(dp[k])
            yield dp
"""

class LogMagnitudeAndPhase(D.ProxyDataFlow):
    """ Takes the absolute value of the processed radar data """
    def __init__(self, ds):
        super().__init__(ds)

    def __iter__(self):
        for dp in self.ds:
            if isinstance(dp["rad"], dict):  # only while saving if included in df
                dp["rad"]["data"] = abs(dp["rad"]["data"])
            else:
                for mod in dp.keys():
                    if mod.endswith('rad'): # allows to process cam_rad as well
                        if dp[mod].shape[-1] > 32:  # not a cube
                            mag = 20 * np.log10(np.abs(dp[mod]) ** 2)
                            # mag = np.abs(dp[mod]) ** 2
                            # mag = np.log10(np.abs(dp[mod]))
                            # mag = np.abs(dp[mod])

                            # ang = np.angle(dp["rad"])
                            # dp["rad"] = np.concatenate(
                            # [mag, ang], -4 if mag.shape[-1] <= 32 else -3
                            # )
                            dp[mod] = mag
                        else:  # cube already has complex magnitude and neglection of phase
                            # assert not np.any(dp['rad'] >= 0)
                            dp[mod] = 20 * np.log10(dp[mod] ** 2)
                """
                if dp['rad'].shape[-1] > 32:  # not a cube
                    mag = 20 * np.log10(np.abs(dp[mod]))
                    # ang = np.angle(dp["rad"])
                    # dp["rad"] = np.concatenate(
                    # [mag, ang], -4 if mag.shape[-1] <= 32 else -3
                    # )
                    dp['rad'] = mag
                else:  # cube already has complex magnitude and neglection of phase
                    # assert not np.any(dp['rad'] >= -10)
                    dp['rad'] = 20 * np.log10(dp['rad'])
                """
            yield dp


class Slicer(D.ProxyDataFlow):
    """
    Slices the radar cube through the respective center line into a 2D map
    """

    def __init__(self, ds, slice_along, slice_idx=None):
        super().__init__(ds)
        self._slice_along = slice_along
        self._slice_idx = slice_idx

    def __iter__(self):
        for dp in self.ds:
            rad_cube = dp["rad"]["data"]

            if self._slice_idx is None:  # if no idx specified, take center slice
                Doppler_idx, range_idx, azimuth_idx = rad_cube.shape // 2
            else:
                Doppler_idx = range_idx = azimuth_idx = self._slice_idx

            if self._slice_along == "Azimuth":
                dp["rad"]["data"] = rad_cube[..., azimuth_idx]
            elif self._slice_along == "Doppler":
                dp["rad"]["data"] = rad_cube[Doppler_idx]
            elif self._slice_along == "Range":
                # self._slice_idx = rad_cube.shape[1] // 2 if self._slice_idx is None
                dp["rad"]["data"] = rad_cube[:, range_idx]

            yield dp


# buffer elements so that data loading is quicker
class BufferQueue(D.ProxyDataFlow):
    class WorkerThread(threading.Thread):
        def __init__(self, ds, queue, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ds = ds
            self.daemon = True
            self.queue = queue

        def run(self):
            try:
                self.reinitialize_dataflow()
                while True:
                    pt = next(self._itr)
                    self.queue.put(pt)
            except StopIteration:
                pass
            except D.DataFlowTerminated:
                print("[WorkerThread] DataFlow has terminated.")
            except Exception as e:
                print("[WorkerThread] An exception occurred:")
                print(e)
                raise e
            finally:
                self.queue.put(None)

        def reinitialize_dataflow(self):
            self._itr = self.ds.__iter__()

    def __init__(self, ds, buffer_size=6):
        super().__init__(ds)
        self.ds = ds if isinstance(ds, list) else [ds]
        assert len(self.ds) > 0
        self.buffer_size = buffer_size
        self.queue = queue.Queue(buffer_size)

    def reset_state(self):
        for ds in self.ds:
            ds.reset_state()

    # def reset_state(self, **kw):
        # for ds in self.ds:
            # ds.reset_state(**kw)

    def __len__(self):
        return sum(map(len, self.ds))

    def __iter__(self):
        self.workers = []
        for ds in self.ds:
            worker = BufferQueue.WorkerThread(ds, self.queue)
            worker.start()
            self.workers.append(worker)
        finished_workers = 0
        while True:
            # start = timer()
            try:
                pt = self.queue.get(timeout=60.0)
            except queue.Empty:
                print("Queue got stuck! Dropping elements.")
                pt = None
            # end = timer()
            # print("Waited for", end-start, "queue size", self.worker.queue.qsize())
            if pt is None:
                finished_workers += 1
                if finished_workers == len(self.workers):
                    break
            yield pt


class DataFlowReentrantGuard:
    """
    A tool to enforce non-reentrancy.
    Mostly used on DataFlow whose :meth:`get_data` is stateful,
    so that multiple instances of the iterator cannot co-exist.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._succ = self._lock.acquire(False)
        if not self._succ:
            raise threading.ThreadError("This DataFlow is not reentrant!")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False


class InrasLMDBSink(D.ProxyDataFlow):
    def __init__(self, ds, lmdb_path, rel_path, desc, write_frequency):
        super().__init__(ds)
        self._write_frequency = write_frequency
        self._num_samples = len(self.ds)

        path, stem = os.path.split(rel_path)
        rec_date = os.path.splitext(stem)[0][-14:]
        rec_date = f"/{rec_date[:4]}-{rec_date[4:6]}-{rec_date[6:8]}_{rec_date[8:10]}:{rec_date[10:12]}:{rec_date[12:14]}____"

        path, stem = os.path.split(lmdb_path)
        self._lmdb_path = (
            path + rec_date + stem + "____" + str(self._num_samples) + "_samples.lmdb"
        )
        self._open_lmdb()
        # put data into lmdb, and doubling the size if full.
        # Ref: https://github.com/NVIDIA/DIGITS/pull/209/files
        def put_or_grow(txn, key, value):
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
            txn = put_or_grow(txn, key, value)
            return txn

        with get_tqdm(total=self._size) as pbar:
            # LMDB transaction is not exception-safe!
            # although it has a context manager interface
            txn = self._db.begin(write=True)
            for idx, dp in enumerate(self.ds):
                # print(f'Sample size in MB: {sys.getsizeof(dp["rad"]["data"])/ (1024 ** 2):.5f}')
                obj = dp['cam']['data']
                print('Sample shape:', obj.shape)
                print('Sample type:', obj.dtype)
                obj = dp['rad']['data']
                print('Sample shape:', obj.shape)
                print('Sample type:', obj.dtype)
                # print(f'Sample size in MB: {len(pickle.dumps(obj))/1024/1024:.5f}')

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

    def _open_lmdb(self):
        assert isinstance(self.ds, D.DataFlow), type(self.ds)
        isdir = os.path.isdir(self._lmdb_path)
        if isdir:
            assert not os.path.isfile(
                os.path.join(self._lmdb_path, "data.mdb")
            ), "LMDB file exists!"
        else:
            assert not os.path.isfile(
                self._lmdb_path
            ), f"LMDB file {self._lmdb_path} exists!"
        # It's OK to use super large map_size on Linux, but not on other platforms
        # See: https://github.com/NVIDIA/DIGITS/issues/206
        map_size = 1099511627776 * 2 if platform.system() == "Linux" else 128 * 10 ** 6

        self._db = lmdb.open(
            self._lmdb_path,
            subdir=isdir,
            map_size=map_size,
            readonly=False,
            meminit=False,
            map_async=True,
        )  # need sync() at the end
        self._size = self._reset_df_and_get_size(self.ds)

    def _reset_df_and_get_size(self, df):
        df.reset_state()
        try:
            sz = len(df)
        except NotImplementedError:
            sz = 0
        return sz


class InrasLMDBSource(D.RNGDataFlow):
    """
    Read a LMDB database and produce (k,v) raw bytes pairs which are decoded
    """

    def __init__(self, src_dir, recursive, shuffle, tCtx, tDis_ctx, both_mod, **kwargs):
        # self._lmdb_path = src_dir
        self._lmdb_files = scrape_files(src_dir, "lmdb", recursive)
        self.both_mod = both_mod
        self.fps = 10
        self.nSeq_ctx = 0
        self.nSeq_eff = 0
        self.nStr_ctx = 1
        assert bool(tCtx) == bool(tDis_ctx), "set both values"

        if tCtx:  # if not 0 or None
            self.nSeq_ctx = int(self.fps * tCtx)
            self.nSeq_eff = round(tCtx / tDis_ctx)
            self.nStr_ctx = self.nSeq_ctx // self.nSeq_eff
            assert self.nSeq_ctx % self.nSeq_eff == 0
            assert tDis_ctx <= tCtx / 2

        self._open_lmdbs()

        lFiles = [txn.stat()["entries"] // 2 - self.nSeq_ctx for txn in self._txn]
        assert all(l > 0 for l in lFiles), "{tCtx} exceeds at least one rec length"
        self.lFil_acc = list(itertools.accumulate(lFiles))
        self.lFil_acc.insert(0, 0)
        self._size = self.lFil_acc.pop(-1)

        # self._size = sum(
        # txn.stat()["entries"] // 2 - self.nSeq_ctx for txn in self._txn
        # )
        logging.info(
            f"For a temporal context of {tCtx} seconds with {self.nSeq_eff} samples {tDis_ctx} seconds apart there are {self._size} valid samples at {src_dir} in files:"
        )
        for lmdb_file in self._lmdb_files:
            logging.info(f"\t-> {Path(lmdb_file).stem}{Path(lmdb_file).suffix}")
        logging.info(f"in a total of {len(self._lmdb_files)} files")
        # Clean them up after finding the list of keys, since we don't want to fork them
        # self._close_lmdb()

        self.all_idxs = list(range(self._size))
        # print(self.all_idxs)
        # self.all_idxs = list(range(4))

        self.idx_association = dict()

        for idx in self.all_idxs:
            for i, k in enumerate(self.lFil_acc[::-1]):
                if idx >= k:  # 5 < 20
                    fIdx = idx - k
                    self.idx_association[idx] = (len(self.lFil_acc) - 1 - i, fIdx)
                    break
        # print(self.idx_association)

        self._keys = {
            "r_k": self.all_idxs,
            "c_k": self.all_idxs,
            "lab": [1] * len(self.all_idxs),
        }

        # if shuffle-list is specified in config.yaml, get iterator on that list
        self._shuffle = iter(shuffle) if shuffle else None

        self.unique_combinations = set()
        self.possible_unique_combinations = 0

    def _open_lmdbs(self):
        self._lmdb = [
            lmdb.open(
                lmdb_file,
                subdir=os.path.isdir(lmdb_file),
                readonly=True,
                lock=False,
                readahead=True,
                map_size=1024 ^ 3 * 2,
                # map_size=1099511627776 * 2,
                max_readers=100,
                # writemap=True
            )
            for lmdb_file in self._lmdb_files
        ]

        self._txn = [lmdb.begin() for lmdb in self._lmdb]

    def _close_lmdb(self):
        for lmdb, txn in zip(self._lmdb, self._txn):
            lmdb.close()
            del lmdb, txn

    def reset_state(self):
        self._guard = DataFlowReentrantGuard()
        super().reset_state()
        self._open_lmdbs()  # open the LMDB in the worker process

    def __len__(self):
        return self._size

    def __iter__(self):
        self.shuffle()

        worker_info = torch.utils.data.get_worker_info()
        # print(worker_info)
        if worker_info is not None:
            per_worker = int(math.ceil((self.__len__()) / float(worker_info.num_workers)))
            # print(per_worker)
            worker_id = worker_info.id
            # logger.info('Worker ID:', worker_id)
            iter_start = 0 + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.__len__())
        else:
            iter_start = 0
            iter_end = self.__len__()

        # print(self._keys['r_k'][iter_start:iter_end])
        # print(self._keys['c_k'][iter_start:iter_end])
        # print(self._keys['lab'][iter_start:iter_end])

        with self._guard:
            for r_k, c_k, lab in zip(
                    self._keys["r_k"][iter_start:iter_end],
                    self._keys["c_k"][iter_start:iter_end],
                    self._keys["lab"][iter_start:iter_end]
            ):

                # r_f: radar file
                # r_k: radar key
                # r_s: radar stride

                r_f, r_k = self.idx_association[r_k]
                c_f, c_k = self.idx_association[c_k]

                rad, cam, rad_cam, cam_rad = [], [], [], []
                for k in range(0, self.nSeq_ctx + 1, self.nStr_ctx):
                    r_s = self._txn[r_f].get(f"{(r_k+k):0>8d}r".encode("ascii"))
                    r_v = pickle.loads(r_s)["data"]
                    c_s = self._txn[c_f].get(f"{(c_k+k):0>8d}c".encode("ascii"))
                    c_v = pickle.loads(c_s)["data"]
                    rad.append(r_v)
                    cam.append(c_v)

                    if self.both_mod:
                        cr_s = self._txn[c_f].get(f"{(c_k+k):08d}r".encode("ascii"))
                        cr_v = pickle.loads(cr_s)["data"]
                        rc_s = self._txn[r_f].get(f"{(r_k+k):08d}c".encode("ascii"))
                        rc_v = pickle.loads(rc_s)["data"]
                        rad_cam.append(rc_v)
                        cam_rad.append(cr_v)

                rad = np.stack(rad)
                cam = np.stack(cam)

                if self.both_mod:
                    rad = np.stack((rad, cam_rad))
                    cam = np.stack((cam, rad_cam))

                cam = np.squeeze(np.rollaxis(cam, -1, -3))

                # rad = np.squeeze(np.rollaxis(rad, -1, -3))
                rad = np.expand_dims(np.squeeze(rad), -4 if rad.shape[-1] <= 32 else -3)

                if self.both_mod:
                    yield {"rad": rad[0], "cam": cam[0],
                           'cam_rad': rad[1], 'rad_cam': cam[1], "lab": lab}
                else:
                    yield {"rad": rad, "cam": cam, "lab": lab}


    def shuffle(self):
        if self._shuffle is not None:
            try:
                shuffler = next(self._shuffle)
                if shuffler in ["inter_modal", "inter_dataset", "all_non_aligned"]:
                    self._shuffler = eval(f"self.{shuffler}")
                elif shuffler == "None":
                    self._shuffler = lambda keys: keys
                else:
                    raise NotImplementedError("SHUFFLE MODE NOT SUPPORTED")
            except StopIteration:
                self._keys = self._shuffler(self._keys)
            else:
                self._keys = self._shuffler(self._keys)

            posi = sum(i for i in self._keys["lab"])
            logging.debug(f"Positive samples ratio/batch {posi/len(self._keys['lab']):.3f}")
            com_idxs = list(zip(self._keys["r_k"], self._keys["c_k"], self._keys["lab"]))
            self.possible_unique_combinations += len(com_idxs)
            for combination in com_idxs:
                self.unique_combinations.add(combination)
            # com_idxs.sort()
            logging.debug(f'Unique samples {len(self.unique_combinations)} / {self.possible_unique_combinations}')
            logging.debug(f'Unique samples ratio {len(self.unique_combinations)/float(self.possible_unique_combinations):.3f}')
            # self._keys["r_k"], self._keys["c_k"], self._keys["lab"] = zip(*com_idxs)
            # print('unqiue rad keys', len(set(self._keys['r_k'])))
            # print('unqiue cma keys', len(set(self._keys['c_k'])))
            # print(self._keys)

        # exit()
    def inter_dataset(self, keys):
        # print("INTER DATASET Shuffle")
        # keys = {
        # "r_k": self.all_idxs,
        # "c_k": self.all_idxs,
        # "lab": [1] * len(self.all_idxs),
        # }
        com_idxs = list(zip(keys["r_k"], keys["c_k"], keys["lab"]))
        # print('Before shuffle', com_idxs)
        self.rng.shuffle(com_idxs)
        # np.random.shuffle(com_idxs)
        # print('AFTER shuffle', com_idxs)
        keys["r_k"], keys["c_k"], keys["lab"] = zip(*com_idxs)
        return keys

    def inter_modal(self, keys):
        # print('INTER MODAL Shuffle')
        keys = {
            "r_k": self.all_idxs,
            "c_k": self.all_idxs,
            "lab": [1] * len(self.all_idxs),
        }
        com_keys = np.array(list(zip(keys["r_k"], keys["c_k"], keys["lab"])))
        com_idxs = np.arange(len(com_keys))
        pos_idxs = self.rng.choice(com_idxs, len(com_idxs) // 2, False)
        # print('pos_idxs', pos_idxs)
        pos_keys = com_keys[pos_idxs]
        # print('pos_keys', pos_keys)
        neg_rad_idxs = [idx for idx in com_idxs if idx not in pos_idxs]
        # print('neg_rad_idxs', neg_rad_idxs)
        # neg_cam_idxs = self.rng.choice(com_idxs, len(com_idxs) // 2, False)
        neg_cam_idxs = self.rng.choice(com_idxs, len(neg_rad_idxs), False)
        # print('neg_cam_idxs', neg_cam_idxs)
        neg_keys = [
            (r_k, c_k, 0) if r_k != c_k else (r_k, c_k, 1)
            for r_k, c_k in zip(neg_rad_idxs, neg_cam_idxs)
        ]
        key = [*pos_keys, *neg_keys]
        self.rng.shuffle(key)
        keys["r_k"], keys["c_k"], keys["lab"] = zip(*key)
        return keys

    def all_non_aligned(self, keys):
        keys = {
            "r_k": self.all_idxs,
            "c_k": self.all_idxs,
            "lab": [1] * len(self.all_idxs),
        }
        com_keys = np.array(list(zip(keys["r_k"], keys["c_k"], keys["lab"])))
        neg_rad_idxs = np.arange(len(com_keys))
        # print(neg_rad_idxs)
        self.rng.shuffle(neg_rad_idxs)
        # print(neg_rad_idxs)
        neg_cam_idxs = self.rng.choice(neg_rad_idxs, len(neg_rad_idxs), False)
        # print('neg_cam_idxs', neg_cam_idxs)
        key = [
            (r_k, c_k, 0) for r_k, c_k in zip(neg_rad_idxs, neg_cam_idxs)
            if r_k != c_k
        ]
        self.rng.shuffle(key)
        keys["r_k"], keys["c_k"], keys["lab"] = zip(*key)
        # print('positive idxs', len(keys['lab']))
        return keys

class InrasData(torch.utils.data.IterableDataset):
    class _Type(Enum):
        Loader = "loader"
        Writer = "writer"

    def __init__(self, _type, params):
        for k, v in params.items():
            setattr(self, k, v)

        if _type == __class__._Type.Writer:
            df = InrasHDF5Source(self.src_dir, self.cutout, self.recursive)
            # df = IF_signal(df)

            df = RangeProfiler(df)
            df = RangeDoppler(df)

            df = Slicer(df, self.slice_along, self.slice_idx)

            # df = Beamformer(df)
            # df = LogMagnitudeAndPhase(df)

            InrasLMDBSink(df, self.dst_dir, self.src_dir, self.desc, self.write_frequency)

        elif _type == __class__._Type.Loader:
            df = InrasLMDBSource(**params)
            self._size = len(df)
            # only use Slicer for cube data
            # df = Slicer(df, self.slice_along, self.slice_idx)

            df = LogMagnitudeAndPhase(df)
            df = D.BatchData(df, self.batch_size, self.remainder)
            df = Torching(df)
            df = Transforming(df, self.trafos if hasattr(self, 'trafos') else {})
            df = BufferQueue(df)
            if self.diagnose:
                df = D.PrintData(df, num=1, max_depth=3, max_list=3)
                D.TestDataSpeed(df, size=self._size, warmup=0).start()
                exit()

        self.df = df

    def __iter__(self):
        # __iter__ itself is a generator function, because of the yield expression
        # so each call to __iter__ returns an independent anonymous generator objecpt
        # if not self.reset:
        self.df.reset_state()
        # self.reset = False
        yield from self.df
            # sudo hdparm -Tt /dev/sda check cached and uncached disk read bandwidth

    def __len__(self):
        return len(self.df)

    # def reset_state(self, **kw):
        # self.df.reset_state(**kw)
        # self.reset = True

    @property
    def size(self):
        return self._size

    @classmethod
    def loader(cls, **cfg):
        return cls(cls._Type.Loader, cfg)

    @classmethod
    def writer(cls, **cfg):
        return cls(cls._Type.Writer, cfg)
