# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |s
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np
import math
# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import onnx

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

import sklearn.metrics
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('AGG')

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

from torch.optim.lr_scheduler import _LRScheduler
import utils

exc = getattr(builtins, "IOError", "FileNotFoundError")


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr


### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln):
        emb_l = nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(n, m, self.qr_collisions,
                                    operation=self.qr_operation, mode="sum", sparse=True)
            elif self.md_flag:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)

            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)

            emb_l.append(EE)

        return emb_l

    def __init__(
            self,
            m_spa=None,
            ln_emb=None,
            ln_bot=None,
            ln_top=None,
            arch_interaction_op=None,
            arch_interaction_itself=False,
            sigmoid_bot=-1,
            sigmoid_top=-1,
            sync_dense_params=True,
            loss_threshold=0.0,
            ndevices=-1,
            qr_flag=False,
            qr_operation="mult",
            qr_collisions=0,
            qr_threshold=200,
            md_flag=False,
            md_threshold=200,
    ):
        super(DLRM_Net, self).__init__()

        if (
                (m_spa is not None)
                and (ln_emb is not None)
                and (ln_bot is not None)
                and (ln_top is not None)
                and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold
            # create operators
            if ndevices <= 1:
                self.emb_l = self.create_emb(m_spa, ln_emb)
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        # for k, sparse_index_group_batch in enumerate(lS_i):
        # what is lS_o, lS_i
        for k in range(len(lS_i)):
            sparse_index_group_batch = lS_i[k]
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)

            ly.append(V)

        # print(ly)
        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        if self.ndevices <= 1:
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            return self.parallel_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                emb.to(d)
                t_list.append(emb.to(d))
            self.emb_l = nn.ModuleList(t_list)
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


def dash_separated_ints(value):
    vals = value.split('-')
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value)

    return value


def dash_separated_floats(value):
    vals = value.split('-')
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value)

    return value


if __name__ == "__main__":
    ### import packages ###
    import sys
    import argparse
    import os
    import logging
    import pickle
    import re
    from collections import OrderedDict
    from torch.autograd import Variable
    import collections
    import random
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2")

    parser.add_argument("--arch-sparse-feature-size", type=int, default=16)
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="13-128-16")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="128-1")

    parser.add_argument(
        "--arch-interaction-op", type=str, choices=['dot', 'cat'], default="dot")
    parser.add_argument(
        "--arch-interaction-itself", action="store_true", default=False)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="bce")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0")  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=True)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="dataset"
    )  # synthetic or dataset
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--raw-data-file", type=str, default="./input/train.txt")
    parser.add_argument("--processed-data-file", type=str, default="./input/kaggleAdDisplayChallenge_processed.npz")  #
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--dataset-multiprocessing", action="store_true", default=False,
                        help="The Kaggle dataset can be multiprocessed in an environment \
                           with more than 7 CPU cores and more than 20 GB of memory. \n \
                           The Terabyte dataset can be multiprocessed in an environment \
                           with more than 24 CPU cores and at least 1 TB of memory.")
    # training
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=True)
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1024)
    parser.add_argument("--test-freq", type=int, default=80000)
    parser.add_argument("--test-mini-batch-size", type=int, default=16384)
    parser.add_argument("--test-num-workers", type=int, default=-16)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    ## added
    parser.add_argument("--gpu-id", type=str, default='1')
    parser.add_argument("--initialization", type=str, default="zero")  # random or zero
    parser.add_argument("--grow-embedding", action='store_true', default=False)
    parser.add_argument("--masking-ratio", type=dash_separated_floats, default='0')
    parser.add_argument("--masking-delay", type=dash_separated_floats, default='0')

    parser.add_argument("--debuglog", type=str, default='DSD')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    log_format = '%(asctime)s   %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M%p')

    mask_ratio = np.fromstring(args.masking_ratio, dtype=float, sep="-")
    mask_delay = np.fromstring(args.masking_delay, dtype=float, sep="-")
    if len(mask_ratio) != len(mask_delay):
        sys.exit("ERROR: len(mask_ratio) != len(mask_delay)")

    if not os.path.exists('./log'):
        os.mkdir('./log')
    if len(mask_delay) == 1 :
        fh = logging.FileHandler(os.path.join(
            './log/log_dlrm_s_pytorch_bot{}_top{}_Baseline.txt'.format(args.arch_mlp_bot, args.arch_mlp_top)))
    else:
        fh = logging.FileHandler(os.path.join(
            './log/log_dlrm_s_pytorch_bot{}_top{}_maskingRatio{}_maskingDelay{}_GrowEmb{}_{}.txt'.format(
                args.arch_mlp_bot, args.arch_mlp_top, args.masking_ratio, args.masking_delay, args.grow_embedding, args.debuglog)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("******************************************************")
    logging.info("                  dlrm_s_pytorch.py                   ")
    logging.info("args = %s", args)
    if args.mlperf_logging:
        print('command line args: ', json.dumps(vars(args)))


    def prepare_dataset():
        if args.data_generation == "dataset":
            train_data, train_ld, test_data, test_ld = \
                dp.make_criteo_data_and_loaders(args)
            nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
            nbatches_test = len(test_ld)
        else:
            raise ValueError('Please specify dataset')
        #     # input and target at random
        #     ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        #     m_den = ln_bot[0]
        #     train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        #     nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        trainset = (train_data, train_ld, nbatches)
        testset = (test_data, test_ld, nbatches_test)
        return trainset, testset


    # ===============
    def instance_dimension(size_scale, trainset):
        if size_scale > 1.0:
            size_scale = 1.0
        ### prepare training data ###
        ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
        if args.grow_embedding:
            ln_bot = int(ln_bot * size_scale)
        else:
            ln_bot[0:-1] = list(map(int, ln_bot[0:-1] * size_scale))
        # print('ln_bot', ln_bot)
        # input data
        train_data, train_ld, nbatches = trainset

        ln_emb = train_data.counts
        m_den = train_data.m_den
        ln_bot[0] = m_den
        num_fea = ln_emb.size + 1  # num sparse + num dense features
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        ### parse command line arguments ###
        if args.grow_embedding:
            m_spa = int(args.arch_sparse_feature_size * size_scale)
        else:
            m_spa = int(args.arch_sparse_feature_size)
        m_den_out = ln_bot[ln_bot.size - 1]
        # print('m_den_out', m_den_out)
        if args.arch_interaction_op == "dot":
            # approach 1: all
            # num_int = num_fea * num_fea + m_den_out
            # approach 2: unique
            if args.arch_interaction_itself:
                num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
            else:
                num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
        elif args.arch_interaction_op == "cat":
            num_int = num_fea * m_den_out
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + args.arch_interaction_op
                + " is not supported"
            )
        arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
        ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
        ln_top[1:-1] = list(map(int, ln_top[1:-1] * size_scale))

        # sanity check: feature sizes and mlp dimensions must match
        if m_den != ln_bot[0]:
            sys.exit(
                "ERROR: arch-dense-feature-size "
                + str(m_den)
                + " does not match first dim of bottom mlp "
                + str(ln_bot[0])
            )
        if args.qr_flag:
            if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
                sys.exit(
                    "ERROR: 2 arch-sparse-feature-size "
                    + str(2 * m_spa)
                    + " does not match last dim of bottom mlp "
                    + str(m_den_out)
                    + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
                )
            if args.qr_operation != "concat" and m_spa != m_den_out:
                sys.exit(
                    "ERROR: arch-sparse-feature-size "
                    + str(m_spa)
                    + " does not match last dim of bottom mlp "
                    + str(m_den_out)
                )
        else:
            if m_spa != m_den_out:
                sys.exit(
                    "ERROR: arch-sparse-feature-size "
                    + str(m_spa)
                    + " does not match last dim of bottom mlp "
                    + str(m_den_out)
                )
        if num_int != ln_top[0]:
            sys.exit(
                "ERROR: # of feature interactions "
                + str(num_int)
                + " does not match first dimension of top mlp "
                + str(ln_top[0])
            )

        # assign mixed dimensions if applicable
        if args.md_flag:
            m_spa = md_solver(
                torch.tensor(ln_emb),
                args.md_temperature,  # alpha
                d0=m_spa,
                round_dim=args.md_round_dims
            ).tolist()

        # test prints (model arch)
        if args.debug_mode:
            print("model arch:")
            print(
                "mlp top arch "
                + str(ln_top.size - 1)
                + " layers, with input to output dimensions:"
            )
            print(ln_top)
            print("# of interactions")
            print(num_int)
            print(
                "mlp bot arch "
                + str(ln_bot.size - 1)
                + " layers, with input to output dimensions:"
            )
            print(ln_bot)
            print("# of features (sparse and dense)")
            print(num_fea)
            print("dense feature size")
            print(m_den)
            print("sparse feature size")
            print(m_spa)
            print(
                "# of embeddings (= # of sparse features) "
                + str(ln_emb.size)
                + ", with dimensions "
                + str(m_spa)
                + "x:"
            )
            print(ln_emb)

            print("data (inputs and targets):")
            for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
                # early exit if nbatches was set by the user and has been exceeded
                if nbatches > 0 and j >= nbatches:
                    break

                print("mini-batch: %d" % j)
                print(X.detach().cpu().numpy())
                # transform offsets to lengths when printing
                print(
                    [
                        np.diff(
                            S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                        ).tolist()
                        for i, S_o in enumerate(lS_o)
                    ]
                )
                print([S_i.detach().cpu().tolist() for S_i in lS_i])
                print(T.detach().cpu().numpy())
        ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

        dimension_info = (m_spa, ln_emb, ln_bot, ln_top, num_fea, num_int)

        return dimension_info, ndevices
        # ===============


    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    ### structure
    def instance_dlrm(m_spa, ln_emb, ln_bot, ln_top, ndevices):
        dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            arch_interaction_op=args.arch_interaction_op,
            arch_interaction_itself=args.arch_interaction_itself,
            sigmoid_bot=-1,
            sigmoid_top=ln_top.size - 2,
            sync_dense_params=args.sync_dense_params,
            loss_threshold=args.loss_threshold,
            ndevices=ndevices,
            qr_flag=args.qr_flag,
            qr_operation=args.qr_operation,
            qr_collisions=args.qr_collisions,
            qr_threshold=args.qr_threshold,
            md_flag=args.md_flag,
            md_threshold=args.md_threshold,
        )
        # logging.info(dlrm.bot_l[0].weight)
        # logging.info(dlrm.bot_l[0].bias)
        # logging.info(dlrm.bot_l[0].weight.grad)

        # test prints
        if args.debug_mode:
            print("initial parameters (weights and bias):")
            for param in dlrm.parameters():
                print(param.detach().cpu().numpy())
            # print(dlrm)

        if use_gpu:
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            dlrm = dlrm.to(device)  # .cuda()
            if dlrm.ndevices > 1:
                dlrm.emb_l = dlrm.create_emb(m_spa, ln_emb)

        param = utils.count_parameters_in_MB(dlrm)
        param_FC = utils.count_parameters_in_FC(dlrm)
        if not os.path.isdir('./saved_model'):
            os.mkdir('./saved_model')
        #
        # file_name = "model_FCparam{:.1}K.pickle".format(param_FC)
        # path = os.path.join('./saved_model', file_name)
        # pickle.dump(dlrm.state_dict(), open(path, 'wb'))
        # logging.info('save_model to ' + path + '\n')

        # logging.info('dlrm: {}'.format(dlrm))
        logging.info(
            'FC param size = {:.2f}K, param size = {:.2f}M,  FLOP = {:.2f}K'.format(param_FC, param, param_FC * 2))
        logging.info('m_spa={}, ln_bot={}, ln_top={} \n'.format(m_spa, ln_bot, ln_top))

        if not args.inference_only:
            # specify the optimizer algorithm
            optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)
            lr_scheduler = LRPolicyScheduler(optimizer, args.lr_num_warmup_steps, args.lr_decay_start_step,
                                             args.lr_num_decay_steps)

        return dlrm, optimizer, lr_scheduler


    # ---------------------

    def save_trained_model(current_net, growth_id):
        if not os.path.isdir('./saved_model'):
            os.mkdir('./saved_model')

        file_name = "model_after_growth{}.pickle".format(growth_id)
        path = os.path.join('./saved_model', file_name)
        pickle.dump(current_net.state_dict(), open(path, 'wb'))
        logging.info('save_model to ' + path + '\n')


    def hook_fn_forward(module, input, output):
        global growth_id
        logging.info('growth_id', growth_id)
        logging.info('output', output, output.shape)
        output = output * growth_id / (growth_id + 1)
        logging.info('output', output, output.shape)


    def load_trained_model_stacking(to_net, growth_id):
        file_name = "model_after_growth{}.pickle".format(growth_id)
        path = os.path.join('./saved_model', file_name)
        logging.info('Reading and loading pre-trained weights............')
        old_param_dict = pickle.load(open(path, "rb"))
        new_param_dict = OrderedDict([(k, None) for k in to_net.state_dict().keys()])

        for layer_name, param_new in to_net.state_dict().items():
            param_old = old_param_dict[layer_name].type(torch.cuda.FloatTensor)
            std = param_old.std().item()
            if args.initialization == "random":
                logging.info('Random initialization')
                if re.search('emb', layer_name):
                    if args.grow_embedding:
                        param_new[:, 0:param_old.shape[1]] = Variable(param_old.clone(), requires_grad=True)
                        random_initialization = torch.empty(param_old.shape[0],
                                                            param_new.shape[1] - param_old.shape[1]).clone().normal_(0,
                                                                                                                     std).type(
                            torch.cuda.FloatTensor)
                        param_new[:, param_old.shape[1]:] = Variable(random_initialization, requires_grad=True)
                    else:
                        param_new = Variable(param_old.clone(), requires_grad=True)
                elif layer_name == 'bot_l.6.weight':
                    # logging.info('bot_l.6.weight', param_old, param_old.shape)
                    if args.grow_embedding:
                        param_new[0:param_old.shape[0], 0: param_old.shape[1]] = Variable(param_old.clone(),
                                                                                          requires_grad=True)
                        random_initialization = torch.empty(param_old.shape[0],
                                                            param_new.shape[1] - param_old.shape[1]).clone().normal_(0,
                                                                                                                     std).type(
                            torch.cuda.FloatTensor)
                        param_new[0:param_old.shape[0], param_old.shape[1]:] = Variable(random_initialization,
                                                                                        requires_grad=True)
                        random_initialization = torch.empty(param_new.shape[0] - param_old.shape[0],
                                                            param_new.shape[1]).clone().normal_(0, std).type(
                            torch.cuda.FloatTensor)
                        param_new[param_old.shape[0]:, :] = Variable(random_initialization, requires_grad=True)
                    else:
                        param_new[:, 0:param_old.shape[1]] = Variable(param_old.clone(), requires_grad=True)
                        random_initialization = torch.empty(param_old.shape[0],
                                                            param_new.shape[1] - param_old.shape[1]).clone().normal_(0,
                                                                                                                     std).type(
                            torch.cuda.FloatTensor)
                        param_new[:, param_old.shape[1]:] = Variable(random_initialization, requires_grad=True)
                    # logging.info('bot_l.6.weight', param_new, param_new.shape)


                elif layer_name == 'bot_l.0.weight':
                    param_new[0:param_old.shape[0], :] = Variable(param_old.clone(), requires_grad=True)
                    random_initialization = torch.empty(param_new.shape[0] - param_old.shape[0],
                                                        param_old.shape[1]).clone().normal_(0, std).type(
                        torch.cuda.FloatTensor)
                    param_new[param_old.shape[0]:, :] = Variable(random_initialization, requires_grad=True)
                elif layer_name == 'bot_l.0.bias':
                    param_new[0:param_old.shape[0]] = Variable(param_old.clone(), requires_grad=True)
                    random_initialization = torch.empty(param_new.shape[0] - param_old.shape[0]).clone().normal_(0,
                                                                                                                 std).type(
                        torch.cuda.FloatTensor)
                    param_new[param_old.shape[0]:] = Variable(random_initialization, requires_grad=True)
                elif layer_name == 'top_l.4.weight':
                    param_new[:, 0:param_old.shape[1]] = Variable(param_old.clone(), requires_grad=True)
                    random_initialization = torch.empty(param_old.shape[0],
                                                        param_new.shape[1] - param_old.shape[1]).clone().normal_(0,
                                                                                                                 std).type(
                        torch.cuda.FloatTensor)
                    param_new[:, param_old.shape[1]:] = Variable(random_initialization, requires_grad=True)
                elif layer_name == 'top_l.4.bias':
                    param_new = Variable(param_old.clone(), requires_grad=True)
                else:
                    if len(param_old.shape) == 2:  # weight
                        param_new[0:param_old.shape[0], 0: param_old.shape[1]] = Variable(param_old.clone(),
                                                                                          requires_grad=True)
                        random_initialization = torch.empty(param_old.shape[0],
                                                            param_new.shape[1] - param_old.shape[1]).clone().normal_(0,
                                                                                                                     std).type(
                            torch.cuda.FloatTensor)
                        param_new[0:param_old.shape[0], param_old.shape[1]:] = Variable(random_initialization,
                                                                                        requires_grad=True)
                        random_initialization = torch.empty(param_new.shape[0] - param_old.shape[0],
                                                            param_new.shape[1]).clone().normal_(0, std).type(
                            torch.cuda.FloatTensor)
                        param_new[param_old.shape[0]:, :] = Variable(random_initialization, requires_grad=True)
                    else:
                        param_new[0:param_old.shape[0]] = Variable(param_old.clone(), requires_grad=True)
                        random_initialization = torch.empty(param_new.shape[0] - param_old.shape[0]).clone().normal_(0,
                                                                                                                     std).type(
                            torch.cuda.FloatTensor)
                        param_new[param_old.shape[0]:] = Variable(random_initialization, requires_grad=True)
            elif args.initialization == 'zero':
                logging.info('zero initialization')
                if re.search('emb', layer_name):
                    param_new[:, 0:param_old.shape[1]] = Variable(param_old.clone(), requires_grad=True)
                    zero_initialization = torch.zeros(param_old.shape[0],
                                                      param_new.shape[1] - param_old.shape[1]).clone().type(
                        torch.cuda.FloatTensor)
                    param_new[:, param_old.shape[1]:] = Variable(zero_initialization, requires_grad=True)
                elif layer_name == 'bot_l.0.weight':
                    param_new[0:param_old.shape[0], :] = Variable(param_old.clone(), requires_grad=True)
                    zero_initialization = torch.zeros(param_new.shape[0] - param_old.shape[0],
                                                      param_old.shape[1]).clone().type(torch.cuda.FloatTensor)
                    param_new[param_old.shape[0]:, :] = Variable(zero_initialization, requires_grad=True)
                elif layer_name == 'bot_l.0.bias':
                    param_new[0:param_old.shape[0]] = Variable(param_old.clone(), requires_grad=True)
                    zero_initialization = torch.zeros(param_new.shape[0] - param_old.shape[0]).clone().type(
                        torch.cuda.FloatTensor)
                    param_new[param_old.shape[0]:] = Variable(zero_initialization, requires_grad=True)
                elif layer_name == 'top_l.4.weight':
                    param_new[:, 0:param_old.shape[1]] = Variable(param_old.clone(), requires_grad=True)
                    zero_initialization = torch.zeros(param_old.shape[0],
                                                      param_new.shape[1] - param_old.shape[1]).clone().type(
                        torch.cuda.FloatTensor)
                    param_new[:, param_old.shape[1]:] = Variable(zero_initialization, requires_grad=True)
                elif layer_name == 'top_l.4.bias':
                    param_new = Variable(param_old.clone(), requires_grad=True)
                else:
                    if len(param_old.shape) == 2:  # weight
                        param_new[0:param_old.shape[0], 0: param_old.shape[1]] = Variable(param_old.clone(),
                                                                                          requires_grad=True)
                        zero_initialization = torch.zeros(param_old.shape[0],
                                                          param_new.shape[1] - param_old.shape[1]).clone().type(
                            torch.cuda.FloatTensor)
                        param_new[0:param_old.shape[0], param_old.shape[1]:] = Variable(zero_initialization,
                                                                                        requires_grad=True)
                        zero_initialization = torch.zeros(param_new.shape[0] - param_old.shape[0],
                                                          param_new.shape[1]).clone().type(torch.cuda.FloatTensor)
                        param_new[param_old.shape[0]:, :] = Variable(zero_initialization, requires_grad=True)
                    else:
                        param_new[0:param_old.shape[0]] = Variable(param_old.clone(), requires_grad=True)
                        zero_initialization = torch.zeros(param_new.shape[0] - param_old.shape[0]).clone().type(
                            torch.cuda.FloatTensor)
                        param_new[param_old.shape[0]:] = Variable(zero_initialization, requires_grad=True)

            new_param_dict[layer_name] = Variable(param_new.type(torch.cuda.FloatTensor), requires_grad=True)
        to_net.load_state_dict(new_param_dict)
        if args.initialization == 'random':
            handle1 = to_net.bot_l[1].register_forward_hook(hook_fn_forward)
            handle2 = to_net.bot_l[3].register_forward_hook(hook_fn_forward)
            handle3 = to_net.bot_l[5].register_forward_hook(hook_fn_forward)
            handle4 = to_net.bot_l[7].register_forward_hook(hook_fn_forward)
            handle5 = to_net.top_l[1].register_forward_hook(hook_fn_forward)
            handle6 = to_net.top_l[3].register_forward_hook(hook_fn_forward)
            handle1.remove()
            handle2.remove()
            handle3.remove()
            handle4.remove()
            handle5.remove()
            handle6.remove()

        logging.info('load_model: Loading {}....'.format(path))


    def random_mask(to_net, prune_ratio):
        all_mask = collections.defaultdict()
        logging.info("Randomly generating mask, prune_ratio={}........\n".format(prune_ratio))

        for name, param in to_net.named_parameters():
            if 'bot_l' in name or 'top_l' in name:
                if 'weight' in name:
                    weight_copy = param.data.abs().clone().cpu().numpy()

                    num_keep = int(weight_copy.shape[0] * (1.0 - prune_ratio))
                    if num_keep <= 0:  # the output layer
                        continue
                    if name == 'bot_l.6.weight':
                        continue
                    index_list = random.choices(np.arange(weight_copy.shape[0]).tolist(),k = num_keep)
                    # print(weight_copy.shape)
                    # print('index_list',index_list)
                    mask = np.zeros(weight_copy.shape)
                    mask[index_list, :] = 1.0  # 1 means keep weights
                    # print(mask, mask.shape)
                    all_mask[name] = mask
                    mask_bias = mask[:, 0]
                    bias_name = name[0:8] + 'bias'
                    all_mask[bias_name] = mask_bias

        return all_mask, index_list



    def generate_mask(to_net, prune_ratio):
        gradient_list = collections.defaultdict()
        weight_list = collections.defaultdict()
        taylor_list = collections.defaultdict()
        threshold_list = collections.defaultdict()
        all_mask = collections.defaultdict()

        logging.info("Generating mask, prune_ratio={}, according to Taylor ........\n".format(prune_ratio))

        for name, param in to_net.named_parameters():
            if 'bot_l' in name or 'top_l' in name:
                if 'weight' in name:
                    weight_copy = param.data.abs().clone().cpu().numpy()
                    grad_copy = param.grad.data.abs().clone().cpu() .numpy()
                    taylor = np.sum(weight_copy * grad_copy, axis=1)
                    num_keep = int(weight_copy.shape[0] * (1.0 - prune_ratio))
                    if num_keep <= 0: # the output layer
                        continue
                    if param.shape[0] == args.arch_sparse_feature_size:
                        logging.info('skipping ' + name)
                        continue
                    arg_max = np.argsort(taylor)  # Returns the indices that would sort an array. small->big
                    arg_max_rev = arg_max[::-1][:num_keep]  # big->small
                    thre = taylor[arg_max_rev[-1]]
                    mask = np.zeros(weight_copy.shape)
                    mask[arg_max_rev.tolist(), :] = 1.0  # 1 means keep weights
                    # print(mask, mask.shape)
                    all_mask[name] = mask
                    threshold_list[name] = thre
                    gradient_list[name] = grad_copy
                    weight_list[name] = weight_copy
                    taylor_list[name] = taylor

                    mask_bias = mask[:, 0]
                    bias_name = name[0:8] + 'bias'
                    all_mask[bias_name] = mask_bias

        return all_mask, arg_max_rev.tolist()


    def structured_masking_initalization(to_net, mask_dict, initialization = None):
        # logging.info('Pruning....')
        param_processed = OrderedDict([(k, None) for k in to_net.state_dict().keys()])
        for layer_name, param_current in to_net.state_dict().items():
            param_current = param_current.type(torch.cuda.FloatTensor)
            if layer_name in mask_dict.keys():
                masked_weight = Variable(torch.mul(param_current, torch.from_numpy(mask_dict[layer_name]).type(torch.cuda.FloatTensor)), requires_grad=True)
                if initialization and param_current.shape[0] != 1:
                    logging.info('Random initialization according to Normal Distribution...........')
                    std = param_current.std().item()
                    random_initialization = torch.empty(param_current.shape).clone().normal_(0, std).type(torch.cuda.FloatTensor)
                    param_processed[layer_name] = torch.where(masked_weight == 0.0, random_initialization, masked_weight)
                else:
                    logging.info('Zero initialization..............')
                    param_processed[layer_name] = masked_weight
            else:
                param_processed[layer_name] = param_current
        to_net.load_state_dict(param_processed)



    def dlrm_wrap(dlrm_net, X, lS_o, lS_i, use_gpu, device):
        if use_gpu:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
                else lS_i.to(device)
            lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
                else lS_o.to(device)
            return dlrm_net(
                X.to(device),
                lS_o,
                lS_i
            )
        else:
            return dlrm_net(X, lS_o, lS_i)


    ### main loop ###
    def time_wrap(use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()


    def loss_fn_wrap(Z, T, use_gpu, device):
        # specify the loss function
        if args.loss_function == "mse":
            loss_fn = torch.nn.MSELoss(reduction="mean")
        elif args.loss_function == "bce":
            loss_fn = torch.nn.BCELoss(reduction="mean")
        elif args.loss_function == "wbce":
            loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
            loss_fn = torch.nn.BCELoss(reduction="none")
        else:
            sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

        if args.loss_function == "mse" or args.loss_function == "bce":
            if use_gpu:
                return loss_fn(Z, T.to(device))
            else:
                return loss_fn(Z, T)
        elif args.loss_function == "wbce":
            if use_gpu:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T).to(device)
                loss_fn_ = loss_fn(Z, T.to(device))
            else:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            # debug prints
            # print(loss_ws_)
            # print(loss_fn_)
            return loss_sc_.mean()


    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda", 0)
        ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    trainset, testset = prepare_dataset()
    train_data, train_ld, nbatches = trainset
    test_data, test_ld, nbatches_test = testset
    logging.info('mask_ratio={}, mask_delay={}'.format(mask_ratio, mask_delay))
    dimension_info, ndevices = instance_dimension(size_scale = 1.0, trainset=trainset)
    m_spa, ln_emb, ln_bot, ln_top, num_fea, num_int = dimension_info
    train_data, train_ld, nbatches = trainset
    test_data, test_ld, nbatches_test = testset

    dlrm, optimizer, lr_scheduler = instance_dlrm(m_spa, ln_emb, ln_bot, ln_top, ndevices)

    # training or inference
    best_gA_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    total_samp = 0
    k = 0

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device('cuda')
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
        dlrm.load_state_dict(ld_model["state_dict"])
        ld_j = ld_model["iter"]
        ld_k = ld_model["epoch"]
        ld_nepochs = ld_model["nepochs"]
        ld_nbatches = ld_model["nbatches"]
        ld_nbatches_test = ld_model["nbatches_test"]
        ld_gA = ld_model["train_acc"]
        ld_gL = ld_model["train_loss"]
        ld_total_loss = ld_model["total_loss"]
        ld_total_accu = ld_model["total_accu"]
        ld_gA_test = ld_model["test_acc"]
        ld_gL_test = ld_model["test_loss"]
        if not args.inference_only:
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            best_gA_test = ld_gA_test
            total_loss = ld_total_loss
            total_accu = ld_total_accu
            skip_upto_epoch = ld_k  # epochs
            skip_upto_batch = ld_j  # batches
        else:
            args.print_freq = ld_nbatches
            args.test_freq = 0

        print(
            "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
                ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
            )
        )
        print(
            "Training state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL, ld_gA * 100
            )
        )
        print(
            "Testing state: loss = {:.6f}, accuracy = {:3.3f} %".format(
                ld_gL_test, ld_gA_test * 100
            )
        )
    time_start = time.time()
    ### train starts
    logging.info("time/loss/accuracy (if enabled):")
    gL_log = []
    gA_log = []
    param_log = []
    param_FC_log = []
    dimension_info_dict = collections.defaultdict()
    stage_id = 0
    pruned_flag = False
    with torch.autograd.profiler.profile(args.enable_profiling, use_gpu) as prof:
        while k < args.nepochs:
            if k < skip_upto_epoch:
                continue

            accum_time_begin = time_wrap(use_gpu)

            if args.mlperf_logging:
                previous_iteration_time = None

            if len(mask_ratio) != 1:
                start_idx = math.ceil(nbatches / len(mask_ratio)) * stage_id
                logging.info('Stage {}, This growth start from input index {}'.format(stage_id, start_idx))
            else:
                start_idx = 0

            for j, (X, lS_o, lS_i, T) in enumerate(train_ld, start_idx):
                # ### locate growth
                if len(mask_delay) != 0:
                    for idx, delay in enumerate(mask_delay):
                        if j == math.floor(nbatches * delay):
                            logging.info(
                                'DSD stage {}, mask_ratio={} mask_delay ={} ------------------DSD starts---------------------\n'.format(stage_id, mask_ratio[idx], delay))
                            if idx == 0:
                                all_mask, mask_idx = random_mask(dlrm, mask_ratio[idx])
                                structured_masking_initalization(dlrm, all_mask, None)

                            elif mask_ratio[idx - 1] > mask_ratio[idx]: # Growth:
                                all_mask, mask_idx = generate_mask(dlrm, mask_ratio[idx])
                                structured_masking_initalization(dlrm, all_mask, True)

                            elif mask_ratio[idx] == 0.0 and mask_ratio[idx - 1] < mask_ratio[idx]: #pruning
                                all_mask, mask_idx = generate_mask(dlrm, mask_ratio[idx])
                                structured_masking_initalization(dlrm, all_mask, None)
                            else:
                                sys.exit('DO NOT HAVE THIS GROWTH OPTION')
                                
                            if mask_ratio[idx] != 0.0:
                                pruned_flag = True
                            else:
                                pruned_flag = False

                            # if idx == 0:
                            #     dimension_info, ndevices = instance_dimension(size_scale=(1.0 - mask_ratio[0]),
                            #                                                   trainset=trainset)
                            #     dimension_info_dict['DSD{}'.format(idx)] = dimension_info
                            #     m_spa, ln_emb, ln_bot, ln_top, num_fea, num_int = dimension_info
                            #     dlrm, optimizer, lr_scheduler = instance_dlrm(m_spa, ln_emb, ln_bot, ln_top, ndevices)
                            #     pruned_flag = False
                            #
                            # elif mask_ratio[idx - 1] > mask_ratio[idx]: # Growth
                            #     all_mask, mask_idx = generate_mask(dlrm, mask_ratio[idx])
                            #     structured_masking_initalization(dlrm, all_mask)
                            #     if mask_ratio[idx] != 0.0:
                            #         pruned_flag = True
                            #     else:
                            #         pruned_flag = False
                            #
                            #
                            # elif mask_ratio[idx - 1] < mask_ratio[idx]: #pruning
                            #     all_mask, mask_idx = generate_mask(dlrm, mask_ratio[idx])
                            #     if mask_ratio[idx] != 0.0:
                            #         pruned_flag = True
                            #     else:
                            #         pruned_flag = False
                            #
                            # elif mask_ratio[idx - 1] == mask_ratio[idx]:
                            #     if mask_ratio[idx] != 0.0:
                            #         pruned_flag = True
                            #     else:
                            #         pruned_flag = False
                            # else:
                            #     sys.exit("Does not match any condition")


                            # save_trained_model(dlrm, growth_id)
                            # logging.info('Growth ID {}, Growing size from {}X to {}X.....\n'.format(growth_id, args.growth_ratio * growth_id + mask_ratio[0], args.growth_ratio * (growth_id+1) + mask_ratio[0]))
                            # dimension_info, ndevices = instance_dimension(size_scale = args.growth_ratio * (growth_id+1) + mask_ratio[0], trainset=trainset)
                            # dimension_info_dict['Growth{}'.format(growth_id)] = dimension_info
                            # m_spa, ln_emb, ln_bot, ln_top, num_fea, num_int = dimension_info
                            # dlrm, optimizer, lr_scheduler = instance_dlrm(m_spa, ln_emb, ln_bot, ln_top, ndevices)
                            # load_trained_model_stacking(dlrm, growth_id)
                            stage_id += 1
                            param_FC = utils.count_parameters_in_FC(dlrm, pruned_flag=True)
                            logging.info('FC param size = {:.2f}K,  FLOP = {:.2f}K'.format(param_FC, param_FC * 2))
                            logging.info(
                                'DSD stage {}, mask_ratio={} mask_delay ={} ------------------DSD finishes---------------------\n'.format(
                                    stage_id, mask_ratio[idx], delay))

                if j == 0 and args.save_onnx:
                    (X_onnx, lS_o_onnx, lS_i_onnx) = (X, lS_o, lS_i)
                if j < skip_upto_batch:
                    continue
                if args.mlperf_logging:
                    current_time = time_wrap(use_gpu)
                    if previous_iteration_time:
                        iteration_time = current_time - previous_iteration_time
                    else:
                        iteration_time = 0
                    previous_iteration_time = current_time
                else:
                    t1 = time_wrap(use_gpu)
                # early exit if nbatches was set by the user and has been exceeded
                if nbatches > 0 and j >= nbatches:
                    break

                '''
                # debug prints
                print("input and targets")
                print(X.detach().cpu().numpy())
                print([np.diff(S_o.detach().cpu().tolist()
                       + list(lS_i[i].shape)).tolist() for i, S_o in enumerate(lS_o)])
                print([S_i.detach().cpu().numpy().tolist() for S_i in lS_i])
                print(T.detach().cpu().numpy())
                '''

                # forward pass
                Z = dlrm_wrap(dlrm, X, lS_o, lS_i, use_gpu, device)
                # loss
                E = loss_fn_wrap(Z, T, use_gpu, device)
                '''
                # debug prints
                print("output and loss")
                print(Z.detach().cpu().numpy())
                print(E.detach().cpu().numpy())
                '''
                # compute loss and accuracy
                L = E.detach().cpu().numpy()  # numpy array
                S = Z.detach().cpu().numpy()  # numpy array
                T = T.detach().cpu().numpy()  # numpy array
                mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                A = np.sum((np.round(S, 0) == T).astype(np.uint8))

                if not args.inference_only:
                    # scaled error gradient propagation
                    # (where we do not accumulate gradients across mini-batches)
                    optimizer.zero_grad()
                    # backward pass
                    E.backward()
                    # debug prints (check gradient norm)
                    # for l in mlp.layers:
                    #     if hasattr(l, 'weight'):
                    #          print(l.param.grad.norm().item())

                    # optimizer
                    optimizer.step()
                    lr_scheduler.step()

                if args.mlperf_logging:
                    total_time += iteration_time
                else:
                    t2 = time_wrap(use_gpu)
                    total_time += t2 - t1
                total_accu += A
                total_loss += L * mbs
                total_iter += 1
                total_samp += mbs

                should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
                should_test = (
                        (args.test_freq > 0)
                        and (args.data_generation == "dataset")
                        and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                )
                if pruned_flag:
                    structured_masking_initalization(dlrm, all_mask, None)

                param = utils.count_parameters_in_MB(dlrm)
                param_FC = utils.count_parameters_in_FC(dlrm, pruned_flag)
                param_log.append(param)
                param_FC_log.append(param_FC)
                # print time, loss and accuracy
                if should_print or should_test:
                    gT = 1000.0 * total_time / total_iter if args.print_time else -1
                    total_time = 0

                    gA = total_accu / total_samp
                    total_accu = 0

                    gL = total_loss / total_samp
                    total_loss = 0
                    gL_log.append(gL)
                    gA_log.append(gA)

                    str_run_type = "inference" if args.inference_only else "training"
                    logging.info(
                        "Finished {} it {}/{} of epoch {}, {:.2f} ms/it, ".format(
                            str_run_type, j + 1, nbatches, k, gT
                        )
                        + "loss {:.6f}, accuracy {:3.3f} %,  lr = {:.3f},  pruned_flag = {}".format(gL, gA * 100, lr_scheduler.get_lr()[0], pruned_flag)
                    )
                    # Uncomment the line below to print out the total time with overhead
                    # print("Accumulated time so far: {}" \
                    # .format(time_wrap(use_gpu) - accum_time_begin))
                    total_iter = 0
                    total_samp = 0

                    # for name, v in dlrm.named_parameters():
                    #     if "bot_l.4" in name:
                    #         logging.info('weight', v.data)
                    #         logging.info('gradient', v.grad.data)
                    #         logging.info('\n\n')
                # testing
                if should_test and not args.inference_only:
                    # don't measure training iter time in a test iteration
                    logging.info('Testing.....')
                    if args.mlperf_logging:
                        previous_iteration_time = None

                    test_accu = 0
                    test_loss = 0
                    test_samp = 0

                    accum_test_time_begin = time_wrap(use_gpu)
                    if args.mlperf_logging:
                        scores = []
                        targets = []

                    for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
                        # early exit if nbatches was set by the user and was exceeded
                        if nbatches > 0 and i >= nbatches:
                            break

                        t1_test = time_wrap(use_gpu)

                        # forward pass
                        Z_test = dlrm_wrap(dlrm,
                                           X_test, lS_o_test, lS_i_test, use_gpu, device
                                           )
                        if args.mlperf_logging:
                            S_test = Z_test.detach().cpu().numpy()  # numpy array
                            T_test = T_test.detach().cpu().numpy()  # numpy array
                            scores.append(S_test)
                            targets.append(T_test)
                        else:
                            # loss
                            E_test = loss_fn_wrap(Z_test, T_test, use_gpu, device)

                            # compute loss and accuracy
                            L_test = E_test.detach().cpu().numpy()  # numpy array
                            S_test = Z_test.detach().cpu().numpy()  # numpy array
                            T_test = T_test.detach().cpu().numpy()  # numpy array
                            mbs_test = T_test.shape[0]  # = mini_batch_size except last
                            A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
                            test_accu += A_test
                            test_loss += L_test * mbs_test
                            test_samp += mbs_test

                        t2_test = time_wrap(use_gpu)

                    if args.mlperf_logging:
                        scores = np.concatenate(scores, axis=0)
                        targets = np.concatenate(targets, axis=0)

                        metrics = {
                            'loss': sklearn.metrics.log_loss,
                            'recall': lambda y_true, y_score:
                            sklearn.metrics.recall_score(
                                y_true=y_true,
                                y_pred=np.round(y_score)
                            ),
                            'precision': lambda y_true, y_score:
                            sklearn.metrics.precision_score(
                                y_true=y_true,
                                y_pred=np.round(y_score)
                            ),
                            'f1': lambda y_true, y_score:
                            sklearn.metrics.f1_score(
                                y_true=y_true,
                                y_pred=np.round(y_score)
                            ),
                            'ap': sklearn.metrics.average_precision_score,
                            'roc_auc': sklearn.metrics.roc_auc_score,
                            'accuracy': lambda y_true, y_score:
                            sklearn.metrics.accuracy_score(
                                y_true=y_true,
                                y_pred=np.round(y_score)
                            ),
                            # 'pre_curve' : sklearn.metrics.precision_recall_curve,
                            # 'roc_curve' :  sklearn.metrics.roc_curve,
                        }

                        # print("Compute time for validation metric : ", end="")
                        # first_it = True
                        validation_results = collections.defaultdict()
                        for metric_name, metric_function in metrics.items():
                            # if first_it:
                            #     first_it = False
                            # else:
                            #     print(", ", end="")
                            # metric_compute_start = time_wrap(False)
                            validation_results[metric_name] = metric_function(
                                targets,
                                scores
                            )
                            # metric_compute_end = time_wrap(False)
                            # met_time = metric_compute_end - metric_compute_start
                            # print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
                            #      end="")
                        # print(" ms")
                        gA_test = validation_results['accuracy']
                        gL_test = validation_results['loss']
                    else:
                        gA_test = test_accu / test_samp
                        gL_test = test_loss / test_samp

                    is_best = gA_test > best_gA_test
                    if is_best:
                        best_gA_test = gA_test
                        if not (args.save_model == ""):
                            logging.info("Saving model to {}".format(args.save_model))
                            torch.save(
                                {
                                    "epoch": k,
                                    "nepochs": args.nepochs,
                                    "nbatches": nbatches,
                                    "nbatches_test": nbatches_test,
                                    "iter": j + 1,
                                    "state_dict": dlrm.state_dict(),
                                    "train_acc": gA,
                                    "train_loss": gL,
                                    "test_acc": gA_test,
                                    "test_loss": gL_test,
                                    "total_loss": total_loss,
                                    "total_accu": total_accu,
                                    "opt_state_dict": optimizer.state_dict(),
                                },
                                args.save_model,
                            )

                    if args.mlperf_logging:
                        is_best = validation_results['roc_auc'] > best_auc_test
                        if is_best:
                            best_auc_test = validation_results['roc_auc']

                        logging.info(
                            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
                            + " loss {:.6f}, recall {:.4f}, precision {:.4f},".format(
                                validation_results['loss'],
                                validation_results['recall'],
                                validation_results['precision']
                            )
                            + " f1 {:.4f}, ap {:.4f},".format(
                                validation_results['f1'],
                                validation_results['ap'],
                            )
                            + " auc {:.4f}, best auc {:.4f},".format(
                                validation_results['roc_auc'],
                                best_auc_test
                            )
                            + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                                validation_results['accuracy'] * 100,
                                best_gA_test * 100
                            )
                        )
                    else:
                        logging.info(
                            "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, 0)
                            + " loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                                gL_test, gA_test * 100, best_gA_test * 100
                            )
                        )
                    # Uncomment the line below to print out the total time with overhead
                    # logging.info("Total test time for this group: {}" \
                    # .format(time_wrap(use_gpu) - accum_test_time_begin))

                    if (args.mlperf_logging
                            and (args.mlperf_acc_threshold > 0)
                            and (best_gA_test > args.mlperf_acc_threshold)):
                        logging.info("MLPerf testing accuracy threshold "
                                     + str(args.mlperf_acc_threshold)
                                     + " reached, stop training")
                        break

                    if (args.mlperf_logging
                            and (args.mlperf_auc_threshold > 0)
                            and (best_auc_test > args.mlperf_auc_threshold)):
                        logging.info("MLPerf testing auc threshold "
                                     + str(args.mlperf_auc_threshold)
                                     + " reached, stop training")
                        break



                torch.cuda.empty_cache()
            k += 1  # nepochs

            #### train ends

    time_end = time.time()
    logging.info('time cost {:.2f} second'.format(time_end - time_start))
    # profiling
    if args.enable_profiling:
        with open("dlrm_s_pytorch.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
            prof.export_chrome_trace("./dlrm_s_pytorch.json")
        # logging.info(prof.key_averages().table(sort_by="cpu_time_total"))

    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # test prints
    if not args.inference_only and args.debug_mode:
        logging.info("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            logging.info(param.detach().cpu().numpy())

    # export the model in onnx
    if args.save_onnx:
        dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
        batch_size = X_onnx.shape[0]
        # debug prints
        # print("batch_size", batch_size)
        # print("inputs", X_onnx, lS_o_onnx, lS_i_onnx)
        # print("output", dlrm_wrap(X_onnx, lS_o_onnx, lS_i_onnx, use_gpu, device))

        # force list conversion
        # if torch.is_tensor(lS_o_onnx):
        #    lS_o_onnx = [lS_o_onnx[j] for j in range(len(lS_o_onnx))]
        # if torch.is_tensor(lS_i_onnx):
        #    lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
        # force tensor conversion
        # if isinstance(lS_o_onnx, list):
        #     lS_o_onnx = torch.stack(lS_o_onnx)
        # if isinstance(lS_i_onnx, list):
        #     lS_i_onnx = torch.stack(lS_i_onnx)
        # debug prints
        print("X_onnx.shape", X_onnx.shape)
        if torch.is_tensor(lS_o_onnx):
            print("lS_o_onnx.shape", lS_o_onnx.shape)
        else:
            for oo in lS_o_onnx:
                print("oo.shape", oo.shape)
        if torch.is_tensor(lS_i_onnx):
            print("lS_i_onnx.shape", lS_i_onnx.shape)
        else:
            for ii in lS_i_onnx:
                print("ii.shape", ii.shape)

        # name inputs and outputs
        o_inputs = ["offsets"] if torch.is_tensor(lS_o_onnx) else ["offsets_" + str(i) for i in range(len(lS_o_onnx))]
        i_inputs = ["indices"] if torch.is_tensor(lS_i_onnx) else ["indices_" + str(i) for i in range(len(lS_i_onnx))]
        all_inputs = ["dense_x"] + o_inputs + i_inputs
        # debug prints
        print("inputs", all_inputs)

        # create dynamic_axis dictionaries
        do_inputs = [{'offsets': {1: 'batch_size'}}] if torch.is_tensor(lS_o_onnx) else [
            {"offsets_" + str(i): {0: 'batch_size'}} for i in range(len(lS_o_onnx))]
        di_inputs = [{'indices': {1: 'batch_size'}}] if torch.is_tensor(lS_i_onnx) else [
            {"indices_" + str(i): {0: 'batch_size'}} for i in range(len(lS_i_onnx))]
        dynamic_axes = {'dense_x': {0: 'batch_size'}, 'pred': {0: 'batch_size'}}
        for do in do_inputs:
            dynamic_axes.update(do)
        for di in di_inputs:
            dynamic_axes.update(di)
        # debug prints
        print(dynamic_axes)

        # export model
        torch.onnx.export(
            dlrm, (X_onnx, lS_o_onnx, lS_i_onnx), dlrm_pytorch_onnx_file, verbose=True, use_external_data_format=True,
            opset_version=11, input_names=all_inputs, output_names=["pred"], dynamic_axes=dynamic_axes
        )
        # recover the model back
        dlrm_pytorch_onnx = onnx.load(dlrm_pytorch_onnx_file)
        # check the onnx model
        onnx.checker.check_model(dlrm_pytorch_onnx)
        '''
        # run model using onnxruntime
        import onnxruntime as rt
        dict_inputs = collections.defaultdict()
        dict_inputs["dense_x"] = X_onnx.numpy().astype(np.float32)
        if torch.is_tensor(lS_o_onnx):
            dict_inputs["offsets"] = lS_o_onnx.numpy().astype(np.int64)
        else:
            for i in range(len(lS_o_onnx)):
                dict_inputs["offsets_"+str(i)] = lS_o_onnx[i].numpy().astype(np.int64)
        if torch.is_tensor(lS_i_onnx):
            dict_inputs["indices"] = lS_i_onnx.numpy().astype(np.int64)
        else:
            for i in range(len(lS_i_onnx)):
                dict_inputs["indices_"+str(i)] = lS_i_onnx[i].numpy().astype(np.int64)
        print("dict_inputs", dict_inputs)
        sess = rt.InferenceSession(dlrm_pytorch_onnx_file, rt.SessionOptions())
        prediction = sess.run(output_names=["pred"], input_feed=dict_inputs)
        print("prediction", prediction)
        '''
    flops = sum(param_FC_log) * 2 / 1000000

    scio.savemat(
        './log/savemat_bot{}_loss{:.5f}_accu{:.5f}_flop{:.2f}G.mat'.format(args.arch_mlp_bot, gL, sum(gA_log[-30:])/30.0, flops),
        {'gL_log': gL_log, 'gA_log': gA_log, 'gA_test':gA_test, 'masking_delay':args.masking_delay, 'masking_ratio':args.masking_ratio,
         'args.grow_embedding': args.grow_embedding, 'param_FC_log': param_FC_log, 'param_log': param_log,
         'm_spa': m_spa, 'ln_emb': ln_emb, 'num_fea': num_fea, 'num_int': num_int,
         'ln_bot': ln_bot, 'ln_top': ln_top, 'nbatches': nbatches, 'dimension_info_dict': dimension_info_dict,
         'args.arch_mlp_bot': args.arch_mlp_bot, 'args.arch_mlp_top': args.arch_mlp_top})

    assert len(gL_log) == len(gA_log)
    title_font = {'size': '8', 'color': 'black', 'weight': 'normal'}  # Bottom vertical alignment for more space
    axis_font = {'size': '10'}

    plt.figure(figsize=(16, 6.5))
    plt.subplot(131)
    # plt.figure()
    x = np.linspace(0, len(gL_log), num=len(gL_log))
    plt.xlim(0, len(gL_log))
    plt.xlabel("Num of Iterations")
    plt.ylabel('BCELoss')
    plt.plot(x, gL_log, 'r-o', alpha=1.0, label='Loss - FC growth')
    plt.yticks(np.arange(0.4, 0.6, step=0.1))
    plt.xticks(np.arange(0, len(gL_log) + 1, step=20), rotation=45)
    plt.legend(loc='lower right')
    plt.title('Kaggle Display Advertising Challenge Dataset')

    plt.subplot(132)
    x = np.linspace(0, len(gA_log), num=len(gA_log))
    plt.xlim(0, len(gA_log))
    plt.xlabel("Num of Iterations")
    plt.ylabel('Accuracy %')
    plt.plot(x, gA_log, 'g-', alpha=1.0, label='Accuracy - FC growth')
    plt.yticks(np.arange(0.75, 0.80, step=0.01))
    plt.xticks(np.arange(0, len(gA_log) + 1, step=20), rotation=45)
    plt.legend(loc='lower left')
    plt.title('Kaggle Display Advertising Challenge Dataset')

    plt.subplot(133)
    x = np.linspace(0, len(param_FC_log), num=len(param_FC_log))
    plt.xlim(0, len(param_FC_log))
    plt.xlabel("Num of Iterations")
    plt.ylabel('Num of parameters (K)')
    plt.plot(x, param_FC_log, 'b-', alpha=1.0, label='FC param - FC growth')
    plt.xticks(np.arange(0, len(param_FC_log) + 1, step=20 * 2048), rotation=45)
    plt.legend(loc='lower right')
    plt.title('Kaggle Display Advertising Challenge Dataset')
    plt.savefig(
        './log/learning_curve_bot{}_maskDelay{}_maskRatio{}_TestAcc{:.5f}.png'.format(args.arch_mlp_bot, mask_delay, mask_ratio, gA_test))
