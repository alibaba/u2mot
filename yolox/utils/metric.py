#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np

import torch

import functools
import os
import time
from collections import defaultdict, deque

__all__ = [
    "AverageMeter",
    "MeterBuffer",
    "get_total_and_free_memory_in_Mb",
    "occupy_mem",
    "gpu_mem_usage",
    "calc_pseudo_acc"
]


def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
    )
    devices_info = devices_info_str.read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_mem(cuda_device, mem_ratio=0.95):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


class AverageMeter:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=50):
        self._deque = deque(maxlen=window_size)
        self._total = 0.0
        self._count = 0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    @property
    def median(self):
        d = np.array(list(self._deque))
        return np.median(d)

    @property
    def avg(self):
        # if deque is empty, nan will be returned.
        d = np.array(list(self._deque))
        return d.mean()

    @property
    def global_avg(self):
        return self._total / max(self._count, 1e-5)

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

    @property
    def total(self):
        return self._total

    def reset(self):
        self._deque.clear()
        self._total = 0.0
        self._count = 0

    def clear(self):
        self._deque.clear()


class MeterBuffer(defaultdict):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=20):
        factory = functools.partial(AverageMeter, window_size=window_size)
        super().__init__(factory)

    def reset(self):
        for v in self.values():
            v.reset()

    def get_filtered_meter(self, filter_key="time"):
        return {k: v for k, v in self.items() if filter_key in k}

    def update(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
            self[k].update(v)

    def clear_meters(self):
        for v in self.values():
            v.clear()


def calc_pseudo_acc(true_dataset, pred_dataset, interval=1):
    sample_num = len(true_dataset)
    assert sample_num == len(pred_dataset)
    id_labels_prev_list, id_pseudo_prev_list, seq_prev = [], [], ''
    accuracy = []
    for index in range(sample_num):
        labels, img_info, file_name = true_dataset.annotations[index]
        preds, _, _ = pred_dataset.annotations[index]
        id_labels = labels[:, -1]
        id_pseudo = preds[:, -1]
        seq_curr = true_dataset.img_path2seq(file_name)
        if seq_prev == seq_curr:
            if len(id_labels_prev_list) >= interval:
                id_labels_prev, id_pseudo_prev = id_labels_prev_list[0], id_pseudo_prev_list[0]
                assign_labels = id_labels.reshape(-1, 1) == id_labels_prev.reshape(1, -1)
                assign_pseudo = id_pseudo.reshape(-1, 1) == id_pseudo_prev.reshape(1, -1)

                unpair_i = np.arange(assign_labels.shape[0]) > -1
                i, j = np.nonzero(assign_labels)
                acc = id_pseudo[i] == id_pseudo_prev[j]
                # accuracy.append(acc)
                
                unpair_i[i] = False
                acc_neg = assign_pseudo[unpair_i].sum(axis=1) == 0
                # accuracy.append(acc_neg)
                accuracy.append(np.concatenate((acc, acc_neg)))

                id_labels_prev_list.pop(0)
                id_pseudo_prev_list.pop(0)
            else:
                pass
        else:
            id_labels_prev_list, id_pseudo_prev_list = [], []
        
        id_labels_prev_list.append(id_labels)
        id_pseudo_prev_list.append(id_pseudo)
        seq_prev = seq_curr

    accuracy = np.concatenate(accuracy)

    print('Interval %3d, ACC = %.4f' % (interval, np.mean(accuracy)))