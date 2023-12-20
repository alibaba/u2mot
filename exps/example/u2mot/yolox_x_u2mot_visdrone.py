#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import math

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 10
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "trainval.json"
        self.val_ann = "test.json"  # change to train.json when running on training set
        self.input_size = (800, 1440)
        self.test_size = (896, 1600)
        # self.test_size = (1088, 1920)
        self.random_size = (18, 28)  # 32 leads to run-out-of-memory
        self.max_epoch = 40
        self.print_interval = 20
        self.eval_interval = 1e5
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

        self.cur_epoch = 0

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def get_moco_scale(self):
        return self.get_adjust_scale(t=1.0)

    def get_reid_scale(self):
        return self.get_adjust_scale(t=0.8)

    def get_adjust_scale(self, t=0.8):
        decay_thresh = self.max_epoch * t
        if self.cur_epoch <= decay_thresh:
            return (1 - math.cos(self.cur_epoch * math.pi / decay_thresh)) / 2  # 0 -> 1
        else:
            return 1.

    # def random_resize(self, data_loader, epoch, rank, is_distributed):
    #     # TODO: close multi-scale training
    #     return self.input_size

    def get_model(self, settings=None):
        from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead, MoCo

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels)
            if settings is not None:
                moco = MoCo(backbone, head, settings['seq_names'], dim=head.emb_dim)
            else:
                moco = None
            self.model = YOLOX(backbone, head, moco=moco)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            list_collate,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "VisDrone-MOT"),
            json_file=self.train_ann,
            name='',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
            max_epoch=self.max_epoch,
            is_training=True
        )
        seq_names = dataset.seq_names
        img_path2seq = dataset.img_path2seq

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True
        }
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["collate_fn"] = list_collate  # TODO: preserve img_info data structure
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        settings = {'seq_names': seq_names, 'img_path2seq': img_path2seq}
        return train_loader, settings

    def get_infer_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform, list_collate

        traindataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "VisDrone-MOT"),
            json_file=self.train_ann,
            img_size=self.test_size,
            name='',  # change to train when running on training set
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                get_tgt=True,
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(traindataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(traindataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        dataloader_kwargs["collate_fn"] = list_collate  # TODO: preserve img_info data structure
        infer_loader = torch.utils.data.DataLoader(traindataset, **dataloader_kwargs)

        return infer_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(get_yolox_datadir(), "VisDrone-MOT"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='',  # change to train when running on training set
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator