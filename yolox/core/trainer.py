#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright (c) Alibaba, Inc. and its affiliates.

from loguru import logger
from collections import deque

import torch

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (MeterBuffer, ModelEMA, all_reduce_norm,
                         get_model_info, get_rank, get_world_size,
                         gpu_mem_usage, load_ckpt, occupy_mem, save_checkpoint,
                         setup_logger, synchronize, bboxes_iou_np,
                         torch_distributed_zero_first, calc_pseudo_acc)
from yolox.tracker.u2mot_tracker import U2MOTTracker, DefaultArgs

import datetime
import os
import time
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = args.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema  # True

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32  # torch.float16
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

        # TODO: for reid
        self.settings = {}
        self.start_epoch = 0  # set default value
        self.freeze_detector = self.args.freeze

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch(end=self.epoch == self.max_epoch - 1)

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    @torch.no_grad()
    def extract_pseudo_feat(self):
        self.model.eval()
        if self.is_distributed:
            dtype = next(self.model.module.parameters()).dtype
            model_head = self.model.module.head
        else:
            dtype = next(self.model.parameters()).dtype
            model_head = self.model.head
        s = min(model_head.strides)

        dataset = self.infer_loader.dataset
        pbar = self.infer_loader
        if self.rank in [0, -1]:
            pbar = tqdm(pbar, desc=f'Extract pseudo embeddings')
        for inps, targets, img_infos, img_ids in pbar:
            if all(dataset.is_img_static(info[4]) for info in img_infos):
                # skip the whole batch
                continue

            inps = inps.to(dtype).to(self.device)
            reid_feat = self.model(inps)[-1]
            ny, nx = reid_feat.shape[1:3]

            for bi, info in enumerate(img_infos):
                file_name = info[4]
                target = targets[bi]

                if dataset.is_img_static(file_name):
                    # skip this image
                    continue
                target = np.concatenate(target)
                if len(target) == 0:
                    continue
                xc, yc = (target[:, [1, 2]].T / s).astype(np.int32)
                embedding = reid_feat[bi,
                                      yc.clip(0, ny - 1),
                                      xc.clip(0, nx - 1), :].cpu().numpy()

                index = dataset.ids.index(int(img_ids[bi]))
                save_path = dataset.get_ann_tmp_path(index, '_emb.npy')
                np.save(save_path, embedding)

    @torch.no_grad()
    def infer_pseudo(self):  # verification and rectification
        if self.rank not in [0, -1]:  # stupid synchronize
            self.train_loader.dataset._dataset.load_anns_to_mem()
            return

        from scipy.optimize import linear_sum_assignment as linear_assignment

        def _linear_assignment(similarity, t=0.3):
            i, j = linear_assignment(1. - similarity)
            valid = similarity[i, j] > t
            i, j = i[valid], j[valid]
            return i, j

        dataset = self.train_loader.dataset._dataset
        embedding_prev, labels_prev, seq_prev, max_id = None, None, '', 0
        feat_dict = {}
        pbar = range(len(dataset))
        pbar = tqdm(pbar, desc='Generate pseudo labels')
        for index in pbar:
            labels, img_info, file_name = dataset.annotations[index]
            if dataset.is_img_static(file_name) or len(labels) == 0:
                embedding_prev, labels_prev = None, None
                continue

            save_path = dataset.get_ann_tmp_path(index, '_emb.npy')
            embedding = np.load(save_path)

            uncertainty = np.zeros_like(labels[:, 0])
            seq_name = img_info[5]
            if embedding_prev is not None and seq_prev == seq_name:  # contiguous frames
                ind_curr, ind_prev = np.arange(len(labels)), np.arange(
                    len(labels_prev))
                similarity0 = (embedding @ embedding_prev.T).clip(0, 1)
                ious = bboxes_iou_np(labels[:, :4].copy(),
                                     labels_prev[:, :4].copy(),
                                     xyxy=False,
                                     giou=False)
                overlaped0 = (ious > 0.1).astype(np.float32)

                # use_uncertain = self.epoch + 1 >= self.max_epoch - self.exp.no_aug_epochs  # close mosaic
                use_uncertain = True
                use_iou = True
                if use_uncertain or False:
                    curr_f = embedding
                    prev_f = [
                        np.stack(feat_dict[_id]).T
                        for _id in labels_prev[:, -1]
                    ]

                for ci in range(int(labels[:, 4].max()) + 1):
                    i0, j0 = labels[:, 4] == ci, labels_prev[:, 4] == ci
                    if i0.sum() * j0.sum() == 0:
                        labels[i0, -1] = np.arange(max_id + 1,
                                                   max_id + 1 + i0.sum())
                        max_id += i0.sum()
                        continue
                    # indices in this class
                    i1, j1 = ind_curr[i0], ind_prev[j0]
                    similarity = similarity0[i1, :][:, j1]
                    if use_iou:  # default NOT
                        overlaped = overlaped0[i1, :][:, j1]
                        i, j = _linear_assignment(similarity * overlaped)
                    else:
                        i, j = _linear_assignment(similarity)

                    risk = U2MOTTracker._risk(similarity, i, j)
                    threshold = U2MOTTracker._threshold(similarity[i, j])
                    match_uncertainty = risk - threshold
                    uncertainty[i1[i]] = np.maximum(match_uncertainty, 0)
                    uncertain = match_uncertainty > 0  # for current class

                    if use_uncertain and uncertain.sum():
                        unpaired_i = i1 > -1
                        unpaired_i[i[~uncertain]] = False
                        unpaired_j = j1 > -1
                        unpaired_j[j[~uncertain]] = False
                        similarity2 = np.zeros(
                            (unpaired_i.sum(), unpaired_j.sum()),
                            dtype=similarity.dtype)
                        for k, ii in enumerate(i1[unpaired_i]):
                            for l, jj in enumerate(j1[unpaired_j]):
                                similarity2[k, l] = \
                                    np.mean(curr_f[ii:ii + 1] @ prev_f[jj]).clip(0, 1) * overlaped0[ii, jj]
                        i2, j2 = _linear_assignment(similarity2)
                        # i, j: indices respect to class-level indicess i1, j1
                        i, j = np.concatenate((i[~uncertain], np.arange(len(i1))[unpaired_i][i2])), \
                               np.concatenate((j[~uncertain], np.arange(len(j1))[unpaired_j][j2]))
                        assert len(i) == len(np.unique(i)) and len(j) == len(
                            np.unique(j))

                    # i1, j1: class-level indices
                    labels[i1[i], -1] = labels_prev[j1[j], -1].copy()
                    newborn = i1 > -100  # all True
                    newborn[i] = False
                    labels[i1[newborn],
                           -1] = np.arange(max_id + 1,
                                           max_id + 1 + newborn.sum())
                    max_id += newborn.sum()
            else:
                labels[:, -1] = np.arange(max_id + 1, max_id + 1 + len(labels))
                feat_dict = {}

            dataset.uncertainties[index] = uncertainty
            embedding_prev = embedding.copy()
            labels_prev = labels.copy()
            max_id = max(max_id, int(labels_prev[:, -1].max()))
            seq_prev = seq_name

            for ii, _id in enumerate(labels_prev[:, -1]):
                if _id not in feat_dict:
                    feat_dict[_id] = deque([], maxlen=5)
                feat_dict[_id].append(embedding[ii])

        self.train_loader.dataset._dataset.save_anns_to_disk()

    @torch.no_grad()
    def infer_pseudo_by_tracker(self):  # infer with integrated MOT tracker
        if self.rank not in [0, -1]:  # stupid synchronize
            self.train_loader.dataset._dataset.load_anns_to_mem()
            return

        dataset = self.train_loader.dataset._dataset
        seq_prev, max_id = '', 0
        tracker = None
        pbar = range(len(dataset))
        pbar = tqdm(pbar, desc='Generate pseudo labels')
        for index in pbar:
            labels, img_info, file_name = dataset.annotations[index]
            if dataset.is_img_static(file_name) or len(labels) == 0:
                if len(labels) == 0 and tracker is not None:
                    tracker.update2(labels.copy(), np.zeros((0, 128)))
                continue

            seq_name = img_info[5]
            if seq_prev != seq_name:
                id_offset = max_id + 1
                args = DefaultArgs(mot20='mot20' in self.file_name)
                tracker = U2MOTTracker(args)
            save_path = dataset.get_ann_tmp_path(index, '_emb.npy')
            embedding = np.load(save_path)

            use_uncertainty = True  #self.epoch + 1 >= self.max_epoch - self.exp.no_aug_epochs  # close mosaic
            identity, uncertainty = tracker.update2(labels.copy(), embedding, use_uncertainty)
            labels[:, -1] = identity + id_offset
            dataset.uncertainties[index] = uncertainty

            max_id = max(max_id, int(labels[:, -1].max()))
            seq_prev = seq_name

        self.train_loader.dataset._dataset.save_anns_to_disk()

    def infer_pseudo_sync(self):
        if self.is_distributed:
            self.extract_pseudo_feat()
            synchronize()
            with torch_distributed_zero_first(self.rank):
                self.infer_pseudo()
                # self.infer_pseudo_by_tracker()
            synchronize()
        else:
            self.extract_pseudo_feat()
            self.infer_pseudo()
            # self.infer_pseudo_by_tracker()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets, infos = self.prefetcher.next()  # imgs and targets
        ids = targets[..., -1].clone()
        infos = (infos, ids)
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets, infos=infos)
        loss = outputs["total_loss"]

        self.optimizer.zero_grad()  # optimizer.zero_grad
        self.scaler.scale(loss).backward()  # loss.backward
        self.scaler.step(self.optimizer)  # optimizer.step
        self.scaler.update()

        if self.use_model_ema:  # ema
            self.ema_model.update(self.model)

        # lr scheduler
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs  # close aug for the last few epochs
        try:  # mot dataset with ids
            self.train_loader, settings = self.exp.get_data_loader(  # dataloader
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                no_aug=self.no_aug,
            )
            self.settings.update(settings)
        except KeyboardInterrupt:
            exit(12)
        except:  # only detection dataset
            self.train_loader = self.exp.get_data_loader(  # dataloader
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                no_aug=self.no_aug,
            )

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model(settings=self.settings)
        logger.info("Model Summary: {}".format(
            get_model_info(model, self.exp.test_size)))
        model.to(self.device)
        model.moco.encoder_k_backbone.to(self.device)
        model.moco.encoder_k_head.to(self.device)
        model.moco.queue = model.moco.queue.to(self.device)
        model.freeze_detector = self.freeze_detector  # TODO: train reid only

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        # self.optimizer.add_param_group({"params": model.head.s_det})    # TODO: Uncertainty loss (0114)
        # self.optimizer.add_param_group({"params": model.head.s_id})     # TODO: Uncertainty loss (0114)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)  # model

        # data related init
        if self.start_epoch != 0:  # with resume, reset dataset
            self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs  # close aug for the last few epochs
            try:  # mot dataset with ids
                self.train_loader, settings = self.exp.get_data_loader(  # dataloader
                    batch_size=self.args.batch_size,
                    is_distributed=self.is_distributed,
                    no_aug=self.no_aug,
                )
                self.settings.update(settings)
            except:  # only detection dataset
                self.train_loader = self.exp.get_data_loader(  # dataloader
                    batch_size=self.args.batch_size,
                    is_distributed=self.is_distributed,
                    no_aug=self.no_aug,
                )

        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter)
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model,
                        device_ids=[self.local_rank],
                        broadcast_buffers=False,
                        find_unused_parameters=True)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(  # COCOevaluator
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed)
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        # before prefetch
        self.train_loader.dataset._dataset.set_epoch(self.start_epoch)
        self.infer_loader = self.exp.get_infer_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed)
        self.infer_pseudo_sync()
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)

        logger.info("Training start...")

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                self.best_ap * 100))

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        self.exp.set_epoch(self.epoch)
        s_reid, s_moco = self.exp.get_reid_scale(), self.exp.get_moco_scale()
        if self.freeze_detector:
            s_reid = 1.
            s_bbox = 0.
        else:
            s_reid = (1. + s_reid * 1.) / 2.  # 0.5 -> 1.0
            s_bbox = 1. + s_reid
        if self.is_distributed:
            self.model.module.head.s_bbox = s_bbox
            self.model.module.head.s_reid = s_reid
            self.model.module.head.s_moco = s_moco
        else:
            self.model.head.s_bbox = s_bbox
            self.model.head.s_reid = s_reid
            self.model.head.s_moco = s_moco

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:  # check epoch to close aug

            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True

            if self.exp.eval_interval < self.max_epoch:
                self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

        self.model.train()

    def after_epoch(self, end=False):
        if self.use_model_ema:  # ema
            self.ema_model.update_attr(self.model)

        self.save_ckpt(ckpt_name="latest")  # save checkpoints
        self.save_moco()

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

        if not end:
            self.train_loader.dataset._dataset.set_epoch(self.epoch + 1)
            self.infer_pseudo_sync()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (
                self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(
                datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter)
            loss_meter = self.meter.get_filtered_meter("loss")

            # TODO: has to be modified after using uncertainty loss, but I don't know what is wrong?
            # loss_str = ", ".join(
            #     ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
            # )
            loss_str = ""
            for k, v in loss_meter.items():
                try:
                    loss_str += "{}: {:.3f} ".format(k, v.latest.item())
                except:
                    loss_str += "{}: {:.3f} ".format(k, v.latest)

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join([
                "{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()
            ])

            logger.info("{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                progress_str,
                gpu_mem_usage(),
                time_str,
                loss_str,
                self.meter["lr"].latest,
            ) + (", size: {:d}, {}".format(self.input_size[0], eta_str)))

            # TODO: log losses in tensorboard (0111)
            if self.rank == 0:  # need this line, or will 'process.wait()'
                for k, v in loss_meter.items():
                    self.tblogger.add_scalar(
                        "loss/" + k, v.latest,
                        self.epoch * self.max_iter + (self.iter + 1))

            self.meter.clear_meters()

        # random resizing
        if self.exp.random_size is not None and (self.progress_in_iter +
                                                 1) % 10 == 0:
            self.input_size = self.exp.random_resize(self.train_loader,
                                                     self.epoch, self.rank,
                                                     self.is_distributed)

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name,
                                         "latest" + "_ckpt.pth.tar")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file,
                              map_location=self.device)  # torch.load
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = (self.args.start_epoch - 1 if self.args.start_epoch
                           is not None else ckpt["start_epoch"])
            self.start_epoch = start_epoch
            logger.info("loaded checkpoint '{}' (epoch {})".format(
                self.args.resume, self.start_epoch))  # noqa

            moco_file = os.path.join(self.file_name,
                                     f"moco_{self.rank}" + "_ckpt.pth.tar")
            if os.path.exists(moco_file):
                ckpt = torch.load(moco_file, map_location=self.device)
                model.moco.encoder_k_backbone.load_state_dict(
                    ckpt["encoder_k_backbone"])
                model.moco.encoder_k_head.load_state_dict(
                    ckpt["encoder_k_head"])
                model.moco.queue = ckpt["queue"]
                model.moco.queue_ptr = ckpt["queue_ptr"].cpu()
                logger.info(f"loaded MoCo params '{self.args.resume}'")  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        ap50_95, ap50, summary = self.exp.eval(evalmodel, self.evaluator,
                                               self.is_distributed)
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95,
                                     self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        #self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", ap50 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

    def save_moco(self):
        if self.is_distributed:
            moco = self.model.module.moco
        else:
            moco = self.model.moco
        logger.info("Save MoCo params to {}".format(self.file_name))
        ckpt_state = {
            "encoder_k_backbone": moco.encoder_k_backbone.state_dict(),
            "encoder_k_head": moco.encoder_k_head.state_dict(),
            "queue": getattr(moco, 'queue').cpu(),
            "queue_ptr": getattr(moco, 'queue_ptr')
        }
        save_checkpoint(ckpt_state, False, self.file_name, f"moco_{self.rank}")