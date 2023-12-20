#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from __future__ import annotations
import cv2
import numpy as np
from pycocotools.coco import COCO

import os
import math
import random
import warnings

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class MOTDataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self,
                 data_dir=None,
                 json_file="train_half.json",
                 name="train",
                 img_size=(608, 1088),
                 preproc=None,
                 max_epoch=80,
                 is_training=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()  # image ids, not track ids
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()  # in order
        self.uncertainties = [
            np.zeros_like(ann[0][:, -1]) for ann in self.annotations
        ]
        # self.affine_cache = [None] * len(self.annotations)
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

        self.is_training = is_training
        self.seq_names = self.get_seq_names()

        self.max_epoch = max_epoch
        self.time_scale = 100
        self.cur_epoch = 0

        self.sync_dir = './ann_tmp'
        os.mkdir(self.sync_dir) if not os.path.exists(self.sync_dir) else None

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]  # format: tlbr
            res[ix, 4] = cls  # class id, 0 for person
            res[ix, 5] = obj["track_id"]  # track id

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        (min_id, max_id) = (res[:, -1].min(), res[:, -1].max()) if num_objs else (0, 0)
        seq_name = self.img_path2seq(file_name)
        img_info = (height, width, frame_id, video_id, file_name, seq_name, min_id, max_id)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def get_ann_tmp_path(self, index, prefix='_ann.txt'):
        file_name = self.annotations[index][-1]
        tail = file_name.split('.')[-1]
        _name = '_'.join(file_name.split('/'))
        _name = _name.replace('.' + tail, prefix)
        return f'{self.sync_dir}/{index:06d}_{_name}'

    def save_anns_to_disk(self):
        for i, (res, _, file_name) in enumerate(self.annotations):
            ann = np.concatenate((res, self.uncertainties[i].reshape(-1, 1)), axis=1)
            save_path = self.get_ann_tmp_path(i, '_ann.txt')
            np.savetxt(save_path, ann, fmt="%.2f %.2f %.2f %.2f %d %d %.3f")

    def load_anns_to_mem(self):
        for i, (res, _, file_name) in enumerate(self.annotations):
            save_path = self.get_ann_tmp_path(i, '_ann.txt')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ann = np.loadtxt(save_path, delimiter=' ').reshape(-1, 7)
            res[:, -1] = ann[:, -2]
            assert (self.annotations[i][0][:, -1] == ann[:, -2]).all()
            # if 'eth' in file_name:
            #     assert (self.annotations[i][0][:, -1] == -1).sum() == 0
            self.uncertainties[i] = ann[:, -1]

    @staticmethod
    def img_path2seq(file_name):
        sep_dict = {
            'VisDrone': '_v/',
            'MOT': 'img1',
            'crowdhuman': ',',
            'eth': 'images',
            'cp': '_0'
        }
        for vid, sep in sep_dict.items():
            if vid in file_name:
                return file_name.split(sep)[0].split('.')[-1]
        if len(os.path.basename(file_name)[:-4].split('-')) == 3:  # bdd100k, stupid
            return os.path.dirname(file_name).split('.')[-1]
        raise ValueError('Unknown dataset: ', file_name)

    @staticmethod
    def check_period(file_name, period):
        if any(d in file_name for d in ['cp_train', 'crowdhuman']):
            return 0
        if any(d in file_name for d in ['ethz_train', 'mot_train', 'mot20_train', 'VisDrone', 'MOT17', 'MOT20']):
            return period
        if len(os.path.basename(file_name)[:-4].split('-')) == 3:  # bdd100k, stupid
            return period
        raise ValueError('Undefined dataset:', file_name)

    @staticmethod
    def get_frame_cnt(file_name):
        if 'eth' in file_name:
            return int(os.path.basename(file_name).split('_')[1]) + 1
        if 'MOT' in file_name:
            return int(os.path.basename(file_name).split('.')[0])
        if len(os.path.basename(file_name)[:-4].split('-')) == 3:  # bdd100k, stupid
            return int(os.path.basename(file_name)[:-4].split('-')[-1])
        return 1  # equals to the 1st frame

    def seq_repeated(self, indices):
        valid_seqs = []
        for index in indices:
            file_name, seq_name = self.annotations[index][1][-4:-2]
            if not self.is_img_static(file_name):
                valid_seqs.append(seq_name)
        return len(valid_seqs) != len(set(valid_seqs))

    def is_img_static(self, file_name):
        return self.check_period(file_name, 2) < 2

    def get_time_scale(self):
        return int(math.pow(self.cur_epoch / self.max_epoch, 3) * self.time_scale + 1)

    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def pull_item(self, index, id_off=-1):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        assert img is not None, img_file

        if not self.is_training:
            return img, res.copy(), img_info, np.array([id_])
        else:
            # load auxilliary data
            frame_cnt = self.get_frame_cnt(file_name)
            time_scale = 0 if self.is_img_static(file_name) else self.get_time_scale()
            # time_scale = self.time_scale  # TODO: only for contiguous sequences with detector freezed
            min_delta = min(1, time_scale)  # 1 or 0
            delta = random.randint(min(min_delta, frame_cnt - 1),
                                   min(time_scale, frame_cnt - 1))
            index_prev = index - delta

            res_prev, img_info_prev, file_name_prev = self.annotations[index_prev]
            if self.img_path2seq(file_name) != self.img_path2seq(file_name_prev):  # stupid ETHZ-named paths
                index_prev = index
                res_prev, img_info_prev, file_name_prev = self.annotations[index_prev]

            img_file_prev = os.path.join(self.data_dir, self.name, file_name_prev)
            img_prev = cv2.imread(img_file_prev)
            assert img_prev is not None, img_file_prev

            res_prev = res_prev.copy()
            res = res.copy()
            if self.is_img_static(file_name):  # compatible for Mosaic and Mixup
                res[:, -1] = -np.arange(len(res)) + id_off
                res_prev[:, -1] = -np.arange(len(res)) + id_off
            if len(res) * len(res_prev) > 0:
                img_info = (*img_info[:-2],
                            min(res[:, -1].min(), res_prev[:, -1].min()),
                            max(res[:, -1].max(), res_prev[:, -1].max()))
            else:
                img_info = (*img_info[:-2], 0, 0)
            img_aug, res_aug = self.tracklet_guided_augment(img, res.copy(), index, index_prev)
            # img_aug, res_aug = img.copy(), res.copy()

            return (img, img_aug, img_prev), (res, res_aug, res_prev), img_info, np.array([id_])  #(img_info, img_info_prev), np.array([id_, id_prev])

    def tracklet_guided_augment(self, img, curr_label, curr_index, prev_index):  # tracklet-guided augmentation
        import os.path as osp
        from ..data_augment import transform_labels

        def _get_affine(src, dst):
            affine = cv2.getAffineTransform(src.astype(np.float32),
                                            dst.astype(np.float32))
            ones = np.array([[0, 0, 1]])
            # print(np.stack((dst.T, affine @ (np.concatenate((src, np.ones((3,1))), axis=1)).T)), '\n\n')
            return np.concatenate((affine, ones), axis=0)

        def _spread_uncertainty(start, end, ids, reverse=True):
            uncertainty = []
            for idx in range(start, end + 1):
                total_ids = self.annotations[idx][0][:, -1].astype(np.int32)
                curr_uncertain = np.zeros((len(ids), ))
                unce_ind, curr_ind = np.nonzero(np.atleast_1d(ids == total_ids.reshape(1, -1)))
                curr_uncertain[unce_ind] = self.uncertainties[idx][curr_ind].copy()
                uncertainty.append(curr_uncertain)

            ### geometric mean
            # uncertainty = np.power(np.prod(np.array(uncertainty), axis=0), 1. / len(uncertainty))
            # uncertainty = np.exp(np.log(np.stack(uncertainty) + 1e-6).mean(axis=0))
            # assert np.isnan(uncertainty).sum() == 0
            ### arithmetic mean
            # uncertainty = sum(uncertainty) / len(uncertainty)
            uncertainty = sum(uncertainty)

            # reverse (select certain samples)
            if reverse:
                uncertainty = -uncertainty

            # softmax
            uncertainty = np.exp(uncertainty) / np.sum(np.exp(uncertainty))

            return uncertainty

        def _ada_target(start, end, anchor_id):
            uncertainty, xyxy = [], []
            for idx in range(start, end + 1):
                label = self.annotations[idx][0]
                total_ids = label[:, -1].astype(np.int32)
                curr_ind = np.nonzero(total_ids == anchor_id)[0]
                if len(curr_ind) > 0:
                    curr_ind = int(curr_ind)
                    uncertainty.append(self.uncertainties[idx][curr_ind])
                    xyxy.append(label[curr_ind:curr_ind + 1, :4])
                else:
                    uncertainty.append(np.array(-np.inf))
                    xyxy.append(label[:1, :4])

            uncertainty = np.array(uncertainty)
            # softmax
            uncertainty = np.exp(uncertainty) / np.sum(np.exp(uncertainty))
            # higher uncertainty means higher probability to choose
            anchor = np.random.choice(np.arange(len(uncertainty)), 1, replace=False, p=uncertainty)[0]
            anchor_index = start + anchor

            prev_xyxy = xyxy[anchor]
            dst = np.stack((prev_xyxy[:, [0, -1]], prev_xyxy[:, :2], prev_xyxy[:, 2:]))

            return anchor_index, dst

        if curr_index == prev_index or \
            any(len(self.annotations[idx][0]) == 0 for idx in [curr_index, prev_index]):
            return img.copy(), curr_label

        prev_label = self.annotations[prev_index][0].copy()
        curr_id, prev_id = curr_label[:, -1].astype(np.int32), prev_label[:, -1].astype(np.int32)
        assign = curr_id.reshape((-1, 1)) == prev_id.reshape((1, -1))
        i, j = np.nonzero(np.atleast_1d(assign))
        assert len(i) == len(np.unique(i)) == len(np.unique(j)) == len(j)
        n_pair = len(i)
        if n_pair == 0:
            return img.copy(), curr_label
        curr_xyxy = curr_label[i, :4]
        # tracklet uncertainty
        uncertainty = _spread_uncertainty(prev_index, curr_index, curr_id[i].reshape(-1, 1))
        anchor = np.random.choice(np.arange(n_pair), 1, replace=False, p=uncertainty)[0]
        src = np.stack((curr_xyxy[anchor, [0, -1]], curr_xyxy[anchor, :2], curr_xyxy[anchor, 2:]))

        prev_index, dst = _ada_target(prev_index, curr_index, curr_id[i][anchor].astype(np.int32))
        M = _get_affine(src, dst)  # x1,y2,x1,y1,x2,y2
        # self.affine_cache[curr_index] = M.copy()

        height, width, _ = img.shape
        dsize = (width, height)
        aug_img = cv2.warpAffine(img, M[:2], dsize=dsize, borderValue=(114, 114, 114))
        aug_ann = transform_labels(curr_label, M, dsize, perspective=False, s=(M[0, 0] + M[1, 1]) / 2.)

        return aug_img, aug_ann

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id

    def get_seq_names(self):
        seq_names = []
        for annotation in self.annotations:
            seq_name = self.img_path2seq(annotation[-1])
            if seq_name not in seq_names:
                seq_names.append(seq_name)

        return seq_names