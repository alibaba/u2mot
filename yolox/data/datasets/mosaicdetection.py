#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np

from yolox.utils import adjust_box_anns

import random

from ..data_augment import box_candidates, random_perspective, augment_hsv
from .datasets_wrapper import Dataset


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
        shear=2.0, perspective=0.0, enable_mixup=True, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup

    def __len__(self):
        return len(self._dataset)
    
    @staticmethod
    def _merge_mosaic_labels(mosaic_labels, input_w, input_h):
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            mosaic_labels = mosaic_labels[mosaic_labels[:, 0] < 2 * input_w]
            mosaic_labels = mosaic_labels[mosaic_labels[:, 2] > 0]
            mosaic_labels = mosaic_labels[mosaic_labels[:, 1] < 2 * input_h]
            mosaic_labels = mosaic_labels[mosaic_labels[:, 3] > 0]

            _clip = False
            if _clip:  # TODO: clip boxes in MOT20
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
            
            return mosaic_labels
        else:
            return np.zeros([0, 6])

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic:
            mosaic_labels, mosaic_labels_aug, mosaic_labels_prev = [], [], []
            image_infos = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices, ensure that no more two images come from a same continuous sequence
            while True:
                indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]
                if not self._dataset.seq_repeated(indices):
                    break

            id_off = -100  # ugly
            for i_mosaic, index in enumerate(indices):
                # _labels format: tlbr(absolute value) + class_id + track_id
                (img, img_aug, img_prev), (_labels, _labels_aug, _labels_prev), img_info, _ = self._dataset.pull_item(index, id_off)
                id_off = min(id_off, img_info[-2]) - 2  # stupid FP16
                image_infos.append(img_info)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                img_aug = cv2.resize(img_aug, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                img_prev = cv2.resize(img_prev, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                    mosaic_img_aug = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                    mosaic_img_prev = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                mosaic_img_aug[l_y1:l_y2, l_x1:l_x2] = img_aug[s_y1:s_y2, s_x1:s_x2]
                mosaic_img_prev[l_y1:l_y2, l_x1:l_x2] = img_prev[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels, labels_aug, labels_prev = _labels.copy(), _labels_aug.copy(), _labels_prev.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                if _labels_aug.size > 0:
                    labels_aug[:, 0] = scale * _labels_aug[:, 0] + padw
                    labels_aug[:, 1] = scale * _labels_aug[:, 1] + padh
                    labels_aug[:, 2] = scale * _labels_aug[:, 2] + padw
                    labels_aug[:, 3] = scale * _labels_aug[:, 3] + padh
                if _labels_prev.size > 0:
                    labels_prev[:, 0] = scale * _labels_prev[:, 0] + padw
                    labels_prev[:, 1] = scale * _labels_prev[:, 1] + padh
                    labels_prev[:, 2] = scale * _labels_prev[:, 2] + padw
                    labels_prev[:, 3] = scale * _labels_prev[:, 3] + padh
                mosaic_labels.append(labels)
                mosaic_labels_aug.append(labels.copy())
                mosaic_labels_prev.append(labels_prev)

            mosaic_labels, mosaic_labels_aug, mosaic_labels_prev = \
                [self._merge_mosaic_labels(l, input_w, input_h) \
                    for l in [mosaic_labels, mosaic_labels_aug, mosaic_labels_prev]]
                
            #augment_hsv(mosaic_img)
            mosaic_img, mosaic_labels, aug_corpus = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove
            mosaic_img_aug, mosaic_labels_aug, _ = random_perspective(
                mosaic_img_aug,
                mosaic_labels_aug,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
                aug_corpus=aug_corpus
            )  # border to remove
            mosaic_img_prev, mosaic_labels_prev, _ = random_perspective(
                mosaic_img_prev,
                mosaic_labels_prev,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
                aug_corpus=aug_corpus
            )  # border to remove

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            mosaic_img_tuple = (mosaic_img, mosaic_img_aug, mosaic_img_prev)
            mosaic_labels_tuple = (mosaic_labels, mosaic_labels_aug, mosaic_labels_prev)
            if self.enable_mixup and not any(len(l) == 0 for l in mosaic_labels_tuple):
                mosaic_img_tuple, mosaic_labels_tuple, cp_img_info = \
                    self.mixup(mosaic_img_tuple, mosaic_labels_tuple, indices, id_off, self.input_dim)
                image_infos = (*image_infos, cp_img_info)
            
            # tlbr, class_id, track_id --> class_id, xywh, track_id
            # mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            # img_info = (mix_img.shape[1], mix_img.shape[0])
            flipped = None
            mosaic_img_list, mosaic_labels_list = [], []
            for mosaic_img, mosaic_labels in zip(mosaic_img_tuple, mosaic_labels_tuple):
                mix_img, padded_labels, flipped = self.preproc(mosaic_img, mosaic_labels, self.input_dim, flipped=flipped)
                mosaic_img_list.append(mix_img)
                mosaic_labels_list.append(padded_labels)
            return np.stack(mosaic_img_list), np.stack(mosaic_labels_list), image_infos, np.array([idx])

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, id_ = self._dataset.pull_item(idx)
            
            flipped = None
            plain_img_list, plain_labels_list = [], []
            for plain_img, plain_labels in zip(img, label):
                mix_img, padded_labels, flipped = self.preproc(plain_img, plain_labels, self.input_dim, flipped=flipped)
                plain_img_list.append(mix_img)
                plain_labels_list.append(padded_labels)

            return np.stack(plain_img_list), np.stack(plain_labels_list), [img_info], id_

    @staticmethod
    def _merge_mixup_labels(origin_labels, cp_labels, cp_bboxes_transformed_np, keep_list, 
                            origin_img, padded_cropped_img, target_w, target_h):
        cls_labels = cp_labels[keep_list, 4:5].copy()
        id_labels = cp_labels[keep_list, 5:6].copy()
        box_labels = cp_bboxes_transformed_np[keep_list]
        labels = np.hstack((box_labels, cls_labels, id_labels))
        # remove outside bbox
        labels = labels[labels[:, 0] < target_w]
        labels = labels[labels[:, 2] > 0]
        labels = labels[labels[:, 1] < target_h]
        labels = labels[labels[:, 3] > 0]
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img, origin_labels

    def mixup(self, origin_img_tuple, origin_labels_tuple, origin_indices, id_off, input_dim):
        def _resize_img(img, cp_img, cp_scale_ratio, jit_factor):
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            cp_img[
                : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
            ] = resized_img
            cp_img = cv2.resize(
                cp_img,
                (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
            )
            return cp_img
        
        def _pad_img(cp_img, origin_w, origin_h, target_w, target_h):
            padded_img = np.zeros(
                (max(origin_h, target_h), max(origin_w, target_w), 3)
            ).astype(np.uint8)
            padded_img[:origin_h, :origin_w] = cp_img

            return padded_img

        origin_img, origin_img_aug, origin_img_prev = origin_img_tuple
        origin_labels, origin_labels_aug, origin_labels_prev = origin_labels_tuple

        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while True:
            cp_index = random.randint(0, self.__len__() - 1)
            if self._dataset.seq_repeated(origin_indices + [cp_index]):
                continue
            cp_labels = self._dataset.load_anno(cp_index)
            if len(cp_labels) > 0:
                break
        (img, img_aug, img_prev), (cp_labels, cp_labels_aug, cp_labels_prev), cp_img_info, _ = \
            self._dataset.pull_item(cp_index, id_off)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
            cp_img_aug = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
            cp_img_prev = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
        else:
            cp_img = np.ones(input_dim) * 114.0
            cp_img_aug = np.ones(input_dim) * 114.0
            cp_img_prev = np.ones(input_dim) * 114.0
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])

        cp_img = _resize_img(img, cp_img, cp_scale_ratio, jit_factor)
        cp_img_aug = _resize_img(img_aug, cp_img_aug, cp_scale_ratio, jit_factor)
        cp_img_prev = _resize_img(img_prev, cp_img_prev, cp_scale_ratio, jit_factor)

        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]
            cp_img_aug = cp_img_aug[:, ::-1, :]
            cp_img_prev = cp_img_prev[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = _pad_img(cp_img, origin_w, origin_h, target_w, target_h)
        padded_img_aug = _pad_img(cp_img_aug, origin_w, origin_h, target_w, target_h)
        padded_img_prev = _pad_img(cp_img_prev, origin_w, origin_h, target_w, target_h)

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]
        padded_cropped_img_aug = padded_img_aug[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]
        padded_cropped_img_prev = padded_img_prev[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        cp_bboxes_origin_np_aug = adjust_box_anns(
            cp_labels_aug[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        cp_bboxes_origin_np_prev = adjust_box_anns(
            cp_labels_prev[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, [0,2]] = (
                origin_w - cp_bboxes_origin_np[:, [2,0]]
            )
            cp_bboxes_origin_np_aug[:, [0,2]] = (
                origin_w - cp_bboxes_origin_np_aug[:, [2,0]]
            )
            cp_bboxes_origin_np_prev[:, [0,2]] = (
                origin_w - cp_bboxes_origin_np_prev[:, [2,0]]
            )
        
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np_aug = cp_bboxes_origin_np_aug.copy()
        cp_bboxes_transformed_np_prev = cp_bboxes_origin_np_prev.copy()
        _clip = False
        if _clip:  # TODO: clip boxes in MOT20
            cp_bboxes_transformed_np[:, 0::2] = np.clip(
                cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
            )
            cp_bboxes_transformed_np[:, 1::2] = np.clip(
                cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
            )
            cp_bboxes_transformed_np_aug[:, 0::2] = np.clip(
                cp_bboxes_transformed_np_aug[:, 0::2] - x_offset, 0, target_w
            )
            cp_bboxes_transformed_np_aug[:, 1::2] = np.clip(
                cp_bboxes_transformed_np_aug[:, 1::2] - y_offset, 0, target_h
            )
            cp_bboxes_transformed_np_prev[:, 0::2] = np.clip(
                cp_bboxes_transformed_np_prev[:, 0::2] - x_offset, 0, target_w
            )
            cp_bboxes_transformed_np_prev[:, 1::2] = np.clip(
                cp_bboxes_transformed_np_prev[:, 1::2] - y_offset, 0, target_h
            )
        else:
            cp_bboxes_transformed_np[:, 0::2] = cp_bboxes_transformed_np[:, 0::2] - x_offset
            cp_bboxes_transformed_np[:, 1::2] = cp_bboxes_transformed_np[:, 1::2] - y_offset
            cp_bboxes_transformed_np_aug[:, 0::2] = cp_bboxes_transformed_np_aug[:, 0::2] - x_offset
            cp_bboxes_transformed_np_aug[:, 1::2] = cp_bboxes_transformed_np_aug[:, 1::2] - y_offset
            cp_bboxes_transformed_np_prev[:, 0::2] = cp_bboxes_transformed_np_prev[:, 0::2] - x_offset
            cp_bboxes_transformed_np_prev[:, 1::2] = cp_bboxes_transformed_np_prev[:, 1::2] - y_offset
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)
        keep_list_aug = box_candidates(cp_bboxes_origin_np_aug.T, cp_bboxes_transformed_np_aug.T, 5)
        keep_list_prev = box_candidates(cp_bboxes_origin_np_prev.T, cp_bboxes_transformed_np_prev.T, 5)

        if keep_list.sum() * keep_list_aug.sum() * keep_list_prev.sum() > 0:
            origin_img, origin_labels = \
                self._merge_mixup_labels(origin_labels, cp_labels, cp_bboxes_transformed_np, keep_list, 
                                         origin_img, padded_cropped_img, target_w, target_h)
            origin_img_aug, origin_labels_aug = \
                self._merge_mixup_labels(origin_labels_aug, cp_labels_aug, cp_bboxes_transformed_np_aug, keep_list_aug, 
                                         origin_img_aug, padded_cropped_img_aug, target_w, target_h)
            origin_img_prev, origin_labels_prev = \
                self._merge_mixup_labels(origin_labels_prev, cp_labels_prev, cp_bboxes_transformed_np_prev, keep_list_prev, 
                                         origin_img_prev, padded_cropped_img_prev, target_w, target_h)

        # return origin_img, origin_labels
        return (origin_img, origin_img_aug, origin_img_prev), \
               (origin_labels, origin_labels_aug, origin_labels_prev), cp_img_info