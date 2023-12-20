#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
# Copyright (c) Alibaba, Inc. and its affiliates.

"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import cv2
import numpy as np

import torch

from yolox.utils import xyxy2cxcywh

import math
import random


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
    aug_corpus=None
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    if aug_corpus is None:
        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(scale[0], scale[1])
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S01 = random.uniform(-shear, shear) * math.pi / 180
        S10 = random.uniform(-shear, shear) * math.pi / 180
        S[0, 1] = math.tan(S01)  # x shear (deg)
        S[1, 0] = math.tan(S10)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T02 = random.uniform(0.5 - translate, 0.5 + translate) * width
        T12 = random.uniform(0.5 - translate, 0.5 + translate) * height
        T[0, 2] = T02  # x translation (pixels)
        T[1, 2] = T12  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

        aug_corpus = {
            'angle': a,
            'scale': s,
            'translation02': T02,
            'translation12': T12,
            'shear01': S01,
            'shear10': S10
        }
    else:
        # get warp matrix for auxiliary frames
        disturb = random.choice(range(5, 20, 1))
        disturb = disturb / 100 #* 0 + 2.  # TODO: ablation

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees) * disturb + aug_corpus['angle']
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = (1 + random.uniform(scale[0]-1, scale[1]-1) * disturb) * aug_corpus['scale']
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S01 = (random.uniform(-shear, shear) * disturb) * math.pi / 180 + aug_corpus['shear01']
        S10 = (random.uniform(-shear, shear) * disturb) * math.pi / 180 + aug_corpus['shear10']
        S[0, 1] = math.tan(S01)  # x shear (deg)
        S[1, 0] = math.tan(S10)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width * disturb + aug_corpus['translation02']  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height * disturb + aug_corpus['translation12']  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    targets = transform_labels(targets, M, (width, height), perspective, s)
        
    return img, targets, aug_corpus


def transform_labels(targets, M, dsize, perspective=True, s=1.0):
    n = len(targets)
    width, height = dsize
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        _clip = False
        if _clip:  # TODO: clip boxes in MOT20
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]
        
        targets = targets[targets[:, 0] < width]
        targets = targets[targets[:, 2] > 0]
        targets = targets[targets[:, 1] < height]
        targets = targets[targets[:, 3] > 0]
    
    return targets


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _mirror(image, boxes, flipped=None):
    _, width, _ = image.shape
    if (flipped is None and random.randrange(2)) or (flipped is True):
        flipped = True
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    else:
        flipped = False
    return image, boxes, flipped


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, p=0.5, rgb_means=None, std=None, max_labels=100):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels

    def __call__(self, image, targets, input_dim, flipped=None):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        ids = targets[:, 5].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 6), dtype=np.float64)
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets, False

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        ids_o = targets_o[:, 5]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        image_t = _distort(image)
        image_t, boxes, flipped = _mirror(image_t, boxes, flipped=flipped)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim, self.means, self.std)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]
        ids_t = ids[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o
            ids_t = ids_o

        labels_t = np.expand_dims(labels_t, 1)
        ids_t = np.expand_dims(ids_t, 1)

        targets_t = np.hstack((labels_t, boxes_t, ids_t))       # get new labels(targets), format: class_id, bbox(xywh), track_id
        padded_labels = np.zeros((self.max_labels, 6))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float64)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels, flipped


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1), get_tgt=False):
        self.means = rgb_means
        self.swap = swap
        self.std = std
        self.get_tgt = get_tgt

    # assume input is cv2 img for now
    def __call__(self, img, targets, input_size):
        img, r = preproc(img, input_size, self.means, self.std, self.swap)

        if not self.get_tgt:
            return img, np.zeros((1, 5))
        else:
            boxes = targets[:, :4].copy()
            labels = targets[:, 4:5].copy()
            ids = targets[:, 5:6].copy()
            if len(boxes) == 0:
                targets = [np.zeros((0, 6), dtype=np.float32)] #.tolist()
            else:
                boxes = xyxy2cxcywh(boxes) * r
                # get new labels(targets), format: class_id, bbox(xywh), track_id
                targets = np.concatenate((labels, boxes, ids), axis=1)#.tolist()
                targets = np.split(targets, len(targets), axis=0)
            
            return img, targets
