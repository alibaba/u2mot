#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np

import torch
import torchvision
import torch.nn.functional as F

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "bboxes_iou_np",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy"
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    box_corner = prediction.new(prediction.shape)       # [batchsize, all_anchors, 6]
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):     # image_pred: [all_anchors, 6]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()    # filter with confidence threshold
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)  # [all_anchors, 7], bbox + obj + class_conf + clsss

        detections = detections[conf_mask]      # get detections whose conf is higher than threshold
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]      # detections after NMS
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def bboxes_iou_np(boxa, boxb, xyxy=False, giou=False):
    # bbox: [n, 4]
    if not xyxy:
        xca, yca, wa, ha = np.split(boxa, 4, axis=1)  # shape(n, 1)
        x1a, x2a, y1a, y2a = xca - wa / 2., xca + wa / 2., yca - ha / 2., yca + ha / 2.
        xcb, ycb, wb, hb = np.split(boxb.T, 4, axis=0)  # shape(1, m)
        x1b, x2b, y1b, y2b = xcb - wb / 2., xcb + wb / 2., ycb - hb / 2., ycb + hb / 2.
    else:
        x1a, y1a, x2a, y2a = np.split(boxa, 4, axis=1)  # shape(n, 1)
        wa, ha = x2a - x1a, y2a - y1a
        x1b, y1b, x2b, y2b = np.split(boxb.T, 4, axis=0)  # shape(1, m)
        wb, hb = x2b - x1b, y2b - y1b
    
    x1i, x2i, y1i, y2i = np.maximum(x1a, x1b), np.minimum(x2a, x2b), \
                         np.maximum(y1a, y1b), np.minimum(y2a, y2b) 
    intersection = (x2i - x1i).clip(0) * (y2i - y1i).clip(0)
    union = wa * ha + wb * hb - intersection
    iou = intersection / (union + 1e-6)
    # assert (iou <= 1.0).all() & (iou >= 0.0).all()
    # print(boxa.shape, boxb.shape, intersection.shape, union.shape, iou.shape)
    if not giou:
        return iou

    x1c, x2c, y1c, y2c = np.minimum(x1a, x1b), np.maximum(x2a, x2b), \
                         np.minimum(y1a, y1b), np.maximum(y2a, y2b) 
    convex = (x2c - x1c) * (y2c - y1c)
    return iou - (convex - union) / (convex + 1e-6)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    _clip = False
    if _clip:  # TODO: clip boxes in MOT20
        bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    else:
        bbox[:, 0::2] = bbox[:, 0::2] * scale_ratio + padw
        bbox[:, 1::2] = bbox[:, 1::2] * scale_ratio + padh
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def cxcywh2xyxy(bboxes):
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    bboxes[:, 0] = bboxes[:, 0] - (bboxes[:, 2] - bboxes[:, 0])
    bboxes[:, 1] = bboxes[:, 1] - (bboxes[:, 3] - bboxes[:, 1])
    return bboxes