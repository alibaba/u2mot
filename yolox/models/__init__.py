#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.

from .darknet import CSPDarknet, Darknet
from .losses import IOUloss
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
from .moco import MoCo