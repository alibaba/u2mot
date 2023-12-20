#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from tqdm import tqdm
from glob import glob
import os
import os.path as osp
import json

cls2id = {1: 'pedestrian', 2: 'rider', 3: 'car', 4: 'truck', 5: 'bus', 
          6: 'train', 7: 'motorcycle', 8: 'bicycle', 
        #   9: 'traffic light', 10: 'traffic sign'
          }
id2cls = {v: k for k, v in cls2id.items()}


def txt2json(split=True):
    if not split:  # split results from different videos
        data = []

    output_dir = 'YOLOX_outputs/yolox_x_u2mot_bdd100k'
    for txt_file in tqdm(sorted(glob(f'{output_dir}/track_res/*.txt')), 'Convert'):  # track_res
        videoName = osp.basename(txt_file)[:-4]
        frameNum = len(glob(f'datasets/BDD100K-MOT/images/*/{videoName}/*.jpg'))
        preds = np.loadtxt(txt_file, delimiter=',').reshape(-1, 10)
        if split:
            data = []
        for frameIndex in range(1, frameNum+1):
            name = f'{videoName}/{videoName}-{frameIndex:07d}.jpg'
            frame_item = {'videoName': videoName, 'name': name, 'frameIndex': frameIndex-1, "labels": []}

            for _, obj_id, x1, y1, w, h, score, cls, _, _ in preds[preds[:, 0].astype(np.int32) == frameIndex]:
                obj = {'id': int(obj_id), 'category': cls2id[int(cls)], 'box2d': {'x1': x1, 'y1': y1, 'x2': x1+w, 'y2': y1+h}}
                frame_item['labels'].append(obj)

            data.append(frame_item)

        if split:
            save_dir = f'{output_dir}/track_res_json'
            os.makedirs(save_dir, exist_ok=True)
            with open(f'{save_dir}/{videoName}.json', 'w+') as f:
                json.dump(data, f)
    if not split:
        with open(f'{output_dir}/track_res.json', 'w+') as f:
            json.dump(data, f)


if __name__ == '__main__':
    txt2json()