#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import numpy as np
import json
import cv2


DATA_PATH = 'datasets/BDD100K-MOT'
OUT_PATH = DATA_PATH+'/annotations'
SPLITS = ['train', 'val']
# For MOT and MOTS, only the first 8 classes are used and evaluated
cls2id = {1: 'pedestrian', 2: 'rider', 3: 'car', 4: 'truck', 5: 'bus', 
          6: 'train', 7: 'motorcycle', 8: 'bicycle', 
        #   9: 'traffic light', 10: 'traffic sign'
          }
cats = list(cls2id.values())

if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):    # check output path
        os.makedirs(OUT_PATH)

    for split in SPLITS:        # iteration over split strategy
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': k, 'name': v} for k, v in cls2id.items()]}
        
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        cls_cnt = {k: 0 for k in cls2id.keys()}
        for vid_path in tqdm(glob(osp.join(DATA_PATH, 'images', split, '*-*')), desc=split):
            video_cnt += 1  # video sequence number.
            vid_name = osp.basename(vid_path)
            out['videos'].append({'id': video_cnt, 'file_name': vid_name})

            img_paths = sorted(glob(osp.join(vid_path, '*.jpg')))
            num_images = len(img_paths)
            for i, img_path in enumerate(img_paths):
                image_info = {'file_name': img_path.split(DATA_PATH+'/')[1],  # image name.
                            'id': image_cnt + i + 1,  # image number in the entire training set.
                            'frame_id': i + 1,  # image number in the video sequence, starting from 1.
                            'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                            'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                            'video_id': video_cnt,
                            'height': 720, 'width': 1280}
                out['images'].append(image_info)

            label_file = osp.join(osp.join(DATA_PATH, 'labels', split, f'{vid_name}.json'))
            with open(label_file, 'r') as f:
                labels = json.load(f)
            tid_last = -1
            for frame_item in labels:
                assert vid_name == frame_item['videoName']
                frame_id = frame_item['frameIndex'] + 1
                for obj in frame_item['labels']:
                    if obj['category'] not in cats:
                        continue
                    cat_id = cats.index(obj['category']) + 1
                    cls_cnt[cat_id] += 1
                    track_id = int(obj['id'])
                    ann_cnt += 1
                    if not track_id == tid_last:
                        tid_curr += 1
                        tid_last = track_id

                    x1, y1, x2, y2 = obj['box2d']['x1'], obj['box2d']['y1'], \
                                     obj['box2d']['x2'], obj['box2d']['y2'], 
                    ann = {'id': ann_cnt,
                        'category_id': cat_id,
                        'image_id': image_cnt + frame_id,
                        'track_id': tid_curr,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'conf': 1.,
                        'iscrowd': int(obj['attributes']['crowd']),
                        'area': (x2 - x1) * (y2 - y1)}
                    
                    out['annotations'].append(ann)
            image_cnt += num_images
            # print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))