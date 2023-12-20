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


DATA_PATH = 'datasets/VisDrone-MOT'
OUT_PATH = DATA_PATH+'/annotations'
SPLITS = ['train', 'val', 'trainval', 'test']
HALF_VIDEO = True               # half video
CREATE_SPLITTED_ANN = True      # create splitted ann
CREATE_SPLITTED_DET = True      # create splitted det
cls2id = {1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van', 6: 'truck',
          7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'}
split2paths = {'train': ['VisDrone2019-MOT-train'], 'val': ['VisDrone2019-MOT-val'], 
               'test': ['VisDrone2019-MOT-test-dev'], 
               'trainval': ['VisDrone2019-MOT-train', 'VisDrone2019-MOT-val']}

def erase_ignore():
    gray = 114
    for split in ['train', 'val', 'test']:
        data_path = split2paths[split][0]
        sequences = sorted(os.listdir(osp.join(DATA_PATH, data_path, 'sequences')))
        for seq in tqdm(sequences, desc=f'{split}: {data_path}'):
            ann_path = osp.join(DATA_PATH, data_path, 'annotations', seq+'.txt')
            ignore_dict = {}
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
            for frame_id, _, x1, y1, w, h, _, cat_id, *_ in anns.astype(np.int32):
                if cat_id not in [0, 11]:
                    continue
                if frame_id not in ignore_dict:
                    ignore_dict[frame_id] = []
                ignore_dict[frame_id].append([x1, y1, x1+w, y1+h])
            
            img_dir = osp.join(DATA_PATH, data_path, 'sequences', seq)
            for frame_id, regions in ignore_dict.items():
                img_path = osp.join(img_dir, '%07d.jpg' % frame_id)
                img = cv2.imread(img_path)
                for x1, y1, x2, y2 in regions:
                    img[y1:y2, x1:x2, :] = gray
                cv2.imwrite(img_path, img)


if __name__ == '__main__':
    erase_ignore()

    if not os.path.exists(OUT_PATH):    # check output path
        os.makedirs(OUT_PATH)

    for split in SPLITS:        # iteration over split strategy
        data_paths = split2paths[split]
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': k, 'name': v} for k, v in cls2id.items()]}
        
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        for data_path in data_paths:
            sequences = sorted(os.listdir(osp.join(DATA_PATH, data_path, 'sequences')))
            for seq in tqdm(sequences, desc=f'{split}: {data_path}'):
                video_cnt += 1  # video sequence number.
                out['videos'].append({'id': video_cnt, 'file_name': seq})
                img_dir = osp.join(DATA_PATH, data_path, 'sequences', seq)
                ann_path = osp.join(DATA_PATH, data_path, 'annotations', seq+'.txt')
                images = sorted(glob(osp.join(img_dir, '*.jpg')))
                num_images = len(images)

                for i, img_path in enumerate(images):
                    img = cv2.imread(img_path)
                    height, width = img.shape[:2]
                    image_info = {'file_name': img_path.split(DATA_PATH+'/')[1],  # image name.
                                'id': image_cnt + i + 1,  # image number in the entire training set.
                                'frame_id': i + 1,  # image number in the video sequence, starting from 1.
                                'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                                'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                                'video_id': video_cnt,
                                'height': height, 'width': width}
                    out['images'].append(image_info)
                # print('{}: {} images'.format(seq, num_images), end=' ')
                
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                # print('{} ann images'.format(int(anns[:, 0].max())))
                tid_last = -1
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    if cat_id in [0, 11]:
                        continue
                    if not track_id == tid_last:
                        tid_curr += 1
                        tid_last = track_id
                        
                    ann = {'id': ann_cnt,
                        'category_id': cat_id,
                        'image_id': image_cnt + frame_id,
                        'track_id': tid_curr,
                        'bbox': anns[i][2:6].tolist(),
                        'conf': float(anns[i][6]),
                        'iscrowd': 0,
                        'area': float(anns[i][4] * anns[i][5])}
                    out['annotations'].append(ann)
                image_cnt += num_images
                # print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))