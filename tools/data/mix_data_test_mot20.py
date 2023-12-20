#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import os
"""
cd datasets
mkdir -p mix_mot20_ch/annotations
cp MOT20/annotations/val_half.json mix_mot20_ch/annotations/val_half.json
cp MOT20/annotations/test.json mix_mot20_ch/annotations/test.json
cd mix_mot20_ch
ln -s ../MOT20/images/train mot20_train
ln -s ../CrowdHuman/images crowdhuman
cd ../..
"""

json_tail = '.json'

mot_json = json.load(open('datasets/MOT20/annotations/train' + json_tail, 'r'))

img_list = list()
for img in mot_json['images']:
    img['file_name'] = 'mot20_train/' + img['file_name']
    img_list.append(img)

ann_list = list()
for ann in mot_json['annotations']:
    ann_list.append(ann)

video_list = mot_json['videos']
category_list = mot_json['categories']

max_img = 10000
max_ann = 2000000
max_video = 10

crowdhuman_json = json.load(
    open('datasets/CrowdHuman/annotations/train' + json_tail, 'r'))
img_id_count = 0
for img in crowdhuman_json['images']:
    img_id_count += 1
    img['file_name'] = 'crowdhuman/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)

for ann in crowdhuman_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

video_list.append({'id': max_video, 'file_name': 'crowdhuman_train'})

max_img = 30000
max_ann = 10000000

crowdhuman_val_json = json.load(
    open('datasets/CrowdHuman/annotations/val' + json_tail, 'r'))
img_id_count = 0
for img in crowdhuman_val_json['images']:
    img_id_count += 1
    img['file_name'] = 'crowdhuman/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)

for ann in crowdhuman_val_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

video_list.append({'id': max_video, 'file_name': 'crowdhuman_val'})

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json,
          open('datasets/mix_mot20_ch/annotations/train' + json_tail, 'w'))