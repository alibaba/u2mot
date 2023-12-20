#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from loguru import logger


import argparse
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:            
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def clean_gt_and_ts(gt, ts):
    valid_cls = np.array([1,4,5,6,9]).reshape(1,-1)  # pedestrian,car,van,truck,bus
    columns = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused']

    def filter_cls(df):
        valid = (df['ClassId'].values.reshape(-1,1) == valid_cls).any(axis=1)
        valid &= df['Confidence'] > 0.
        return df[valid]

    def filter_ignore(df, ignore_regions):
        ignore = df['ClassId'] == -1
        frame_ids = np.array([v[0] for v in df.index.values])
        for frame_id, regions in ignore_regions.items():
            x1, y1, x2, y2 = np.split(np.array(regions).reshape(-1, 4).T, 4, axis=0)
            f_ind = frame_ids == frame_id
            xc, yc = (df['X'][f_ind] + df['Width'][f_ind] / 2.).values.reshape(-1,1), \
                     (df['Y'][f_ind] + df['Height'][f_ind] / 2.).values.reshape(-1,1)
            ignore[f_ind] = ((xc > x1) & (xc < x2) & (yc > y1) & (yc < y2)).any(axis=1)
        return df[~ignore]

    _gt, _ts = OrderedDict(), OrderedDict()
    for k, label in tqdm(gt.items(), desc='filtering bboxes'):
        ignore_indices = (label['ClassId'] == 0) | (label['ClassId'] == 11)
        ig_x1, ig_y1, ig_w, ig_h = label['X'][ignore_indices].values, \
                                   label['Y'][ignore_indices].values, \
                                   label['Width'][ignore_indices].values, \
                                   label['Height'][ignore_indices].values
        ig_x2, ig_y2 = ig_x1 + ig_w, ig_y1 + ig_h
        ignore_regions = {}
        frame_ids = np.array([v[0] for v in label.index.values])
        for i, frame_id in enumerate(frame_ids[ignore_indices]):
            if frame_id not in ignore_regions:
                ignore_regions[frame_id] = []
            ignore_regions[frame_id].append([ig_x1[i], ig_y1[i], ig_x2[i], ig_y2[i]])
        
        label = filter_cls(label)
        label = filter_ignore(label, ignore_regions)
        pred = filter_cls(ts[k])
        pred = filter_ignore(pred, ignore_regions)
        
        _gt[k] = label
        _ts[k] = pred

    return _gt, _ts


# evaluate MOTA
results_folder = 'YYOLOX_outputs/yolox_x_u2mot_visdrone/track_res'
mm.lap.default_solver = 'lap'

gt_type = '_test_dev'
gtfiles = sorted(glob.glob('datasets/VisDrone-MOT/VisDrone2019-MOT-test-dev/annotations/*txt'))
tsfiles = sorted(glob.glob(os.path.join(results_folder, '*.txt')))

logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
logger.info('Loading files.')

gt = OrderedDict([(os.path.basename(f)[:-4], mm.io.load_motchallenge(f, min_confidence=-1.0)) for f in gtfiles])
ts = OrderedDict([(os.path.basename(f)[:-4], mm.io.load_motchallenge(f, min_confidence=-1.0)) for f in tsfiles])    

gt, ts = clean_gt_and_ts(gt, ts)

mh = mm.metrics.create()    
accs, names = compare_dataframes(gt, ts)

logger.info('Running metrics')
metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
            'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
            'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
# summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
# print(mm.io.render_summary(
#   summary, formatters=mh.formatters, 
#   namemap=mm.io.motchallenge_metric_names))
div_dict = {
    'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
    'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
for divisor in div_dict:
    for divided in div_dict[divisor]:
        summary[divided] = (summary[divided] / summary[divisor])
fmt = mh.formatters
change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                    'partially_tracked', 'mostly_lost']
for k in change_fmt_list:
    fmt[k] = fmt['mota']
print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

metrics = mm.metrics.motchallenge_metrics + ['num_objects']
summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
logger.info('Completed')