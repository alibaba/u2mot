#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from loguru import logger

import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.dataset import Dataset

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.u2mot_tracker import U2MOTTracker
from yolox.tracking_utils.timer import Timer

import argparse
import os
from glob import glob
import os.path as osp
import shutil
import cv2
import numpy as np

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# python3 tools/track.py -f exps/example/mot/yolox_s_mot17_half.py -c YOLOX_outputs/yolox_s_mot17_half(0114_uncertainty_NoUpdate)/latest_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse
def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', 
                        help="benchmark to evaluate: MOT17 | MOT20 | VisDrone | BDD100K")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--save-result", action="store_true",default=True, help="whether to save the inference result of image/video")
    parser.add_argument('--no-save-result', dest='save_result', action='store_false')
    parser.add_argument("--plot-result", action="store_true",default=False, help="whether to plot the inference result of image/video")

    # model
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("-d", "--device", default='0', type=str, help="device for tracking")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file",)
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.",)
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold (high)")
    parser.add_argument("--low_thresh", type=float, default=0.1, help="tracking confidence threshold (low)")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--aspect-ratio-thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument("--fps", default=30, type=float, help="frame rate")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--mask-emb-with-iou", default=True, action="store_true", help="mask emb-cost matrix with iou in tracking")
    parser.add_argument('--no-mask-emb-with-iou', dest='mask_emb_with_iou', action='store_false')
    parser.add_argument("--fuse-emb-and-iou", default='min', type=str, help="fuse emb-cost and iou-cost in tracking")
    parser.add_argument("--use-uncertainty", default=False, action="store_true", help="use UTL for tracking")
    # cmc args
    parser.add_argument("--cmc-method", default="none", type=str, help="cmc method", 
                        choices=['none', 'file', 'files', 'orb', 'sift', 'ecc'])
    parser.add_argument("--cmc-file-dir", default='', type=str, help="cmc file directory")
    parser.add_argument("--cmc-seq-name", default='', type=str, help="cmc sequence name")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class ImageReader(Dataset):
    def __init__(self, img_files, test_size):
        self.img_files = img_files
        self.test_size = test_size
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        img_path = self.img_files[index]
        img_info = {"id": 0}
        img_info["file_name"] = osp.basename(img_path)
        img = cv2.imread(img_path)

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, Predictor.rgb_means, Predictor.std)
        img_info["ratio"] = ratio

        return img, img_info


def create_data_reader(img_files, test_size):
    dataset = ImageReader(img_files, test_size)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, 
                            pin_memory=True, num_workers=4)

    return dataloader


class Predictor(object):
    rgb_means = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt

    def inference(self, img, img_info, timer):
        # Parse Info
        img_info['id'] = img_info['id'].item()
        img_info['file_name'] = img_info['file_name'][0]
        img_info['height'] = img_info['height'].item()
        img_info['width'] = img_info['width'].item()
        img_info['raw_img'] = img_info['raw_img'].numpy()[0]

        # Run Detector
        img = img.float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs, reid_feat = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs, reid_feat, img_info


def image_track(predictor, vis_folder, res_folder, video_path, args):
    files = get_image_list(video_path)
    files.sort()
    num_frames = len(files)

    start = 0
    if args.benchmark == 'MOT17-val':
        start += len(files) // 2 + 1
        files = files[start:]

    video_name = osp.basename(video_path)
    if args.plot_result:
        save_path = osp.join(vis_folder, video_name+'.mp4')
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, cv2.imread(files[0]).shape[:2][::-1]
        )
    
    if args.cmc_method != 'none':
        assert args.cmc_file_dir != ''
        args.cmc_seq_name = video_name

    res_file = osp.join(res_folder, f"{video_name}.txt")
    tracker = U2MOTTracker(args, frame_rate=args.fps)

    timer = Timer()
    results = []
    s = min(predictor.model.head.strides)
    dataloader = create_data_reader(files, predictor.test_size)
    for frame_id, (img, img_info) in enumerate(dataloader, start+1):
        outputs, reid_feat, img_info = predictor.inference(img, img_info, timer)
        ny, nx = reid_feat.shape[1:3]
        scale = min(predictor.test_size[0] / float(img_info['height'], ), predictor.test_size[1] / float(img_info['width']))

        if outputs[0] is not None:
            outputs = outputs[0].cpu().numpy()
            outputs = outputs[np.argsort(outputs[:, 0])]  # TODO: debug

            reid_feat = reid_feat[0].cpu().numpy()
            x1, y1, x2, y2 = outputs[:, :4].T / s
            xc, yc = ((x1 + x2) / 2.).clip(0, nx-1).astype(np.int32), \
                     ((y1 + y2) / 2.).clip(0, ny-1).astype(np.int32)
            embeddings = reid_feat[yc, xc, :]

            # TODO: have been down inside
            # detections = outputs[:, :7]
            # detections[:, :4] /= scale

            # Run tracker
            online_targets = tracker.update(outputs, (img_info['height'], img_info['width']), predictor.test_size, 
                                            embeddings=embeddings, img=img_info['raw_img'], use_uncertainty=args.use_uncertainty)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > args.min_box_area and tlwh[2] / tlwh[3] < args.aspect_ratio_thresh and tlwh[3] / tlwh[2] < 4.:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{int(t.cls)+1},-1,-1\n"
                    )
            timer.toc()
            if args.plot_result:
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time, ids2=online_cls
                )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if args.plot_result:
            vid_writer.write(online_im)

        if frame_id % 100 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    if args.save_result:
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

        if args.benchmark == 'MOT17':
            for _seq in ['DPM', 'SDP']:
                dst_file = res_file.replace('FRCNN', _seq)
                shutil.copyfile(res_file, dst_file)
                logger.info(f"copy results to {dst_file}")


@logger.catch
def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        res_folder = osp.join(output_dir, "track_res")
        os.makedirs(vis_folder, exist_ok=True)
        os.makedirs(res_folder, exist_ok=True)

    if args.device != 'cpu' and isinstance(eval(args.device), int):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if args.trt:
        args.device = "cuda:0"
    else:
        args.device = torch.device("cuda" if args.device != "cpu" else "cpu")
        

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    
    for video_path in sorted(glob(osp.join(args.path, '*'))):
        if not parse_benchmark(predictor, video_path, exp, args):
            continue
        image_track(predictor, vis_folder, res_folder, video_path, args)
        # if args.plot_result:
        #     break


def parse_benchmark(predictor, video_path, exp, args):
    flag = True

    args.fps = 30
    args.track_thresh = 0.6
    args.low_thresh = 0.1
    args.match_thresh = 0.8
    args.track_buffer = 30
    predictor.test_size = exp.test_size

    seq = osp.basename(video_path)

    if args.benchmark == 'MOT17-val':
        if 'FRCNN' not in seq:
            flag = False
    elif args.benchmark == 'MOT17':
        if 'FRCNN' not in seq:
            flag = False
        else:
            if seq == 'MOT17-05-FRCNN' or seq == 'MOT17-06-FRCNN':
                args.track_buffer = 14
            elif seq == 'MOT17-13-FRCNN' or seq == 'MOT17-14-FRCNN':
                args.track_buffer = 25
            else:
                args.track_buffer = 30

            if seq == 'MOT17-01-FRCNN':
                args.track_thresh = 0.65
            elif seq == 'MOT17-06-FRCNN':
                args.track_thresh = 0.65
            elif seq == 'MOT17-12-FRCNN':
                args.track_thresh = 0.7
            elif seq == 'MOT17-14-FRCNN':
                args.track_thresh = 0.67

    elif args.benchmark == 'MOT20':
        args.mot20 = True
        args.match_thresh = 0.7

        if seq in ['MOT20-06', 'MOT20-08']:
            args.track_thresh = 0.5
            predictor.test_size = (736, 1920)
    
    elif args.benchmark == 'VisDrone':
        args.track_thresh = 0.5  # too many small objects
        # args.low_thresh = 0.05
        args.track_buffer = 15
        args.aspect_ratio_thresh = 1e5
        args.min_box_area = 1
    
    elif args.benchmark == 'BDD100K':
        args.track_thresh = 0.5
        args.track_buffer = 5
        args.aspect_ratio_thresh = 1e5
        args.min_box_area = 1

    else:
        raise ValueError('Invalid benchmark: ', args.benchmark)
    
    predictor.confthre = max(0.001, args.low_thresh - 0.01)

    return flag


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)