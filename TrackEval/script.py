from glob import glob
from tqdm import tqdm
import os
import os.path as osp
import numpy as np


def read_mot17_val_frames():
    ret = {}
    with open('../UnsupMOT_fm/src/data/mot17.val', 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        seq = osp.basename(osp.dirname(osp.dirname(line)))
        if seq not in ret:
            ret[seq] = []
        frame = int(osp.basename(line)[:-4])
        ret[seq].append(frame)
    
    keys = list(ret.keys())
    for k in keys:
        ret[k.replace('SDP', 'FRCNN')] = ret[k]
        ret[k.replace('SDP', 'DPM')] = ret[k]
    
    return ret


def create_mot17_half_gt():
    root = './data/gt/mot_challenge/MOT17-train'
    val_dict = read_mot17_val_frames()
    for seq in tqdm(os.listdir(root)):
        with open(osp.join(root, seq, 'gt', 'gt.txt'), 'r') as f:
            lines = f.readlines()
        val_lines = []
        for line in lines:
            frame = int(line.split(',')[0])
            if frame in val_dict[seq]:
                val_lines.append(line)
        with open(osp.join(root, seq, 'gt', 'gt.val.txt'), 'w+') as f:
            f.writelines(val_lines)

def cvt_half_gt():
    root = './data/gt/mot_challenge/MOT17-train'
    for seq in tqdm(os.listdir(root)):
        label = np.loadtxt(osp.join(root, seq, 'gt', 'gt.val.txt'), delimiter=',')
        label[:, 0] -= label[:, 0].min() - 1
        np.savetxt(osp.join(root, seq, 'gt', 'gt.val_new.txt'), label, fmt='%d,%d,%d,%d,%d,%d,%d,%d,%s')


if __name__ == '__main__':
    # create_mot17_half_gt()
    cvt_half_gt()
