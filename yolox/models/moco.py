#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import random
import os


class MoCo(object):
    """
    Build a MoCo model with: a key encoder and a queue, query encoder is not included (the code structure of YOLOX is too complicated)
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,
                 backbone,
                 head,
                 seq_names,
                 dim=128,
                 K=65536,
                 m=0.999,
                 T=0.07,
                 sim_thresh=0.3,
                 supervised=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.sim_thresh = sim_thresh
        self.supervised = supervised
        self.dim = dim
        self.num_classes = head.num_classes
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)

        # create the key encoder
        self.encoder_k_backbone = deepcopy(backbone)
        self.encoder_k_head = deepcopy(head)
        device = next(self.encoder_k_backbone.parameters()).device

        for param_q, param_k in zip(backbone.parameters(),
                                    self.encoder_k_backbone.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(head.parameters(),
                                    self.encoder_k_head.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue list
        self.seq_names = seq_names
        self.seq_num = len(seq_names)
        self.K_per_seq = K // (self.seq_num - 1)  # divided by sequence number
        self.K = 4096  # TODO
        self.K_per_obj = int(self.K // (self.seq_num - 1) * 1.2)
        self.__setattr__(
            'queue',
            self._gen_randn_feat((dim, self.K_per_seq, self.num_classes, self.seq_num), 0, device))

        self.__setattr__('queue_indices', torch.arange(self.seq_num))
        # cls_ptr, obj_ptr
        self.__setattr__(
            'queue_ptr',
            torch.zeros((self.seq_num, self.num_classes), dtype=torch.long))

    @staticmethod
    def _gen_randn_feat(size, norm_dim=0, device='0'):
        return F.normalize(torch.randn(*size).to(device), dim=norm_dim)
        # return F.normalize(torch.ones(*size).to(device), dim=norm_dim)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, backbone, head):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(backbone.parameters(),
                                    self.encoder_k_backbone.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(head.parameters(),
                                    self.encoder_k_head.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        self.device = next(backbone.parameters()).device

    def _gen_pos_logits(self, feat_q, feat_k, id_q=None, id_k=None):
        if id_q is None or (id_q == -1).all():
            assign = torch.mm(feat_q, feat_k.T)  # shape(Q, K)
            i = torch.arange(len(feat_q))
            values, indices = assign.max(dim=1)
            valid = values >= self.sim_thresh
            i, j = i[valid], indices[valid]
        else:
            id_q_, id_k_ = id_q.view(-1, 1), id_k.view(1, -1)
            # shape(Q, K)
            assign = (id_q_ == id_k_) & (id_q_ != -1) & (id_k_ != -1)
            i, j = torch.nonzero(assign.int(), as_tuple=True)

        return i, j, torch.einsum('nc,nc->n', feat_q[i], feat_k[j]).unsqueeze(-1)

    def _gen_neg_logits_intra(self, feat_q, feat_k):
        similarity = torch.einsum('nc,mc->nm', feat_q, feat_k)
        diag = torch.diag_embed(torch.diag(similarity) - 1e18)
        similarity = similarity + diag
        return similarity

    def _gen_neg_logits_inter(self, feat_q, seq_list_q, cls_q, intra_num=0):
        queue_indices = getattr(self, 'queue_indices')
        seq_i_list = [self.seq_names.index(seq_q) for seq_q in seq_list_q]
        valid_neg = torch.zeros_like(queue_indices) > -1
        valid_neg[seq_i_list] = False
        seq_indices = queue_indices[valid_neg]
        valid_num = valid_neg.sum()
        seq_indices = seq_indices.reshape(-1, 1).expand(valid_num,self.K_per_obj).reshape(-1)
        # TODO: more random choices with in the classes loop
        obj_indices = torch.tensor(random.sample(range(self.K_per_seq), self.K_per_obj))
        obj_indices = obj_indices.reshape(1, -1).expand(valid_num, self.K_per_obj).reshape(-1)
        cls_indices = torch.zeros_like(seq_indices)
        logits = []
        for ci in range(self.num_classes):
            query = feat_q[cls_q == ci, :]
            if len(query) == 0:
                continue
            queue = getattr(self, 'queue')[:, obj_indices, cls_indices + ci, seq_indices]

            delta_len = queue.shape[-1] + intra_num - self.K
            if delta_len > 0:
                queue = queue[..., random.sample(range(queue.shape[-1]), self.K - intra_num)]
            else:
                device = feat_q.device
                supplement = self._gen_randn_feat((self.dim, -delta_len), -2, device)
                queue = torch.cat((queue, supplement), dim=-1)

            logits.append(torch.einsum('nc,ck->nk', query, queue))

        return torch.cat(logits, 0)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, k_feats_dict):
        queue = getattr(self, 'queue')
        queue_ptr = getattr(self, 'queue_ptr')

        for seq_name, keys in k_feats_dict.items():
            seq_i = self.seq_names.index(seq_name)
            keys = torch.cat(keys, 0)
            cls = keys[:, -1]
            keys = keys[:, :-1]

            for cls_i in range(self.num_classes):
                cls_indices = cls == cls_i
                if cls_indices.sum() == 0:
                    continue
                _keys = keys[cls_indices]
                # get pointer
                ptr = int(queue_ptr[seq_i, cls_i])
                # in case of indices overflow
                curr_num = min(_keys.shape[0], self.K_per_seq - ptr)
                # replace the keys at ptr (dequeue and enqueue)
                queue[:, ptr:ptr + curr_num, cls_i,
                      seq_i] = keys[:curr_num, :].T
                # move pointer
                ptr = (ptr + curr_num) % self.K_per_seq

                queue_ptr[seq_i, cls_i] = ptr

        setattr(self, 'queue_ptr', queue_ptr)
        setattr(self, 'queue', queue)

    def _gen_logits_with_ids(self,
                             seq_list,
                             q_reid,
                             k_reid,
                             q_ids,
                             k_ids,
                             q_cls,
                             k_cls,
                             id_ranges=None):  # pseudo labels

        if len(q_reid) * len(k_reid) == 0:
            logit = torch.zeros([0, 1 + self.K], device=q_reid.device, dtype=q_reid.dtype)
        else:
            # positive logits: Nx1
            valid_q, valid_k, l_pos = self._gen_pos_logits(q_reid, k_reid, q_ids, k_ids)
            assert len(valid_q) == len(
                torch.unique(valid_q)), f'\n{q_ids}\n{k_ids}'
            assert len(valid_k) == len(torch.unique(valid_k))
            if len(valid_q) == 0:
                logit = torch.zeros([0, 1 + self.K],
                                    device=q_reid.device,
                                    dtype=q_reid.dtype)
            else:
                # negative logits: NxK
                # negative samples within current frame
                l_neg_intra = self._gen_neg_logits_intra(q_reid[valid_q], q_reid[valid_q])
                # negative samples within adjacent two frames
                l_neg_intra_2 = self._gen_neg_logits_intra(q_reid[valid_q], k_reid[valid_k])
                # inter-class variance is contained
                intra_num = len(l_neg_intra) + len(l_neg_intra_2)
                # negative samples in other videos
                l_neg_inter = self._gen_neg_logits_inter(q_reid[valid_q],
                                                         seq_list,
                                                         q_cls[valid_q],
                                                         intra_num=intra_num)
                # total logits: Nx(1+K)
                logit = torch.cat([l_pos, l_neg_intra, l_neg_intra_2, l_neg_inter], dim=1)

        if id_ranges is None:
            return logit
        else:
            k_feats = []
            for (id0, id1) in id_ranges:
                indices = (k_ids >= id0) & (k_ids <= id1)
                k_feats.append(torch.cat([k_cls[indices].reshape(-1, 1), k_reid[indices]], dim=1))
            return logit, k_feats

    @torch.no_grad()
    def extract_feat(self, x):
        return self.encoder_k_head(
            self.encoder_k_backbone(x))[-1]  # reid_feat only

    def parse_batch(self, feats, targets, ids):
        nb, ny, nx, dim = feats.shape
        # nlabel = (targets[..., :5].abs().sum(dim=2) > 1e-2).sum(dim=1)  # number of objects
        nlabel = (targets[..., 3].abs() > 1e-2).sum(dim=1)  # number of objects
        res = []
        s = min(self.encoder_k_head.strides)
        for bi in range(nb):
            num_gt = int(nlabel[bi])
            classes, xc, yc, _, _, _ = targets[bi, :num_gt, :].T
            _ids = ids[bi, :num_gt]
            assert len(_ids) == len(torch.unique(_ids)), f'{_ids}'
            yc, xc = (yc / s).clamp(0, ny - 1).long(), (xc / s).clamp(0, nx - 1).long()
            feat = feats[bi, yc, xc, :]
            res.append((feat, classes.long(), _ids))
        return res

    def compute_loss(self, inps):
        infos, reid_feat, reid_feat_aug, reid_feat_prev, targets, targets_aug, targets_prev = inps
        ids = infos[1]
        infos = infos[0]
        ids, ids_aug, ids_prev = ids.unbind(dim=1)
        data_pair = self.parse_batch(reid_feat, targets, ids)
        data_pair_aug = self.parse_batch(reid_feat_aug, targets_aug, ids_aug)
        data_pair_prev = self.parse_batch(reid_feat_prev, targets_prev, ids_prev)
        logits, logits_aug = [], []
        k_feats_dict = {}
        for bi in range(reid_feat.shape[0]):
            feat, cls, ids = data_pair[bi]
            feat_aug, cls_aug, ids_aug = data_pair_aug[bi]
            feat_prev, cls_prev, ids_prev = data_pair_prev[bi]
            seq_list, start_ids, end_ids = zip(*[item[-3:] for item in infos[bi]])

            # multiple images in a mosaic input
            logit, k_feats = self._gen_logits_with_ids(seq_list, feat,
                                                       feat_prev, ids,
                                                       ids_prev, cls, cls_prev,
                                                       zip(start_ids, end_ids))
            logits.append(logit)
            logits_aug.append(self._gen_logits_with_ids(seq_list, feat, feat_aug, ids, ids_aug, cls, cls_aug))
            for si, seq_name in enumerate(seq_list):
                if seq_name not in k_feats_dict:
                    k_feats_dict[seq_name] = [k_feats[si]]
                else:
                    k_feats_dict[seq_name].append(k_feats[si])

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_feats_dict)

        logits = torch.cat(logits, dim=0)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        logits_aug = torch.cat(logits_aug, dim=0)
        logits_aug /= self.T
        labels_aug = torch.zeros(logits_aug.shape[0], dtype=torch.long).to(logits_aug.device)

        return self.IDLoss(logits, labels), self.IDLoss(logits_aug, labels_aug)