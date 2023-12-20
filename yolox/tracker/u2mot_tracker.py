#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from collections import deque

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
from .gmc import GMC
from . import matching

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, cls=0, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        if feat is not None:
            self.update_features(feat)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        # self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))


        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        # )
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        # self.mean, self.covariance = self.kalman_filter.update(
        #     self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        # ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class DefaultArgs(object):
    def __init__(self, mot20=False):
        self.track_thresh = 0.6
        self.low_thresh = 0.1
        self.track_buffer = 30
        self.match_thresh = 0.8

        self.mot20 = mot20
        self.mask_emb_with_iou = True
        self.fuse_emb_and_iou = 'min'

        self.cmc_method = 'none'
        self.cmc_seq_name = ''
        self.cmc_file_dir = ''


class U2MOTTracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        # self.args = args
        self.mot20 = args.mot20

        # self.det_thresh = args.track_thresh + 0.1

        self.track_high_thresh = args.track_thresh
        self.track_low_thresh = args.low_thresh
        self.new_track_thresh = args.track_thresh + 0.1
        self.match_thresh = args.match_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.iou_only = False
        self.mask_emb_with_iou = args.mask_emb_with_iou  # default True
        self.fuse_emb_and_iou = args.fuse_emb_and_iou  # default min, choice: [min, mean]
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25

        self.gmc = GMC(method=args.cmc_method, 
                       verbose=[args.cmc_seq_name, args.cmc_file_dir])

        self.super_cls = []
        # self.super_cls = np.array([0, 0, 1, 1, 1, 2, 3, 3])  # for bdd100k

    def _fuse_cost_matrix(self, iou_cost, emb_cost):
        if self.fuse_emb_and_iou == 'min':  # default min
            cost = np.minimum(iou_cost, emb_cost)
        elif self.fuse_emb_and_iou == 'mean':
            cost = (iou_cost + emb_cost) / 2.
        else:
            raise ValueError('Invalid fusion mode:', self.fuse_emb_and_iou)

        return cost

    @staticmethod
    def _risk(x, i, j):
        x = x.clip(0., 1.)
        pos = x[i, j].copy()
        neg = x[i, :].copy()
        neg[np.arange(len(i)), j] = 0.
        
        eps = 1e-6
        j2 = np.argsort(-neg, axis=1)[:, 0]  # 2nd maximal similarity
        i2 = np.arange(len(j2))
        neg2 = neg[i2, j2].copy()

        x = -np.log(pos + eps) - np.log(1. - neg2 + eps)

        return x

    @staticmethod
    def _threshold(x, m1=0.5, m2=0.05):
        eps = 1e-6
        border = -np.log(1. + m2 - x + eps) - np.log(m1)
        border[x > 0.98] = 1e4
        return border

    def ua_associate(self, strack_pool, detections, thresh=0.4, fuse_score=False):  # uncertainty-aware association
        iou_dists = matching.iou_distance(strack_pool, detections)

        if fuse_score:
            iou_dists = matching.fuse_score(iou_dists, detections)

        emb_dists = matching.embedding_distance(strack_pool, detections)
        if len(self.super_cls) > 1:
            emb_dists = matching.gate_cost_matrix_by_cls(emb_dists, strack_pool, detections, self.super_cls)
        similarity = 1. - emb_dists
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        if self.mask_emb_with_iou:
            emb_dists[iou_dists > self.proximity_thresh] = 1.0
        dists = self._fuse_cost_matrix(iou_dists, emb_dists)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=thresh)
        if len(strack_pool) * len(detections) == 0 or len(matches) == 0:
            return matches, u_track, u_detection, np.zeros(len(matches), dtype=np.float32)

        i, j = matches.T
        risk = self._risk(similarity.T, j, i)  # compute uncertainty between object against track
        uncertainty = np.maximum(risk - self._threshold(similarity[i, j]), 0.)
        invalid = uncertainty > 0

        if invalid.sum() and len(strack_pool):
            n, m = dists.shape
            i0, j0 = np.arange(n), np.arange(m)
            unpaired_i, unpaired_j = i0 > -1, j0 > -1
            unpaired_i[i[~invalid]] = False
            unpaired_j[j[~invalid]] = False
            i1, j1 = i0[unpaired_i], j0[unpaired_j]

            det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
            trk_features_trklet = [np.stack(list(track.features)[-5:]) for track in strack_pool]

            similarity2 = np.zeros((len(i1), len(j1)))
            for k, ii in enumerate(i1):
                for l, jj in enumerate(j1):
                    similarity2[k, l] = np.mean(trk_features_trklet[ii] @ det_features[jj:jj+1].T).clip(0,1)
            
            emb_dists2 = 1. - similarity2
            iou_dists2 = iou_dists[i1, :][:, j1]
            emb_dists2[emb_dists2 > self.appearance_thresh] = 1.0
            if self.mask_emb_with_iou:
                emb_dists2[iou_dists2 > self.proximity_thresh] = 1.0
            dists2 = self._fuse_cost_matrix(iou_dists2, emb_dists2)
            matches2, u_track2, u_detection2 = matching.linear_assignment(dists2, thresh=0.9)

            if len(matches2) > 0:
                i2, j2 = matches2.T
                uncertainty2 = np.maximum(self._risk(similarity2.T, j2, i2) - self._threshold(similarity2[i2, j2]), 0.)
                matches = np.stack((np.concatenate((i[~invalid], i1[i2])), np.concatenate((j[~invalid], j1[j2])))).T
                uncertainty = np.concatenate((uncertainty[~invalid], uncertainty2))
                unpaired_i[i1[i2]] = False
                unpaired_j[j1[j2]] = False
            else:
                matches = np.stack((i[~invalid], j[~invalid])).T
            u_track, u_detection = i0[unpaired_i], j0[unpaired_j]
            
        return matches, u_track, u_detection, uncertainty

    def associate(self, trks, dets, thresh, fuse_score=False, iou_only=False):
        iou_dists = matching.iou_distance(trks, dets)

        if fuse_score:
            iou_dists = matching.fuse_score(iou_dists, dets)

        dists = iou_dists
        if not iou_only:
            # Popular ReID method (JDE / FairMOT)
            emb_dists = matching.embedding_distance(trks, dets)

            if len(self.super_cls) > 1:
                emb_dists = matching.gate_cost_matrix_by_cls(emb_dists, trks, dets, self.super_cls)

            # dists = matching.fuse_motion(self.kalman_filter, emb_dists, trks, dets)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            if self.mask_emb_with_iou:
                emb_dists[iou_dists > self.proximity_thresh] = 1.0  # TODO: iou mask or not
            dists = self._fuse_cost_matrix(dists, emb_dists)
        
        return matching.linear_assignment(dists, thresh=thresh)

    def update(self, output_results, img_info, img_size, embeddings=None, use_uncertainty=False, img=None):
        """
        update tracks, e.g. activated, refind, lost and removed tracks
        Args:
            output_results: tensor of shape [bbox_num, 7], 7 for bbox(4) + obj_conf + cls_conf + cls
            img_info:list, [origin_H, origin_W, 1, 1 img_path]
            img_size: tuple, (input_H, input_W)

        Returns:

        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            # output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] #* output_results[:, 5]        # score = obj_conf * cls_conf
            bboxes = output_results[:, :4]                              # x1y1x2y2, e.g. tlbr
            classes = output_results[:, 6]
            
            img_h, img_w = img_info[0], img_info[1]         # origin image size
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))     # input/origin, <1
            bboxes /= scale     # map bbox to origin image size

            remain_inds = scores > self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_high = scores < self.track_high_thresh

            inds_second = np.logical_and(inds_low, inds_high)       # self.track_high_thresh > score > self.track_low_thresh, for second matching
            dets_second = bboxes[inds_second]                       # detections for second matching
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
            feats_second = embeddings[inds_second]

            dets = bboxes[remain_inds]                              # detections for first matching, high quality
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            feats_keep = embeddings[remain_inds]

        else:
            dets_second = []
            scores_second = []
            classes_second = []
            feats_second = []
            dets = []
            scores_keep = []
            classes_keep = []
            feats_keep = []

        # detections for first matching (high score)
        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                            (tlbr, s, c, f) in zip(dets, scores_keep, classes_keep, feats_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)       # tracked but bot confirmed, e.g. newly init tracks
            else:
                tracked_stracks.append(track)   # normally tracked objects, e.g. activated

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)     # combine tracked_stracks and lost_stracks to strack_pool
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)       # Kalman Filter predict, only for normally tracked and lost, not for uncomfirmed

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        if not use_uncertainty:
            matches, u_track, u_detection = self.associate(strack_pool, detections, thresh=self.match_thresh, 
                                                           fuse_score=not self.mot20, iou_only=self.iou_only)
        else:
            matches, u_track, u_detection, _ = self.ua_associate(strack_pool, detections, 
                                                                thresh=self.match_thresh, fuse_score=not self.mot20)

        for itracked, idet in matches:          # do update w.r.t matching results
            track = strack_pool[itracked]       # track w.r.t index
            det = detections[idet]              # detection w.r.t index
            if track.state == TrackState.Tracked:       # normally tracked successfully tracked
                track.update(detections[idet], self.frame_id)   # update of class STrack, update Kalman Filter, track state and other settings
                activated_starcks.append(track)
            else:                                       # lost tracklets re-found
                track.re_activate(det, self.frame_id, new_id=False)     # re-activate and update Kalman Filter
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                                 (tlbr, s, c, f) in zip(dets_second, scores_second, classes_second, feats_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        matches, u_track, u_detection_second = self.associate(r_tracked_stracks, detections_second, thresh=0.5, 
                                                              fuse_score=False, iou_only=True)

        for itracked, idet in matches:      # do update w.r.t second matching results
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:       # normally tracklets successfully tracked
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:                                       # lost tracklets re-found
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:              # set unmatched tracks as 'lost'
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        matches, u_unconfirmed, u_detection = self.associate(unconfirmed, detections, thresh=0.7, 
                                                       fuse_score=not self.mot20, iou_only=self.iou_only) # True
        for itracked, idet in matches:      # unconfirmed tracklets matched
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:            # remove unconfirmed tracklets for 2 frames
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:     # remove lost tracklets for 'max_time_lost' frames
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]

        return output_stracks

    def update2(self, gt_targets, img_info, img_size, embeddings):  # generate pseudo-identities only
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        identities = np.zeros((len(gt_targets))) - 100.
        uncertainty = np.zeros_like(identities)

        if len(gt_targets):
            img_h, img_w = img_info[0], img_info[1]         # origin image size
            scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))     # input/origin, <1
            x1, y1, w, h = (gt_targets[:, 1:5] / scale).T  # map bbox to origin image size
            x2, y2 = x1 + w, y1 + h
            dets = np.stack((x1, y1, x2, y2)).T
            classes = gt_targets[:, 0]
        else:
            dets = []
            classes = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), 1., c, f) for
                            (tlbr, c, f) in zip(dets, classes, embeddings)]
        else:
            detections = []
        det_indices = np.arange(len(detections))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Associate with high score detection boxes
        # matches, u_track, u_detection = self.associate(strack_pool, detections, thresh=self.match_thresh, 
        #                                                fuse_score=not self.mot20, iou_only=False)
        matches, u_track, u_detection, match_uncertainty = \
                self.ua_associate(strack_pool, detections, thresh=self.match_thresh, fuse_score=not self.mot20)

        for ind, (itracked, idet) in enumerate(matches):
            track = strack_pool[itracked]
            det = detections[idet]
            identities[idet] = track.track_id
            uncertainty[idet] = match_uncertainty[ind]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        det_indices = [det_indices[i] for i in u_detection]
        matches, u_unconfirmed, u_detection = self.associate(unconfirmed, detections, thresh=0.7, 
                                                       fuse_score=not self.mot20, iou_only=False)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            identities[det_indices[idet]] = unconfirmed[itracked].track_id
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            identities[det_indices[inew]] = track.track_id

        """ Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        pf = f'{self.frame_id}\n{identities}'
        assert len(identities) == len(np.unique(identities)), pf
        assert (identities < 0).sum() == 0, pf
        assert len(identities) == len(uncertainty) == len(gt_targets)

        return identities, uncertainty

    def update2(self, labels, embeddings, use_uncertainty=True):  # generate pseudo-identities only
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        identities = np.zeros((len(labels))) - 100.
        uncertainty = np.zeros_like(identities)

        if len(labels):
            dets = labels[:, :4]
            classes = labels[:, 4]
        else:
            dets = []
            classes = []

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), 1., c, f) for
                            (tlbr, c, f) in zip(dets, classes, embeddings)]
        else:
            detections = []
        det_indices = np.arange(len(detections))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Associate with high score detection boxes
        if use_uncertainty:
            matches, u_track, u_detection, match_uncertainty = \
                self.ua_associate(strack_pool, detections, thresh=self.match_thresh, fuse_score=not self.mot20)
        else:
            matches, u_track, u_detection = self.associate(strack_pool, detections, thresh=self.match_thresh, 
                                                        fuse_score=not self.mot20, iou_only=False)

        for ind, (itracked, idet) in enumerate(matches):
            track = strack_pool[itracked]
            det = detections[idet]
            identities[idet] = track.track_id
            if use_uncertainty:
                uncertainty[idet] = match_uncertainty[ind]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        det_indices = [det_indices[i] for i in u_detection]
        matches, u_unconfirmed, u_detection = self.associate(unconfirmed, detections, thresh=0.7, 
                                                       fuse_score=not self.mot20, iou_only=False)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            identities[det_indices[idet]] = unconfirmed[itracked].track_id
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            identities[det_indices[inew]] = track.track_id

        """ Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        pf = f'{self.frame_id}\n{identities}'
        assert len(identities) == len(np.unique(identities)), pf
        assert (identities < 0).sum() == 0, pf
        assert len(identities) == len(uncertainty) == len(embeddings)

        return identities, uncertainty

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb