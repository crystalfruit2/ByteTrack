from collections import deque
import numpy as np
import os
import os.path as osp
import copy
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, feature=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.features = deque(maxlen=3)
        self.scores = deque(maxlen=3)
        self.curr_feat = None
        self.smooth_feat = None
        
        if feature is not None:
            self.features.append(feature)
            self.scores.append(score)
            self.curr_feat = feature
            self.smooth_feat = feature.copy() # Represents Z^t
            
        # Hyperparameters (Set these based on your paper's specs)
        self.agg_option = 'B'  # 'A' for L1 Bias, 'B' for Softmax
        self.beta = 4.0        # Inertia of Memory (Option A)
        self.tau = 0.5         # Softmax Temperature (Option B)
    def _calculate_gammas(self):
        """
        Calculates dynamic weights based on the historical confidences.
        Assumes self.scores has exactly 3 elements: [c_{t-2}, c_{t-1}, c_t]
        """
        c_t2 = self.scores[0]
        c_t1 = self.scores[1]
        c_t = self.scores[2]
        
        if self.agg_option == 'A':
            # Option A: L1 Normalization with Historical Bias
            denom = c_t + self.beta * (c_t1 + c_t2) + 1e-6 # Add epsilon to prevent div by zero
            
            gamma_0 = c_t / denom
            gamma_1 = (self.beta * c_t1) / denom
            gamma_2 = (self.beta * c_t2) / denom
            
            return np.array([gamma_2, gamma_1, gamma_0])
            
        elif self.agg_option == 'B':
            # Option B: Temperature-Scaled Softmax
            scores_array = np.array([c_t2, c_t1, c_t])
            scaled_scores = scores_array / self.tau
            
            # Numerical stability: subtract max before exponentiating
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            gammas = exp_scores / np.sum(exp_scores)
            
            return gammas
            
        else:
            raise ValueError("agg_option must be 'A' or 'B'")

    def update_features(self, new_feature, new_score):
        """
        Replaces standard EMA. Updates Z^t using the calculated gammas.
        """
        self.features.append(new_feature)
        self.scores.append(new_score)
        
        # Need full history (t, t-1, t-2) to apply the order-2 math
        if len(self.features) < 3:
            # Fallback for the first two frames: simple averaging
            feats_array = np.array(self.features)
            self.smooth_feat = np.mean(feats_array, axis=0)
        else:
            # Apply Dynamic Aggregation
            gammas = self._calculate_gammas()
            feats_array = np.array(self.features)
            
            # gammas = [gamma_2, gamma_1, gamma_0]
            # feats_array = [Z^{t-2}, Z^{t-1}, f^t]
            # np.average with axis=0 applies the weights correctly to the feature vectors
            self.smooth_feat = np.average(feats_array, axis=0, weights=gammas)
            
        # Re-normalize the feature vector to keep it on the unit hypersphere
        norm = np.linalg.norm(self.smooth_feat)
        if norm > 0:
            self.smooth_feat /= norm


    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

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
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
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

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

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


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # --- LIGHTWEIGHT FEATURE EXTRACTOR ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = mobilenet_v2(pretrained=True).features.to(self.device).eval()
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((128, 64)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def update(self, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        raw_frame = img_info[0] if isinstance(img_info[0], np.ndarray) else None

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        if raw_frame is not None:
            img_h, img_w = raw_frame.shape[:2]
        else:
            img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        if raw_frame is not None and len(detections) > 0:
            for det in detections:
                tlwh = det.tlwh
                x, y, w, h = map(int, tlwh)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(raw_frame.shape[1], x + w), min(raw_frame.shape[0], y + h)
                crop = raw_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    if crop.ndim == 3 and crop.shape[2] == 3:
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_t = self.transform(crop).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self.extractor(crop_t).mean([2, 3]).squeeze().cpu().numpy()
                    det.curr_feat = feat
                else:
                    det.curr_feat = np.zeros(1280, dtype=np.float32)

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
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                # 1. Calculate Areas for the Occlusion Hard-Lock
                current_area = det.tlwh[2] * det.tlwh[3]
                kf_pred_area = track.mean[2] * track.mean[3]

                # 2. Check Mutation Threshold (e.g., 50% change)
                area_ratio = current_area / (kf_pred_area + 1e-6)
                is_area_mutated = area_ratio < 0.5 or area_ratio > 1.5

                # 3. Apply Dynamic Aggregation OR Trigger Hard-Lock
                if det.score >= 0.4 and not is_area_mutated and det.curr_feat is not None:
                    track.update_features(det.curr_feat, det.score)
                else:
                    pass
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        if raw_frame is not None and len(detections_second) > 0:
            for det in detections_second:
                tlwh = det.tlwh
                x, y, w, h = map(int, tlwh)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(raw_frame.shape[1], x + w), min(raw_frame.shape[0], y + h)
                crop = raw_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    if crop.ndim == 3 and crop.shape[2] == 3:
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_t = self.transform(crop).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self.extractor(crop_t).mean([2, 3]).squeeze().cpu().numpy()
                    det.curr_feat = feat
                else:
                    det.curr_feat = np.zeros(1280, dtype=np.float32)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                # 1. Calculate Areas for the Occlusion Hard-Lock
                current_area = det.tlwh[2] * det.tlwh[3]
                kf_pred_area = track.mean[2] * track.mean[3]

                # 2. Check Mutation Threshold (e.g., 50% change)
                area_ratio = current_area / (kf_pred_area + 1e-6)
                is_area_mutated = area_ratio < 0.5 or area_ratio > 1.5

                # 3. Apply Dynamic Aggregation OR Trigger Hard-Lock
                if det.score >= 0.4 and not is_area_mutated and det.curr_feat is not None:
                    track.update_features(det.curr_feat, det.score)
                else:
                    pass
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


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
