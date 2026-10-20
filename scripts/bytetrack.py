"""
ByteTrack integration for multi-object tracking in ML pipelines.
Maintains object IDs across frames for continuous tracking.
"""

import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Optional
import cv2


class STrack:
    """Single target track with Kalman filtering."""
    
    shared_kalman = None
    track_id_count = 0
    
    def __init__(self, tlwh, score, cls):
        # tlwh: top-left width height
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.track_id = 0
        self.frame_id = 0
        self.start_frame = 0
        
    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
        self.tracklet_len = 0
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        
    def update(self, new_track, frame_id):
        """Update track with new detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.is_activated = True
        self.score = new_track.score
        
    @property
    def tlwh(self):
        """Get current position in tlwh format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
        
    @property
    def tlbr(self):
        """Get current position in tlbr (top-left bottom-right) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
        
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to xyah (center x, center y, aspect ratio, height)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
        
    @staticmethod
    def next_id():
        STrack.track_id_count += 1
        return STrack.track_id_count
        
    def predict(self):
        """Predict next position using Kalman filter."""
        if self.is_activated:
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)


class KalmanFilter:
    """Simple Kalman filter for bbox tracking."""
    
    def __init__(self):
        ndim, dt = 4, 1.0
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
            
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        
    def initiate(self, measurement):
        """Create track from initial measurement."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
        
    def predict(self, mean, covariance):
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
            
        return mean, covariance
        
    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        # Project state to measurement space
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        # Compute Kalman gain
        innovation_cov_total = projected_cov + innovation_cov
        kalman_gain = np.linalg.multi_dot((
            covariance, self._update_mat.T, np.linalg.inv(innovation_cov_total)))
        
        # Update state
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, innovation_cov_total, kalman_gain.T))
        
        return new_mean, new_covariance


class ByteTracker:
    """ByteTrack multi-object tracker."""
    
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        
        self.kalman_filter = KalmanFilter()
        
    def update(self, detections):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of [x1, y1, x2, y2, score, class_id]
            
        Returns:
            List of active tracks with [x1, y1, x2, y2, track_id, score, class_id]
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if len(detections) > 0:
            scores = detections[:, 4]
            classes = detections[:, 5].astype(int)
            bboxes = detections[:, :4]
            
            # Convert to tlwh format
            remain_inds = scores > self.track_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            
        else:
            dets = []
            scores_keep = []
            classes_keep = []
            
        if len(dets) > 0:
            # Convert detections to tlwh
            detections_tlwh = []
            for i, det in enumerate(dets):
                x1, y1, x2, y2 = det
                w, h = x2 - x1, y2 - y1
                detections_tlwh.append([x1, y1, w, h])
                
            detections = [STrack(tlwh, score, cls) 
                         for tlwh, score, cls in zip(detections_tlwh, scores_keep, classes_keep)]
        else:
            detections = []
            
        # Add newly detected tracklets
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                
        # Matching
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.shared_kalman = self.kalman_filter
        
        # Predict current location
        for strack in strack_pool:
            strack.predict()
            
        # Associate detections with tracks using IoU
        matches, u_track, u_detection = matching_by_iou(
            strack_pool, detections, self.match_thresh
        )
        
        # Update matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.is_activated:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        # Handle unmatched tracks
        for it in u_track:
            track = strack_pool[it]
            if not track.is_activated:
                continue
            track.mark_lost()
            lost_stracks.append(track)
            
        # Handle unmatched detections - create new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            
        # Update lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                
        # Merge lists
        self.tracked_stracks = [t for t in self.tracked_stracks if t.is_activated]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        # Get output tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        return output_stracks
        
    def reset(self):
        """Reset tracker state."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        STrack.track_id_count = 0


def matching_by_iou(tracks, detections, thresh=0.5):
    """Match tracks and detections using IoU."""
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))
        
    # Compute IoU matrix
    iou_matrix = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou_matrix[i, j] = bbox_iou(track.tlbr, det.tlbr)
            
    # Hungarian matching
    matches = []
    unmatched_tracks = []
    unmatched_detections = []
    
    matched_indices = linear_assignment(-iou_matrix)
    
    for i in range(len(tracks)):
        if i not in matched_indices[:, 0]:
            unmatched_tracks.append(i)
            
    for j in range(len(detections)):
        if j not in matched_indices[:, 1]:
            unmatched_detections.append(j)
            
    for i, j in matched_indices:
        if iou_matrix[i, j] < thresh:
            unmatched_tracks.append(i)
            unmatched_detections.append(j)
        else:
            matches.append((i, j))
            
    return matches, unmatched_tracks, unmatched_detections


def linear_assignment(cost_matrix):
    """Simple greedy matching."""
    try:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
    except ImportError:
        # Fallback greedy matching
        matches = []
        cost = cost_matrix.copy()
        for _ in range(min(cost.shape[0], cost.shape[1])):
            i, j = np.unravel_index(cost.argmin(), cost.shape)
            if cost[i, j] < 1e9:
                matches.append([i, j])
                cost[i, :] = 1e9
                cost[:, j] = 1e9
        return np.array(matches)


def bbox_iou(box1, box2):
    """Calculate IoU between two bboxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
        
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def joint_stracks(tlista, tlistb):
    """Join two track lists."""
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
    """Subtract tlistb from tlista."""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


# Mark methods for STrack
def mark_lost(self):
    self.is_activated = False

def mark_removed(self):
    self.is_activated = False

STrack.mark_lost = mark_lost
STrack.mark_removed = mark_removed
