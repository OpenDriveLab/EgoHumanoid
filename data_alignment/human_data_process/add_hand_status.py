#!/usr/bin/env python3
"""
process HDF5 file to add hand_status key using square wave approximation.

Three-directory workflow:
- raw_source: Read hand_pose data from here
- mid_source: Copy HDF5 from here (base file)
- target: Save result with hand_status added

processing pipeline:
- Compute tip_distance and curvature metrics (weighted average)
- Normalize to 0-1, invert
- Apply enhancement (push values toward 0 and 1)
- Square wave approximation (optimize x, y parameters)
- Downsample by 5 (frame 0-4→0, 5-9→5, etc.)

Usage:
    python add_hand_status.py --raw /path/to/raw --mid /path/to/mid --target /path/to/target
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from tqdm import tqdm
import shutil
from concurrent.futures import processPoolExecutor, as_completed
import multiprocessing


# Finger definitions
FINGERS = [
    ("Thumb", 2, 5),
    ("Index", 6, 10),
    ("Middle", 11, 15),
    ("Ring", 16, 20),
    ("Little", 21, 25),
]

# Finger joint indices for curvature calculation
FINGER_JOINTS = {
    "Thumb": [2, 3, 4, 5],
    "Index": [6, 7, 8, 9, 10],
    "Middle": [11, 12, 13, 14, 15],
    "Ring": [16, 17, 18, 19, 20],
    "Little": [21, 22, 23, 24, 25],
}

# Finger tip indices
FINGER_TIPS = {
    "Thumb": 5,
    "Index": 10,
    "Middle": 15,
    "Ring": 20,
    "Little": 25,
}


def compute_tip_avg_distance(hand_pose_data):
    """
    Compute average distance between all finger tips.
    """
    n_frame = hand_pose_data.shape[0]
    finger_names = list(FINGER_TIPS.keys())
    n_fingers = len(finger_names)
    
    # Get all tip positions
    tip_positions = np.zeros((n_fingers, n_frame, 3))
    for i, name in enumerate(finger_names):
        tip_idx = FINGER_TIPS[name]
        tip_positions[i] = hand_pose_data[:, tip_idx, :3]
    
    # For each finger, compute average distance to other fingers
    finger_avg_distances = np.zeros((n_fingers, n_frame))
    for i in range(n_fingers):
        distances_to_others = []
        for j in range(n_fingers):
            if i != j:
                dist = np.linalg.norm(tip_positions[i] - tip_positions[j], axis=1)
                distances_to_others.append(dist)
        finger_avg_distances[i] = np.mean(distances_to_others, axis=0)
    
    avg_distance = np.mean(finger_avg_distances, axis=0)
    return avg_distance


def compute_finger_curvature(hand_pose_data):
    """
    Compute average curvature of all fingers.
    Curvature = path_length / direct_distance
    """
    n_frame = hand_pose_data.shape[0]
    finger_names = list(FINGER_JOINTS.keys())
    n_fingers = len(finger_names)
    
    finger_curvatures = np.zeros((n_fingers, n_frame))
    
    for i, name in enumerate(finger_names):
        joint_indices = FINGER_JOINTS[name]
        n_joints = len(joint_indices)
        
        joint_positions = np.zeros((n_joints, n_frame, 3))
        for j, idx in enumerate(joint_indices):
            joint_positions[j] = hand_pose_data[:, idx, :3]
        
        # Total path length
        path_length = np.zeros(n_frame)
        for j in range(n_joints - 1):
            segment_length = np.linalg.norm(joint_positions[j+1] - joint_positions[j], axis=1)
            path_length += segment_length
        
        # Direct distance
        direct_distance = np.linalg.norm(joint_positions[-1] - joint_positions[0], axis=1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            curvature = np.where(direct_distance > 1e-6, 
                                 path_length / direct_distance, 
                                 1.0)
        
        finger_curvatures[i] = curvature
    
    avg_curvature = np.mean(finger_curvatures, axis=0)
    return avg_curvature


def compute_weighted_trajectory(hand_pose_data, weight_tip_distance=0.5, weight_curvature=0.5):
    """
    Compute weighted combination of tip_distance and curvature metrics.
    Both are normalized to 0-1 and inverted so that 1 = closed, 0 = open.
    """
    trajectory = np.zeros(hand_pose_data.shape[0])
    total_weight = weight_tip_distance + weight_curvature
    
    if weight_tip_distance > 0:
        raw = compute_tip_avg_distance(hand_pose_data)
        min_val, max_val = raw.min(), raw.max()
        if max_val > min_val:
            normalized = (raw - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(raw)
        # Invert: large distance = open = 0, small distance = closed = 1
        trajectory += weight_tip_distance * (1.0 - normalized)
    
    if weight_curvature > 0:
        raw = compute_finger_curvature(hand_pose_data)
        min_val, max_val = raw.min(), raw.max()
        if max_val > min_val:
            normalized = (raw - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(raw)
        # Curvature: high = bent = closed = 1, no inversion needed
        trajectory += weight_curvature * normalized
    
    if total_weight > 0:
        trajectory /= total_weight
    
    return trajectory


def enhance_trajectory(trajectory, thresh_high=0.8, thresh_low=0.2, 
                       enhance_high=1.0, enhance_low=1.0, use_mean_thresh=False,
                       mean_thresh_offset=0.25):
    """
    Enhance trajectory to push values closer to 0 and 1.
    
    Args:
        mean_thresh_offset: offset to subtract from mean when computing mean threshold
    """
    result = trajectory.copy()
    
    if use_mean_thresh:
        mean_height = float(np.clip(np.mean(trajectory) + mean_thresh_offset, 0.0, 1.0))
        
        mask_high = trajectory >= mean_height
        if np.any(mask_high) and (1 - mean_height) > 0:
            normalized = (trajectory[mask_high] - mean_height) / (1 - mean_height)
            enhanced = 1 - (1 - normalized) ** enhance_high
            result[mask_high] = mean_height + enhanced * (1 - mean_height)
        
        mask_low = trajectory < mean_height
        if np.any(mask_low) and mean_height > 0:
            normalized = trajectory[mask_low] / mean_height
            enhanced = normalized ** enhance_low
            result[mask_low] = enhanced * mean_height
    else:
        mask_high = trajectory >= thresh_high
        if np.any(mask_high) and (1 - thresh_high) > 0:
            normalized = (trajectory[mask_high] - thresh_high) / (1 - thresh_high)
            enhanced = 1 - (1 - normalized) ** enhance_high
            result[mask_high] = thresh_high + enhanced * (1 - thresh_high)
        
        mask_low = trajectory <= thresh_low
        if np.any(mask_low) and thresh_low > 0:
            normalized = trajectory[mask_low] / thresh_low
            enhanced = normalized ** enhance_low
            result[mask_low] = enhanced * thresh_low
    
    return result


def generate_square_wave(n_frame, x, y=None, wave_type='0-1-0', transitions=None):
    """
    Generate a square wave of specified type.
    
    Wave types:
    - '0-1-0': starts at 0, becomes 1 at frame x, returns to 0 at frame y (2 transitions)
    - '1-0': starts at 1, becomes 0 at frame x (1 transition)
    - '0-1': starts at 0, becomes 1 at frame x (1 transition)
    - '0-1-0-1-0': starts at 0, alternates 0-1-0-1-0 (4 transitions, requires transitions list)
    - '0-1-0-1-0-1-0': starts at 0, alternates 0-1-0-1-0-1-0 (6 transitions, requires transitions list)
    
    Args:
        transitions: list of transition points (frame indices) for multi-transition waves
    """
    x_i = int(np.floor(x))
    x_i = max(0, min(n_frame - 1, x_i))
    
    if wave_type == '1-0':
        # Start at 1, become 0 at x
        square_wave = np.ones(n_frame, dtype=float)
        square_wave[x_i:] = 0.0
        return square_wave
    
    elif wave_type == '0-1':
        # Start at 0, become 1 at x
        square_wave = np.zeros(n_frame, dtype=float)
        square_wave[x_i:] = 1.0
        return square_wave
    
    elif wave_type == '0-1-0-1-0':
        # Multi-transition: 0-1-0-1-0 (4 transitions)
        square_wave = np.zeros(n_frame, dtype=float)
        if transitions is None or len(transitions) < 4:
            return square_wave
        t0, t1, t2, t3 = [max(0, min(n_frame, int(np.floor(t)))) for t in transitions[:4]]
        if t0 < t1 < t2 < t3:
            square_wave[t0:t1] = 1.0
            square_wave[t2:t3] = 1.0
        return square_wave
    
    elif wave_type == '0-1-0-1-0-1-0':
        # Multi-transition: 0-1-0-1-0-1-0 (6 transitions)
        square_wave = np.zeros(n_frame, dtype=float)
        if transitions is None or len(transitions) < 6:
            return square_wave
        t0, t1, t2, t3, t4, t5 = [max(0, min(n_frame, int(np.floor(t)))) for t in transitions[:6]]
        if t0 < t1 < t2 < t3 < t4 < t5:
            square_wave[t0:t1] = 1.0
            square_wave[t2:t3] = 1.0
            square_wave[t4:t5] = 1.0
        return square_wave
    
    else:  # '0-1-0' (default)
        square_wave = np.zeros(n_frame, dtype=float)
        if y is None:
            return square_wave
        
        y_i = int(np.floor(y))
        y_i = max(0, min(n_frame, y_i))
        
        if y_i <= x_i:
            return square_wave
        
        square_wave[x_i:y_i] = 1.0
        return square_wave


def compute_weights(trajectory, thresh_high=0.8, thresh_low=0.2, weight_high=1.0, weight_low=1.0):
    """Compute position-dependent weights for trajectory."""
    weights = np.ones_like(trajectory)
    weights[trajectory >= thresh_high] = weight_high
    weights[trajectory <= thresh_low] = weight_low
    return weights


def optimize_square_wave(trajectory, thresh_high=0.8, thresh_low=0.2, 
                         weight_high=1.0, weight_low=1.0, wave_type='0-1-0',
                         margin=0):
    """
    Find optimal square wave parameters that minimize weighted MSE.
    
    Args:
        wave_type: type of wave ('0-1-0', '1-0', '0-1', '0-1-0-1-0', '0-1-0-1-0-1-0')
        margin: minimum distance from boundaries for transition points (frame).
                if margin > 0, transition points are forced to be within [margin, n_frame - margin],
                guaranteeing the specified number of transitions.
                - For '1-0' and '0-1': ensures 1 transition
                - For '0-1-0': ensures 2 transitions
                - For '0-1-0-1-0': ensures 4 transitions
                - For '0-1-0-1-0-1-0': ensures 6 transitions
    
    Returns:
        best_x: optimal first transition point (or transitions list for multi-transition waves)
        best_y: optimal second transition point (None for single-transition waves, transitions list for multi-transition waves)
        best_mse: minimum MSE achieved
        best_wave: the optimal square wave
    """
    n_frame = len(trajectory)
    weights = compute_weights(trajectory, thresh_high, thresh_low, weight_high, weight_low)
    
    if wave_type in ('1-0', '0-1'):
        # Single transition point optimization
        # With margin constraint: x must be in [margin, n_frame - margin - 1]
        x_min = margin
        x_max = n_frame - 1 - margin
        
        # Ensure valid range
        if x_max < x_min:
            x_min = n_frame // 2
            x_max = n_frame // 2
        
        def objective(params):
            x = params[0]
            x_i = int(np.floor(x))
            x_i = max(x_min, min(x_max, x_i))
            wave = generate_square_wave(n_frame, x_i, wave_type=wave_type)
            weighted_mse = np.mean(weights * (wave - trajectory) ** 2)
            return float(weighted_mse)
        
        best_mse = np.inf
        best_x = (x_min + x_max) // 2
        
        # Grid search for good initialization
        n_init = 20
        for i in range(n_init):
            x_init = x_min + int(i * (x_max - x_min) / max(n_init - 1, 1))
            result = minimize(objective, [x_init], method='Powell')
            if result.success and result.fun < best_mse:
                x_val = int(np.floor(result.x[0]))
                x_val = max(x_min, min(x_max, x_val))
                best_mse = float(result.fun)
                best_x = x_val
        
        best_wave = generate_square_wave(n_frame, best_x, wave_type=wave_type)
        return int(best_x), None, best_mse, best_wave
    
    elif wave_type == '0-1-0-1-0':
        # Multi-transition: 0-1-0-1-0 (4 transitions: t0, t1, t2, t3)
        def objective(params):
            t0, t1, t2, t3 = params
            transitions = [max(margin, min(n_frame - margin, int(np.floor(t)))) for t in [t0, t1, t2, t3]]
            # Ensure ordering
            transitions = sorted(transitions)
            if transitions[0] < transitions[1] < transitions[2] < transitions[3]:
                wave = generate_square_wave(n_frame, 0, wave_type=wave_type, transitions=transitions)
                weighted_mse = np.mean(weights * (wave - trajectory) ** 2)
                return float(weighted_mse)
            return np.inf
        
        best_mse = np.inf
        best_transitions = [n_frame // 5, n_frame // 4, n_frame // 2, 3 * n_frame // 4]
        
        # Grid search for good initialization
        n_init = 10
        for i0 in range(n_init):
            for i1 in range(i0 + 1, n_init):
                for i2 in range(i1 + 1, n_init):
                    for i3 in range(i2 + 1, n_init):
                        t0_init = margin + int(i0 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                        t1_init = margin + int(i1 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                        t2_init = margin + int(i2 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                        t3_init = margin + int(i3 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                        result = minimize(objective, [t0_init, t1_init, t2_init, t3_init], method='Powell')
                        if result.success and result.fun < best_mse:
                            transitions = sorted([max(margin, min(n_frame - margin, int(np.floor(t)))) for t in result.x])
                            if len(transitions) == 4 and transitions[0] < transitions[1] < transitions[2] < transitions[3]:
                                best_mse = float(result.fun)
                                best_transitions = transitions
        
        best_wave = generate_square_wave(n_frame, 0, wave_type=wave_type, transitions=best_transitions)
        return best_transitions, best_transitions, best_mse, best_wave
    
    elif wave_type == '0-1-0-1-0-1-0':
        # Multi-transition: 0-1-0-1-0-1-0 (6 transitions: t0, t1, t2, t3, t4, t5)
        def objective(params):
            t0, t1, t2, t3, t4, t5 = params
            transitions = [max(margin, min(n_frame - margin, int(np.floor(t)))) for t in [t0, t1, t2, t3, t4, t5]]
            # Ensure ordering
            transitions = sorted(transitions)
            if transitions[0] < transitions[1] < transitions[2] < transitions[3] < transitions[4] < transitions[5]:
                wave = generate_square_wave(n_frame, 0, wave_type=wave_type, transitions=transitions)
                weighted_mse = np.mean(weights * (wave - trajectory) ** 2)
                return float(weighted_mse)
            return np.inf
        
        best_mse = np.inf
        best_transitions = [n_frame // 7, n_frame // 6, 2 * n_frame // 7, 3 * n_frame // 7, 4 * n_frame // 7, 5 * n_frame // 7]
        
        # Grid search for good initialization (reduced grid for 6 transitions)
        n_init = 8
        for i0 in range(n_init):
            for i1 in range(i0 + 1, n_init):
                for i2 in range(i1 + 1, n_init):
                    for i3 in range(i2 + 1, n_init):
                        for i4 in range(i3 + 1, n_init):
                            for i5 in range(i4 + 1, n_init):
                                t0_init = margin + int(i0 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                                t1_init = margin + int(i1 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                                t2_init = margin + int(i2 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                                t3_init = margin + int(i3 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                                t4_init = margin + int(i4 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                                t5_init = margin + int(i5 * (n_frame - 2 * margin) / max(n_init - 1, 1))
                                result = minimize(objective, [t0_init, t1_init, t2_init, t3_init, t4_init, t5_init], method='Powell')
                                if result.success and result.fun < best_mse:
                                    transitions = sorted([max(margin, min(n_frame - margin, int(np.floor(t)))) for t in result.x])
                                    if len(transitions) == 6 and transitions[0] < transitions[1] < transitions[2] < transitions[3] < transitions[4] < transitions[5]:
                                        best_mse = float(result.fun)
                                        best_transitions = transitions
        
        best_wave = generate_square_wave(n_frame, 0, wave_type=wave_type, transitions=best_transitions)
        return best_transitions, best_transitions, best_mse, best_wave
    
    else:  # '0-1-0' (default)
        # With margin constraint: 
        # x must be in [margin, n_frame - 2*margin - 1]
        # y must be in [x + margin, n_frame - margin]
        x_min = margin
        x_max = n_frame - 2 * margin - 1
        
        # Ensure valid range
        if x_max < x_min:
            x_min = n_frame // 3
            x_max = n_frame // 3
        
        def objective(params):
            x, y = params
            x_i = int(np.floor(x))
            y_i = int(np.floor(y))
            x_i = max(x_min, min(x_max, x_i))
            y_min = x_i + margin if margin > 0 else x_i + 1
            y_max = n_frame - margin if margin > 0 else n_frame
            y_i = max(y_min, min(y_max, y_i))
            wave = generate_square_wave(n_frame, x_i, y_i, wave_type='0-1-0')
            weighted_mse = np.mean(weights * (wave - trajectory) ** 2)
            return float(weighted_mse)
        
        best_mse = np.inf
        best_x = x_min
        best_y = n_frame // 2
        
        # Grid search for good initialization
        n_init = 20
        for i in range(n_init):
            for j in range(i + 1, n_init):
                x_init = x_min + int(i * (x_max - x_min) / max(n_init - 1, 1))
                y_min_init = x_init + (margin if margin > 0 else 1)
                y_max_init = n_frame - margin if margin > 0 else n_frame
                y_init = y_min_init + int(j * (y_max_init - y_min_init) / max(n_init - 1, 1))
                if y_init > x_init:
                    result = minimize(objective, [x_init, y_init], method='Powell')
                    if result.success and result.fun < best_mse:
                        x_val = int(np.floor(result.x[0]))
                        y_val = int(np.floor(result.x[1]))
                        x_val = max(x_min, min(x_max, x_val))
                        y_min = x_val + (margin if margin > 0 else 1)
                        y_max = n_frame - margin if margin > 0 else n_frame
                        y_val = max(y_min, min(y_max, y_val))
                        best_mse = float(result.fun)
                        best_x = x_val
                        best_y = y_val
        
        if best_y <= best_x:
            best_y = min(n_frame - margin if margin > 0 else n_frame - 1, best_x + max(margin, 1))
        
        best_wave = generate_square_wave(n_frame, best_x, best_y, wave_type='0-1-0')
        return int(best_x), int(best_y), best_mse, best_wave


def shift_square_wave(square_wave, wave_type, shifts):
    """
    Shift square wave transition points.
    
    Args:
        square_wave: original square wave array
        wave_type: square wave type
        shifts: Transition point shift frame count list (negative=forward, positive=backward)
                - For '0-1' or '1-0': [shift_1]
                - For '0-1-0': [shift_1, shift_2]
                - For '0-1-0-1-0': [shift_1, shift_2, shift_3, shift_4]
                - For '0-1-0-1-0-1-0': [shift_1, shift_2, shift_3, shift_4, shift_5, shift_6]
    
    Returns:
        Shifted square wave
    """
    if not shifts or all(s == 0 for s in shifts):
        return square_wave
    
    n_frame = len(square_wave)
    result = square_wave.copy()
    
    if wave_type in ('0-1', '1-0'):
        # Single transition point: find transition point position and shift
        if len(shifts) < 1:
            return square_wave
        shift_1 = shifts[0]
        if wave_type == '0-1':
            # Find position of first 1
            ones_idx = np.where(square_wave == 1.0)[0]
            if len(ones_idx) > 0:
                old_transition = ones_idx[0]
                new_transition = max(0, min(n_frame, old_transition + shift_1))
                result = np.zeros(n_frame, dtype=square_wave.dtype)
                result[new_transition:] = 1.0
        else:  # '1-0'
            # Find position of first 0
            zeros_idx = np.where(square_wave == 0.0)[0]
            if len(zeros_idx) > 0:
                old_transition = zeros_idx[0]
                new_transition = max(0, min(n_frame, old_transition + shift_1))
                result = np.ones(n_frame, dtype=square_wave.dtype)
                result[new_transition:] = 0.0
    
    elif wave_type == '0-1-0':
        # Two transition points: find two transition point positions and shift
        if len(shifts) < 2:
            return square_wave
        shift_1, shift_2 = shifts[0], shifts[1]
        ones_idx = np.where(square_wave == 1.0)[0]
        if len(ones_idx) > 0:
            old_start = ones_idx[0]
            old_end = ones_idx[-1] + 1
            new_start = max(0, min(n_frame, old_start + shift_1))
            new_end = max(0, min(n_frame, old_end + shift_2))
            result = np.zeros(n_frame, dtype=square_wave.dtype)
            if new_end > new_start:
                result[new_start:new_end] = 1.0
    
    elif wave_type == '0-1-0-1-0':
        # Multiple transition points: find all transition point positions and shift (4 transition points)
        if len(shifts) < 4:
            # if provided shift count is insufficient, only use first two (backward compatible)
            shifts = shifts + [0] * (4 - len(shifts))
        shift_1, shift_2, shift_3, shift_4 = shifts[0], shifts[1], shifts[2], shifts[3]
        ones_idx = np.where(square_wave == 1.0)[0]
        if len(ones_idx) > 0:
            # Find all 0-1 and 1-0 transition points
            transitions = []
            prev_val = square_wave[0]
            for i in range(1, n_frame):
                if square_wave[i] != prev_val:
                    transitions.append(i)
                    prev_val = square_wave[i]
            if len(transitions) >= 4:
                # Shift all transition points
                transitions[0] = max(0, min(n_frame, transitions[0] + shift_1))
                transitions[1] = max(0, min(n_frame, transitions[1] + shift_2))
                transitions[2] = max(0, min(n_frame, transitions[2] + shift_3))
                transitions[3] = max(0, min(n_frame, transitions[3] + shift_4))
                # Ensure order
                transitions = sorted(transitions)
                # Regenerate waveform
                result = np.zeros(n_frame, dtype=square_wave.dtype)
                if len(transitions) >= 4:
                    t0, t1, t2, t3 = transitions[0], transitions[1], transitions[2], transitions[3]
                    if t0 < t1 < t2 < t3:
                        result[t0:t1] = 1.0
                        result[t2:t3] = 1.0
    
    elif wave_type == '0-1-0-1-0-1-0':
        # Multiple transition points: find all transition point positions and shift (6 transition points)
        if len(shifts) < 6:
            # if provided shift count is insufficient, only use first two (backward compatible)
            shifts = shifts + [0] * (6 - len(shifts))
        shift_1, shift_2, shift_3, shift_4, shift_5, shift_6 = shifts[0], shifts[1], shifts[2], shifts[3], shifts[4], shifts[5]
        ones_idx = np.where(square_wave == 1.0)[0]
        if len(ones_idx) > 0:
            # Find all 0-1 and 1-0 transition points
            transitions = []
            prev_val = square_wave[0]
            for i in range(1, n_frame):
                if square_wave[i] != prev_val:
                    transitions.append(i)
                    prev_val = square_wave[i]
            if len(transitions) >= 6:
                # Shift all transition points
                transitions[0] = max(0, min(n_frame, transitions[0] + shift_1))
                transitions[1] = max(0, min(n_frame, transitions[1] + shift_2))
                transitions[2] = max(0, min(n_frame, transitions[2] + shift_3))
                transitions[3] = max(0, min(n_frame, transitions[3] + shift_4))
                transitions[4] = max(0, min(n_frame, transitions[4] + shift_5))
                transitions[5] = max(0, min(n_frame, transitions[5] + shift_6))
                # Ensure order
                transitions = sorted(transitions)
                # Regenerate waveform
                result = np.zeros(n_frame, dtype=square_wave.dtype)
                if len(transitions) >= 6:
                    t0, t1, t2, t3, t4, t5 = transitions[0], transitions[1], transitions[2], transitions[3], transitions[4], transitions[5]
                    if t0 < t1 < t2 < t3 < t4 < t5:
                        result[t0:t1] = 1.0
                        result[t2:t3] = 1.0
                        result[t4:t5] = 1.0
    
    elif wave_type == '1-0-1':
        # Two transition points: find two transition point positions and shift
        if len(shifts) < 2:
            return square_wave
        shift_1, shift_2 = shifts[0], shifts[1]
        zeros_idx = np.where(square_wave == 0.0)[0]
        if len(zeros_idx) > 0:
            old_start = zeros_idx[0]
            old_end = zeros_idx[-1] + 1
            new_start = max(0, min(n_frame, old_start + shift_1))
            new_end = max(0, min(n_frame, old_end + shift_2))
            result = np.ones(n_frame, dtype=square_wave.dtype)
            if new_end > new_start:
                result[new_start:new_end] = 0.0
    
    return result


def compute_hand_status(hand_pose_data, 
                        weight_tip_distance=0.5, weight_curvature=0.5,
                        thresh_high=0.8, thresh_low=0.0,
                        enhance_high=100.0, enhance_low=100.0,
                        use_mean_thresh=True, mean_thresh_offset=0.25,
                        sw_weight_high=1e18, sw_weight_low=0.0,
                        downsample=1,
                        wave_type='0-1-0',
                        transition_margin=0,
                        transition_shift_1=0,
                        transition_shift_2=0,
                        transition_shift_3=0,
                        transition_shift_4=0,
                        transition_shift_5=0,
                        transition_shift_6=0):
    """
    Compute hand status using square wave approximation.
    
    Steps:
    1. Compute weighted trajectory (tip_distance + curvature)
    2. Apply enhancement
    3. Optimize square wave
    4. Apply transition shifts
    5. Downsample (if downsample > 1)
    
    Args:
        downsample: Downsample factor (1 = no downsampling, 5 = take every 5th frame)
        wave_type: type of square wave ('0-1-0', '1-0', '0-1', '0' for all zeros, '1' for all ones)
        mean_thresh_offset: offset to subtract from mean when computing mean threshold
        transition_margin: minimum distance (frame) from boundaries for transition points.
                          if > 0, forces the specified number of transitions to exist.
        transition_shift_1-6: shift for transition points (negative=earlier, positive=later)
                             - For '0-1-0': uses shift_1, shift_2
                             - For '0-1-0-1-0': uses shift_1, shift_2, shift_3, shift_4
                             - For '0-1-0-1-0-1-0': uses shift_1, shift_2, shift_3, shift_4, shift_5, shift_6
    
    Returns:
        binary signal (1 = closed, 0 = open), optionally downsampled
    """
    n_frame = hand_pose_data.shape[0]
    
    # Special processing: all 0 or all 1
    if wave_type == '0':
        # Directly return all 0
        if downsample > 1:
            return np.zeros((n_frame + downsample - 1) // downsample, dtype=np.float32)
        else:
            return np.zeros(n_frame, dtype=np.float32)
    elif wave_type == '1':
        # Directly return all 1
        if downsample > 1:
            return np.ones((n_frame + downsample - 1) // downsample, dtype=np.float32)
        else:
            return np.ones(n_frame, dtype=np.float32)
    
    # Compute weighted trajectory
    trajectory = compute_weighted_trajectory(hand_pose_data, weight_tip_distance, weight_curvature)
    
    # Apply enhancement
    if enhance_high != 1.0 or enhance_low != 1.0:
        trajectory = enhance_trajectory(trajectory, thresh_high, thresh_low,
                                         enhance_high, enhance_low, use_mean_thresh, mean_thresh_offset)
    
    # Optimize square wave
    best_x, best_y, _, square_wave = optimize_square_wave(trajectory, thresh_high, thresh_low,
                                                 sw_weight_high, sw_weight_low,
                                                 wave_type=wave_type,
                                                 margin=transition_margin)
    
    # Apply transition shifts
    # Build shift list based on waveform type
    if wave_type == '0-1-0-1-0':
        shifts = [transition_shift_1, transition_shift_2, transition_shift_3, transition_shift_4]
    elif wave_type == '0-1-0-1-0-1-0':
        shifts = [transition_shift_1, transition_shift_2, transition_shift_3, transition_shift_4, transition_shift_5, transition_shift_6]
    elif wave_type == '0-1-0':
        shifts = [transition_shift_1, transition_shift_2]
    elif wave_type in ('0-1', '1-0'):
        shifts = [transition_shift_1]
    else:
        shifts = []
    
    if any(s != 0 for s in shifts):
        square_wave = shift_square_wave(square_wave, wave_type, shifts)
    
    # Downsample if needed
    if downsample > 1:
        result = square_wave[::downsample].astype(np.float32)
    else:
        result = square_wave.astype(np.float32)
    
    return result


class HandParams:
    """Store processing parameters for single hand"""
    def __init__(self,
                 weight_tip_distance=0.5, weight_curvature=0.5,
                 thresh_high=0.8, thresh_low=0.0,
                 enhance_high=100.0, enhance_low=100.0,
                 use_mean_thresh=True, mean_thresh_offset=0.25,
                 sw_weight_high=1e18, sw_weight_low=0.0,
                 wave_type='0-1-0', transition_margin=0,
                 transition_shift_1=0, transition_shift_2=0,
                 transition_shift_3=0, transition_shift_4=0,
                 transition_shift_5=0, transition_shift_6=0):
        self.weight_tip_distance = weight_tip_distance
        self.weight_curvature = weight_curvature
        self.thresh_high = thresh_high
        self.thresh_low = thresh_low
        self.enhance_high = enhance_high
        self.enhance_low = enhance_low
        self.use_mean_thresh = use_mean_thresh
        self.mean_thresh_offset = mean_thresh_offset
        self.sw_weight_high = sw_weight_high
        self.sw_weight_low = sw_weight_low
        self.wave_type = wave_type
        self.transition_margin = transition_margin
        self.transition_shift_1 = transition_shift_1  # First transition point shift frame count (negative=forward, positive=backward)
        self.transition_shift_2 = transition_shift_2  # Second transition point shift frame count
        self.transition_shift_3 = transition_shift_3  # Third transition point shift frame count (used for 0-1-0-1-0 and 0-1-0-1-0-1-0)
        self.transition_shift_4 = transition_shift_4  # Fourth transition point shift frame count (used for 0-1-0-1-0 and 0-1-0-1-0-1-0)
        self.transition_shift_5 = transition_shift_5  # Fifth transition point shift frame count (used for 0-1-0-1-0-1-0)
        self.transition_shift_6 = transition_shift_6  # Sixth transition point shift frame count (used for 0-1-0-1-0-1-0)
    
    def to_dict(self):
        return {
            'weight_tip_distance': self.weight_tip_distance,
            'weight_curvature': self.weight_curvature,
            'thresh_high': self.thresh_high,
            'thresh_low': self.thresh_low,
            'enhance_high': self.enhance_high,
            'enhance_low': self.enhance_low,
            'use_mean_thresh': self.use_mean_thresh,
            'mean_thresh_offset': self.mean_thresh_offset,
            'sw_weight_high': self.sw_weight_high,
            'sw_weight_low': self.sw_weight_low,
            'wave_type': self.wave_type,
            'transition_margin': self.transition_margin,
            'transition_shift_1': self.transition_shift_1,
            'transition_shift_2': self.transition_shift_2,
            'transition_shift_3': self.transition_shift_3,
            'transition_shift_4': self.transition_shift_4,
            'transition_shift_5': self.transition_shift_5,
            'transition_shift_6': self.transition_shift_6,
        }


def process_hdf5(raw_path, mid_path, output_path, 
                 left_params: HandParams = None,
                 right_params: HandParams = None,
                 downsample=1,
                 in_place=False):
    """
    process a single HDF5 file with separate parameters for left and right hands.
    
    Args:
        raw_path: Path to raw HDF5 file (for hand_pose data)
        mid_path: Path to mid HDF5 file (base file to copy)
        output_path: Path to output HDF5 file
        left_params: HandParams for left hand processing
        right_params: HandParams for right hand processing
        downsample: Downsample factor
        in_place: if True, skip copying (mid == target)
    """
    if left_params is None:
        left_params = HandParams()
    if right_params is None:
        right_params = HandParams()
    
    if not in_place:
        shutil.copy2(mid_path, output_path)
    
    with h5py.File(raw_path, 'r') as f_raw:
        left_hand_pose = f_raw['left_hand_pose'][:]
        right_hand_pose = f_raw['right_hand_pose'][:]
    
    original_frame = len(left_hand_pose)
    
    with h5py.File(output_path, 'r+') as f:
        left_status = compute_hand_status(
            left_hand_pose,
            weight_tip_distance=left_params.weight_tip_distance,
            weight_curvature=left_params.weight_curvature,
            thresh_high=left_params.thresh_high, thresh_low=left_params.thresh_low,
            enhance_high=left_params.enhance_high, enhance_low=left_params.enhance_low,
            use_mean_thresh=left_params.use_mean_thresh, mean_thresh_offset=left_params.mean_thresh_offset,
            sw_weight_high=left_params.sw_weight_high, sw_weight_low=left_params.sw_weight_low,
            downsample=downsample,
            wave_type=left_params.wave_type,
            transition_margin=left_params.transition_margin,
            transition_shift_1=left_params.transition_shift_1,
            transition_shift_2=left_params.transition_shift_2,
            transition_shift_3=left_params.transition_shift_3,
            transition_shift_4=left_params.transition_shift_4,
            transition_shift_5=left_params.transition_shift_5,
            transition_shift_6=left_params.transition_shift_6
        )
        right_status = compute_hand_status(
            right_hand_pose,
            weight_tip_distance=right_params.weight_tip_distance,
            weight_curvature=right_params.weight_curvature,
            thresh_high=right_params.thresh_high, thresh_low=right_params.thresh_low,
            enhance_high=right_params.enhance_high, enhance_low=right_params.enhance_low,
            use_mean_thresh=right_params.use_mean_thresh, mean_thresh_offset=right_params.mean_thresh_offset,
            sw_weight_high=right_params.sw_weight_high, sw_weight_low=right_params.sw_weight_low,
            downsample=downsample,
            wave_type=right_params.wave_type,
            transition_margin=right_params.transition_margin,
            transition_shift_1=right_params.transition_shift_1,
            transition_shift_2=right_params.transition_shift_2,
            transition_shift_3=right_params.transition_shift_3,
            transition_shift_4=right_params.transition_shift_4,
            transition_shift_5=right_params.transition_shift_5,
            transition_shift_6=right_params.transition_shift_6
        )
        
        hand_status = np.stack([left_status, right_status], axis=1)
        
        validated = False
        if 'local_timestamps_ns' in f:
            timestamps_frame = len(f['local_timestamps_ns'])
            hand_status_frame = len(hand_status)
            
            if hand_status_frame != timestamps_frame:
                raise ValueError(
                    f"Frame count mismatch! "
                    f"local_timestamps_ns has {timestamps_frame} frame, "
                    f"but hand_status has {hand_status_frame} frame. "
                    f"(raw hand_pose: {original_frame} frame, downsampled by 5 = {hand_status_frame})"
                )
            validated = True
        
        if 'hand_status' in f:
            del f['hand_status']
        
        f.create_dataset('hand_status', data=hand_status)
        
        # Add processing metadata
        f['hand_status'].attrs['method'] = 'square_wave_approximation'
        f['hand_status'].attrs['downsample'] = downsample
        f['hand_status'].attrs['description'] = 'Binary hand status via square wave approximation: 1=closed, 0=open. Columns: [left, right]'
        
        # Left hand params
        for key, val in left_params.to_dict().items():
            f['hand_status'].attrs[f'left_{key}'] = val
        
        # Right hand params
        for key, val in right_params.to_dict().items():
            f['hand_status'].attrs[f'right_{key}'] = val
    
    return len(hand_status), validated


def _process_single_file(args):
    """Wrapper function for multiprocessing single file"""
    (raw_path, mid_path, output_path, rel_path,
     left_params_dict, right_params_dict,
     downsample, in_place) = args
    
    left_params = HandParams(**left_params_dict)
    right_params = HandParams(**right_params_dict)
    
    try:
        n_frame, validated = process_hdf5(
            raw_path, mid_path, output_path,
            left_params=left_params,
            right_params=right_params,
            downsample=downsample,
            in_place=in_place
        )
        return {
            'status': 'success',
            'rel_path': str(rel_path),
            'n_frame': n_frame,
            'validated': validated
        }
    except Exception as e:
        return {
            'status': 'error',
            'rel_path': str(rel_path),
            'error': str(e)
        }


def add_hand_params(parser, prefix, desc):
    """Add arguments for single hand"""
    parser.add_argument(f'--{prefix}_weight_tip_distance', type=float, default=0.5,
                        help=f'{desc} weight for tip_distance metric (default: 0.5)')
    parser.add_argument(f'--{prefix}_weight_curvature', type=float, default=0.5,
                        help=f'{desc} weight for curvature metric (default: 0.5)')
    parser.add_argument(f'--{prefix}_thresh_high', type=float, default=0.8,
                        help=f'{desc} threshold for high values (default: 0.8)')
    parser.add_argument(f'--{prefix}_thresh_low', type=float, default=0.0,
                        help=f'{desc} threshold for low values (default: 0.0)')
    parser.add_argument(f'--{prefix}_enhance_high', type=float, default=100.0,
                        help=f'{desc} enhancement factor for high values (default: 100.0)')
    parser.add_argument(f'--{prefix}_enhance_low', type=float, default=100.0,
                        help=f'{desc} enhancement factor for low values (default: 100.0)')
    parser.add_argument(f'--{prefix}_use_mean_thresh', action='store_true', default=True,
                        help=f'{desc} use mean height as enhancement threshold (default: True)')
    parser.add_argument(f'--{prefix}_no_mean_thresh', action='store_false', dest=f'{prefix}_use_mean_thresh',
                        help=f'{desc} disable mean height threshold')
    parser.add_argument(f'--{prefix}_mean_thresh_offset', type=float, default=0.25,
                        help=f'{desc} offset for mean threshold (default: 0.25)')
    parser.add_argument(f'--{prefix}_sw_weight_high', type=float, default=1e18,
                        help=f'{desc} weight for values close to 1 in square wave optimization (default: 1e18)')
    parser.add_argument(f'--{prefix}_sw_weight_low', type=float, default=0.0,
                        help=f'{desc} weight for values close to 0 in square wave optimization (default: 0.0)')
    parser.add_argument(f'--{prefix}_wave_type', type=str, default='0-1-0',
                        choices=['0-1-0', '1-0', '0-1', '0', '1', '0-1-0-1-0', '0-1-0-1-0-1-0'],
                        help=f'{desc} square wave type: 0-1-0/1-0/0-1 for transitions, 0 for all zeros, 1 for all ones, 0-1-0-1-0/0-1-0-1-0-1-0 for multi-transition (default: 0-1-0)')
    parser.add_argument(f'--{prefix}_transition_margin', type=int, default=0,
                        help=f'{desc} minimum distance from boundaries for transition points (default: 0)')
    parser.add_argument(f'--{prefix}_transition_shift_1', type=int, default=0,
                        help=f'{desc} shift for first transition point in frame (negative=earlier, positive=later, default: 0)')
    parser.add_argument(f'--{prefix}_transition_shift_2', type=int, default=0,
                        help=f'{desc} shift for second transition point in frame (negative=earlier, positive=later, default: 0)')
    parser.add_argument(f'--{prefix}_transition_shift_3', type=int, default=0,
                        help=f'{desc} shift for third transition point in frame, for 0-1-0-1-0 and 0-1-0-1-0-1-0 (negative=earlier, positive=later, default: 0)')
    parser.add_argument(f'--{prefix}_transition_shift_4', type=int, default=0,
                        help=f'{desc} shift for fourth transition point in frame, for 0-1-0-1-0 and 0-1-0-1-0-1-0 (negative=earlier, positive=later, default: 0)')
    parser.add_argument(f'--{prefix}_transition_shift_5', type=int, default=0,
                        help=f'{desc} shift for fifth transition point in frame, for 0-1-0-1-0-1-0 (negative=earlier, positive=later, default: 0)')
    parser.add_argument(f'--{prefix}_transition_shift_6', type=int, default=0,
                        help=f'{desc} shift for sixth transition point in frame, for 0-1-0-1-0-1-0 (negative=earlier, positive=later, default: 0)')


def get_hand_params_from_args(args, prefix):
    """Extract single hand parameters from command line arguments"""
    return HandParams(
        weight_tip_distance=getattr(args, f'{prefix}_weight_tip_distance'),
        weight_curvature=getattr(args, f'{prefix}_weight_curvature'),
        thresh_high=getattr(args, f'{prefix}_thresh_high'),
        thresh_low=getattr(args, f'{prefix}_thresh_low'),
        enhance_high=getattr(args, f'{prefix}_enhance_high'),
        enhance_low=getattr(args, f'{prefix}_enhance_low'),
        use_mean_thresh=getattr(args, f'{prefix}_use_mean_thresh'),
        mean_thresh_offset=getattr(args, f'{prefix}_mean_thresh_offset'),
        sw_weight_high=getattr(args, f'{prefix}_sw_weight_high'),
        sw_weight_low=getattr(args, f'{prefix}_sw_weight_low'),
        wave_type=getattr(args, f'{prefix}_wave_type'),
        transition_margin=getattr(args, f'{prefix}_transition_margin'),
        transition_shift_1=getattr(args, f'{prefix}_transition_shift_1'),
        transition_shift_2=getattr(args, f'{prefix}_transition_shift_2'),
        transition_shift_3=getattr(args, f'{prefix}_transition_shift_3'),
        transition_shift_4=getattr(args, f'{prefix}_transition_shift_4'),
        transition_shift_5=getattr(args, f'{prefix}_transition_shift_5'),
        transition_shift_6=getattr(args, f'{prefix}_transition_shift_6'),
    )


def main():
    parser = argparse.ArgumentParser(
        description='Add hand_status to HDF5 file using square wave approximation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--raw', type=str, required=True, 
                        help='Raw source directory (for hand_pose data)')
    parser.add_argument('--mid', type=str, required=True, 
                        help='Mid source directory (base HDF5 file to copy)')
    parser.add_argument('--target', type=str, required=True, 
                        help='Target directory for output HDF5 file')
    
    # Left hand parameters
    add_hand_params(parser, 'left', 'Left hand:')
    
    # Right hand parameters
    add_hand_params(parser, 'right', 'Right hand:')
    
    # Downsample (totalusing)
    parser.add_argument('--downsample', type=int, default=5,
                        help='Downsample factor for output (1 = no downsampling, 5 = take every 5th frame, default: 5)')
    
    parser.add_argument('--recursive', action='store_true',
                        help='Search for HDF5 file recursively')
    
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers (default: 1, single process)')
    
    args = parser.parse_args()
    
    raw_dir = Path(args.raw)
    mid_dir = Path(args.mid)
    target_dir = Path(args.target)
    
    if not raw_dir.exists():
        print(f"Error: Raw source directory not found: {raw_dir}")
        return
    if not mid_dir.exists():
        print(f"Error: Mid source directory not found: {mid_dir}")
        return
    
    in_place = mid_dir.resolve() == target_dir.resolve()
    
    if not in_place:
        target_dir.mkdir(parents=True, exist_ok=True)
    
    if args.recursive:
        hdf5_file = list(mid_dir.glob('**/*.hdf5'))
    else:
        hdf5_file = list(mid_dir.glob('*.hdf5'))
    
    if not hdf5_file:
        print(f"No HDF5 file found in {mid_dir}")
        return
    
    # Extract left and right hand parameters
    left_params = get_hand_params_from_args(args, 'left')
    right_params = get_hand_params_from_args(args, 'right')
    
    print("=" * 80)
    print("Add Hand Status (Square Wave Approximation)")
    print("=" * 80)
    print(f"Raw source (hand_pose): {raw_dir}")
    print(f"Mid source (base file): {mid_dir}")
    print(f"Target:                 {target_dir}")
    if in_place:
        print(f"Mode:   IN-PLACE (mid == target, no copy)")
    print(f"Files:  {len(hdf5_file)}")
    print("-" * 80)
    print(f"{'Parameter':<25} {'Left Hand':<25} {'Right Hand':<25}")
    print("-" * 80)
    print(f"{'weight_tip_distance':<25} {left_params.weight_tip_distance:<25} {right_params.weight_tip_distance:<25}")
    print(f"{'weight_curvature':<25} {left_params.weight_curvature:<25} {right_params.weight_curvature:<25}")
    print(f"{'thresh_high':<25} {left_params.thresh_high:<25} {right_params.thresh_high:<25}")
    print(f"{'thresh_low':<25} {left_params.thresh_low:<25} {right_params.thresh_low:<25}")
    print(f"{'enhance_high':<25} {left_params.enhance_high:<25} {right_params.enhance_high:<25}")
    print(f"{'enhance_low':<25} {left_params.enhance_low:<25} {right_params.enhance_low:<25}")
    print(f"{'use_mean_thresh':<25} {str(left_params.use_mean_thresh):<25} {str(right_params.use_mean_thresh):<25}")
    print(f"{'mean_thresh_offset':<25} {left_params.mean_thresh_offset:<25} {right_params.mean_thresh_offset:<25}")
    print(f"{'sw_weight_high':<25} {left_params.sw_weight_high:<25.2e} {right_params.sw_weight_high:<25.2e}")
    print(f"{'sw_weight_low':<25} {left_params.sw_weight_low:<25.2e} {right_params.sw_weight_low:<25.2e}")
    print(f"{'wave_type':<25} {left_params.wave_type:<25} {right_params.wave_type:<25}")
    print(f"{'transition_margin':<25} {left_params.transition_margin:<25} {right_params.transition_margin:<25}")
    print(f"{'transition_shift_1':<25} {left_params.transition_shift_1:<25} {right_params.transition_shift_1:<25}")
    print(f"{'transition_shift_2':<25} {left_params.transition_shift_2:<25} {right_params.transition_shift_2:<25}")
    print(f"{'transition_shift_3':<25} {left_params.transition_shift_3:<25} {right_params.transition_shift_3:<25}")
    print(f"{'transition_shift_4':<25} {left_params.transition_shift_4:<25} {right_params.transition_shift_4:<25}")
    print(f"{'transition_shift_5':<25} {left_params.transition_shift_5:<25} {right_params.transition_shift_5:<25}")
    print(f"{'transition_shift_6':<25} {left_params.transition_shift_6:<25} {right_params.transition_shift_6:<25}")
    print("-" * 80)
    print(f"Downsample: {args.downsample}x" if args.downsample > 1 else "Downsample: None (1:1)")
    print(f"Workers: {args.num_workers}")
    print("=" * 80)
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    # Prepare task list
    tasks = []
    left_params_dict = left_params.to_dict()
    right_params_dict = right_params.to_dict()
    
    for mid_path in hdf5_file:
        rel_path = mid_path.relative_to(mid_dir)
        raw_path = raw_dir / rel_path
        
        if in_place:
            output_path = mid_path
        else:
            output_path = target_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not raw_path.exists():
            print(f"  ⊘ {rel_path}: raw file not found, skipping")
            skip_count += 1
            continue
        
        tasks.append((
            raw_path, mid_path, output_path, rel_path,
            left_params_dict, right_params_dict,
            args.downsample, in_place
        ))
    
    if args.num_workers > 1 and len(tasks) > 1:
        # Multiprocess processing
        print(f"🚀 use {args.num_workers} processes for parallel processing {len(tasks)} files")
        with processPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(_process_single_file, task): task[3] for task in tasks}
            
            with tqdm(total=len(futures), desc="processing") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result['status'] == 'success':
                        if result['validated']:
                            tqdm.write(f"  ✓ {result['rel_path']}: {result['n_frame']} frame, timestamps validated")
                        else:
                            tqdm.write(f"  ✓ {result['rel_path']}: {result['n_frame']} frame")
                        success_count += 1
                    else:
                        tqdm.write(f"  ✗ {result['rel_path']}: {result['error']}")
                        error_count += 1
                    pbar.update(1)
    else:
        # Single process processing
        for task in tqdm(tasks, desc="processing"):
            raw_path, mid_path, output_path, rel_path = task[:4]
            
            try:
                n_frame, validated = process_hdf5(
                    raw_path, mid_path, output_path,
                    left_params=left_params,
                    right_params=right_params,
                    downsample=args.downsample,
                    in_place=in_place
                )
                if validated:
                    tqdm.write(f"  ✓ {rel_path}: {n_frame} frame, timestamps validated")
                else:
                    tqdm.write(f"  ✓ {rel_path}: {n_frame} frame")
                success_count += 1
            except Exception as e:
                tqdm.write(f"  ✗ {rel_path}: {e}")
                error_count += 1
    
    print("=" * 60)
    print(f"Done! Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")
    print(f"output saved to: {target_dir}")


if __name__ == '__main__':
    main()
