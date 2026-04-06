"""
Hand End-Effector (EEF) data processing script

Function: 
  1) Read left and right hand poses from HDF5 body_pose (index 22=left hand, 23=right hand)
  2) downsample and calculate action_eef and action_delta_eef
  3) Write to target h5 file（batch processing mode）or original file

Usage:
  # single file mode（Write to original file）
  python data_utils/process_human_eef_pipeline.py /cpfs01/shared/shimodi/data/debug/0113_test/episode_0.hdf5 --target /cpfs01/shared/shimodi/data/debug/0113_test_reorder/hdf5/downsample_episode_0.hdf5 --out ./vis.html
  
  # batch processing mode（Write to file with same name in target directory）
  python data_utils/process_human_eef_pipeline.py --data-dir /cpfs01/shared/shimodi/data/human/source/toy/version_6/reorder/hdf5 --target-dir /cpfs01/shared/shimodi/data/human/source/toy/version_6/final/ --overwrite
"""

from __future__ import annotations

import argparse
from concurrent.futures import processPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R


# Keypoint indices (second dimension of body_pose)
BASE_IDX = 3          # body base coordinate frame
LEFT_HAND_IDX = 22    # left hand
RIGHT_HAND_IDX = 23   # right hand


# ========== Pose transformation ==========

def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pose[:3]
    T[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
    return T


def matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    return np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_quat()])


def transform_to_base_frame(hand_pose: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """Transform hand pose from world coordinate frame to base coordinate frame"""
    n = len(hand_pose)
    result = np.zeros((n, 7), dtype=np.float64)
    for i in range(n):
        T_base = pose7_to_matrix(base_pose[i])
        T_hand = pose7_to_matrix(hand_pose[i])
        result[i] = matrix_to_pose7(np.linalg.inv(T_base) @ T_hand)
    return result


# ========== Savgol smoothing ==========

def _make_valid_savgol_params(n: int, window_length: int, polyorder: int) -> tuple:
    max_window = n if (n % 2 == 1) else (n - 1)
    wl = max(3, min(window_length, max_window))
    if wl % 2 == 0:
        wl -= 1
    return max(3, wl), min(max(1, polyorder), wl - 1)


def smooth_hand_pose(hand_pose: np.ndarray, sg_window: int, sg_poly: int, 
                     passes: int = 1) -> np.ndarray:
    """
    Smooth hand pose data
    
    Position: using Savitzky-Golay filtering
    Rotation: using exponential map method（filtering in tangent space, more suitable for SO(3) manifold）
    
    Args:
        hand_pose: (N, 7) pose data [x,y,z, qx,qy,qz,qw]
        sg_window: Savgol window size
        sg_poly: Savgol polynomial order
        passes: number of filtering passes, more passes means smoother
    """
    if len(hand_pose) < 3:
        return hand_pose
    
    wl, po = _make_valid_savgol_params(len(hand_pose), sg_window, sg_poly)
    result = hand_pose.copy()
    
    # multiple filtering passes
    for _ in range(passes):
        # Position smoothing: Savgol filtering
        for i in range(3):
            result[:, i] = savgol_filter(result[:, i], wl, po, mode="nearest")
        
        # Rotation smoothing: exponential map method
        result[:, 3:7] = smooth_quaternions_expmap(result[:, 3:7], wl, po)
    
    return result


def smooth_quaternions_expmap(quats: np.ndarray, sg_window: int, sg_poly: int) -> np.ndarray:
    """
    using exponential map to smooth quaternion sequence in tangent space
    
    Steps:
    1. Ensure quaternion signs are consistent (avoid jumps)
    2. Using first frame as reference, convert subsequent frames to tangent space (rotation vectors)
    3. Perform Savgol filtering in tangent space
    4. Map back to quaternions
    """
    n = len(quats)
    q = quats.copy()
    
    # Ensure quaternion signs are consistent
    for i in range(1, n):
        if np.dot(q[i-1], q[i]) < 0:
            q[i] = -q[i]
    
    # Use middle frame as reference (reduce boundary effects)
    ref_idx = n // 2
    q_ref = R.from_quat(q[ref_idx])
    
    # Convert to tangent space (rotation vectors relative to reference frame)
    rotvecs = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        q_i = R.from_quat(q[i])
        # Calculate relative rotation: q_ref^-1 * q_i
        rel_rot = q_ref.inv() * q_i
        rotvecs[i] = rel_rot.as_rotvec()
    
    # Smooth in tangent space
    for i in range(3):
        rotvecs[:, i] = savgol_filter(rotvecs[:, i], sg_window, sg_poly, mode="nearest")
    
    # Map back to quaternions
    result = np.zeros((n, 4), dtype=np.float64)
    for i in range(n):
        rel_rot = R.from_rotvec(rotvecs[i])
        q_smooth = q_ref * rel_rot
        result[i] = q_smooth.as_quat()
    
    return result


# ========== EEF process ==========

# Coordinate system transformation: first rotate +90° around world X axis, then rotate -90° around world Z axis
_R_x_90 = R.from_euler('x', 90, degrees=True)
_R_z_neg90 = R.from_euler('z', -90, degrees=True)
WORLD_TRANSFORM = _R_z_neg90 * _R_x_90  # Combined transformation
WORLD_TRANSFORM_MATRIX = WORLD_TRANSFORM.as_matrix()

# Hand's own coordinate frame adjustment: rotate 180° around its own X axis (applies to both left and right hands)
HAND_LOCAL_X180 = R.from_euler('x', 180, degrees=True)

# Left hand additional adjustment: rotate 180° around its own Z axis
LEFT_HAND_LOCAL_Z180 = R.from_euler('z', 180, degrees=True)


def apply_local_rotation(poses: np.ndarray, local_rot: R) -> np.ndarray:
    """
    Apply local coordinate frame rotation to pose (only change orientation, not position)
    
    Args:
        poses: (N, 7) pose data [x,y,z, qx,qy,qz,qw]
        local_rot: Local rotation (post-multiply)
    
    Returns:
        (N, 7) Transformed pose
    """
    result = poses.copy()
    for i in range(len(poses)):
        q_old = R.from_quat(poses[i, 3:7])
        result[i, 3:7] = (q_old * local_rot).as_quat()
    return result


def apply_world_transform(poses: np.ndarray) -> np.ndarray:
    """
    Apply world coordinate system transformation to pose
    Transformation: first rotate +90° around world X axis, then rotate -90° around world Z axis
    
    Args:
        poses: (N, 7) pose data [x,y,z, qx,qy,qz,qw]
    
    Returns:
        (N, 7) Transformed pose
    """
    n = len(poses)
    result = np.zeros((n, 7), dtype=np.float64)
    
    for i in range(n):
        # Transform position
        result[i, :3] = WORLD_TRANSFORM_MATRIX @ poses[i, :3]
        # Transform rotation
        q_old = R.from_quat(poses[i, 3:7])
        result[i, 3:7] = (WORLD_TRANSFORM * q_old).as_quat()
    
    return result


def compute_eef_in_base(body_pose_data: np.ndarray, sg_window: int = 0, sg_poly: int = 3,
                        sg_passes: int = 1) -> tuple:
    """
    Calculate hand EEF pose in base coordinate frame
    
    Processing pipeline:
    1. convert to base coordinate frame
    2. Apply world coordinate transformation (rotate +90° around X axis, -90° around Z axis)
    3. Optionalsmoothprocess
    
    Args:
        body_pose_data: (N, num_joints, 7) original body pose data
        sg_window: Savgol window size(0 means no smoothing)
        sg_poly: Savgol polynomial order
        sg_passes: Filtering passes (more passes means smoother, recommend 1-3)
    
    Returns:
        left_eef: (N, 7) left hand EEF [x,y,z, qx,qy,qz,qw]
        right_eef: (N, 7) right hand EEF
    """
    base_pose = body_pose_data[:, BASE_IDX, :].copy()
    left_eef = transform_to_base_frame(body_pose_data[:, LEFT_HAND_IDX, :].copy(), base_pose)
    right_eef = transform_to_base_frame(body_pose_data[:, RIGHT_HAND_IDX, :].copy(), base_pose)
    
    # Apply world coordinate transformation
    left_eef = apply_world_transform(left_eef)
    right_eef = apply_world_transform(right_eef)
    
    # Hand's own coordinate frame adjustment: rotate 180° around its own X axis (applies to both left and right hands)
    left_eef = apply_local_rotation(left_eef, HAND_LOCAL_X180)
    right_eef = apply_local_rotation(right_eef, HAND_LOCAL_X180)
    
    # Left hand additional adjustment: rotate 180° around its own Z axis
    left_eef = apply_local_rotation(left_eef, LEFT_HAND_LOCAL_Z180)
    
    # smoothprocess
    if sg_window > 0:
        left_eef = smooth_hand_pose(left_eef, sg_window, sg_poly, sg_passes)
        right_eef = smooth_hand_pose(right_eef, sg_window, sg_poly, sg_passes)
    
    return left_eef, right_eef


def pose7_to_transform(pose: np.ndarray) -> np.ndarray:
    """
    Convert 7D pose to 4x4 homogeneous transformation matrix
    
    Args:
        pose: (7,) [x, y, z, qx, qy, qz, qw] - scipy format
    
    Returns:
        (4, 4) homogeneous transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = pose[:3]
    T[:3, :3] = R.from_quat(pose[3:7]).as_matrix()
    return T


def transform_to_xyzrpy(T: np.ndarray) -> np.ndarray:
    """
    Extract xyz translation and rpy rotation from transformation matrix
    
    Args:
        T: (4, 4) transformation matrix
    
    Returns:
        (6,) [dx, dy, dz, roll, pitch, yaw] - rotation in radians
    """
    xyz = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler('xyz')
    return np.concatenate([xyz, rpy])


def compute_delta_eef_single(pose_prev: np.ndarray, pose_curr: np.ndarray) -> np.ndarray:
    """
    Calculate delta EEF for single hand (6D: dx,dy,dz,roll,pitch,yaw)
    
    calculatemethod:delta_T = T_prev^{-1} @ T_curr
    i.e. incremental transformation in previous frame coordinate frame
    
    Args:
        pose_prev: (7,) previous framepose [x,y,z, qx,qy,qz,qw]
        pose_curr: (7,) Current frame pose [x,y,z, qx,qy,qz,qw]
    
    Returns:
        (6,) [dx, dy, dz, roll, pitch, yaw]
    """
    T_prev = pose7_to_transform(pose_prev)
    T_curr = pose7_to_transform(pose_curr)
    
    # Compute incremental transformation:delta_T = T_prev^{-1} @ T_curr
    delta_T = np.linalg.inv(T_prev) @ T_curr
    
    return transform_to_xyzrpy(delta_T)


def compute_delta_from_eef(left_eef: np.ndarray, right_eef: np.ndarray) -> np.ndarray:
    """
    Calculate delta EEF from EEF pose
    
    Calculation method: delta_T = T_{t-1}^{-1} @ T_t (incremental transformation in previous frame coordinate frame)
    First frame filled with identity transformation [0,0,0,0,0,0, 0,0,0,0,0,0]
    
    Args:
        left_eef: (N, 7) left hand EEF [x,y,z, qx,qy,qz,qw]
        right_eef: (N, 7) right hand EEF
    
    Returns:
        (N, 12) delta EEF [left hand: dx,dy,dz,roll,pitch,yaw, right hand: dx,dy,dz,roll,pitch,yaw]
    """
    n = len(left_eef)
    delta_eef = np.zeros((n, 12), dtype=np.float64)
    
    # First frame remains all zeros (identity transformation)
    for i in range(1, n):
        delta_eef[i, :6] = compute_delta_eef_single(left_eef[i-1], left_eef[i])
        delta_eef[i, 6:] = compute_delta_eef_single(right_eef[i-1], right_eef[i])
    
    return delta_eef


def downsample_eef(left_eef: np.ndarray, right_eef: np.ndarray, 
                   target_n: int, factor: int = 5) -> tuple:
    """
    downsample EEF data
    
    Args:
        left_eef: (N, 7) originalleft hand EEF
        right_eef: (N, 7) originalright hand EEF
        target_n: Target frame count (determined based on target h5's local_timestamps_ns)
        factor: Downsample factor
    
    Returns:
        left_eef_ds: (target_n, 7)
        right_eef_ds: (target_n, 7)
    """
    # Generate downsample indices: 0, factor, 2*factor, ...
    indices = np.arange(target_n) * factor
    
    # Ensure indices don't exceed original data length
    max_idx = len(left_eef) - 1
    indices = np.clip(indices, 0, max_idx)
    
    left_eef_ds = left_eef[indices]
    right_eef_ds = right_eef[indices]
    
    return left_eef_ds, right_eef_ds


# ========== HDF5 Write ==========

def write_h5_dataset(h5_path: str, key: str, data: np.ndarray, attrs: dict = None, overwrite: bool = False):
    """Write HDF5 dataset"""
    with h5py.File(h5_path, "r+") as f:
        if "/" in key:
            grp_path, ds_name = key.rsplit("/", 1)
            grp = f.require_group(grp_path)
        else:
            grp, ds_name = f, key

        if ds_name in grp:
            if not overwrite:
                raise RuntimeError(f"HDF5 dataset already exists: {key} (use --overwrite)")
            del grp[ds_name]

        dset = grp.create_dataset(ds_name, data=data, compression="gzip", compression_opts=4)
        if attrs:
            for k, v in attrs.items():
                dset.attrs[k] = str(v) if isinstance(v, (list, tuple, np.ndarray)) else v


# ========== Visualization ==========

def generate_plotly_interactive(left_eef: np.ndarray, right_eef: np.ndarray,
                                 out_html: Path, dt: float):
    """Generate Plotly interactive visualization (only 3D view + world coordinate frame)"""
    import plotly.graph_objects as go
    
    left_pos, left_quat = left_eef[:, :3], left_eef[:, 3:7]
    right_pos, right_quat = right_eef[:, :3], right_eef[:, 3:7]
    n_frame = len(left_pos)
    
    step = max(1, n_frame // 150)
    indices = list(range(0, n_frame, step))
    if indices[-1] != n_frame - 1:
        indices.append(n_frame - 1)
    
    # Calculate display range (include origin and all data)
    all_pos = np.vstack([left_pos, right_pos, [[0, 0, 0]]])  # include origin
    pos_min = all_pos.min(axis=0)
    pos_max = all_pos.max(axis=0)
    data_range = max(pos_max - pos_min)
    margin = data_range * 0.15
    half_range = (data_range + margin) / 2
    
    # Use center of data + origin as view center
    x_center, y_center, z_center = (pos_min + pos_max) / 2
    x_range = [x_center - half_range, x_center + half_range]
    y_range = [y_center - half_range, y_center + half_range]
    z_range = [z_center - half_range, z_center + half_range]
    hand_axis_len = data_range * 0.06
    world_axis_len = data_range * 0.12  # World coordinate frame axis length
    
    print(f"  Generate visualization ({len(indices)} frame)...")
    
    def quat_to_matrix(quat):
        return R.from_quat(quat).as_matrix()
    
    fig = go.Figure()
    
    # === World coordinate frame (at origin, static) ===
    origin = np.array([0, 0, 0])
    for axis_idx, (color, name) in enumerate([('red', 'World X'), ('green', 'World Y'), ('blue', 'World Z')]):
        axis_dir = np.zeros(3)
        axis_dir[axis_idx] = world_axis_len
        fig.add_trace(go.Scatter3d(
            x=[origin[0], axis_dir[0]], y=[origin[1], axis_dir[1]], z=[origin[2], axis_dir[2]],
            mode='lines', line=dict(color=color, width=6), name=name, hoverinfo='skip'
        ))
    # World coordinate frame origin marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers',
        marker=dict(size=6, color='black'), name='Origin', hoverinfo='skip'
    ))
    
    # === Trajectory path (static, semi-transparent) ===
    for pos, name, color in [(left_pos, 'Left Path', 'rgba(100,149,237,0.3)'), 
                              (right_pos, 'Right Path', 'rgba(250,128,114,0.3)')]:
        fig.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                      mode='lines', line=dict(color=color, width=1.5), name=name, hoverinfo='skip'))
    
    # Legend: hand coordinate axis color description
    for color, name in [('red', 'Hand X'), ('green', 'Hand Y'), ('blue', 'Hand Z')]:
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                      line=dict(color=color, width=3), name=name))
    
    n_static = 9  # Static trace count: 3(world axes) + 1(origin) + 2(paths) + 3(legend)
    i0 = indices[0]
    
    # === Initial state of dynamic elements (hand trajectory + hand coordinate axes) ===
    for pos, quat, color in [(left_pos, left_quat, '#6495ED'), (right_pos, right_quat, '#FA8072')]:
        fig.add_trace(go.Scatter3d(x=pos[:1, 0], y=pos[:1, 1], z=pos[:1, 2],
                      mode='lines', line=dict(color=color, width=3), showlegend=False, hoverinfo='skip'))
        rot = quat_to_matrix(quat[i0])
        for axis_idx, c in enumerate(['red', 'green', 'blue']):
            end = pos[i0] + rot[:, axis_idx] * hand_axis_len
            fig.add_trace(go.Scatter3d(x=[pos[i0, 0], end[0]], y=[pos[i0, 1], end[1]], z=[pos[i0, 2], end[2]],
                          mode='lines', line=dict(color=c, width=5), showlegend=False, hoverinfo='skip'))
    
    # === Animation frames ===
    frame = []
    for idx, i in enumerate(indices):
        frame_data = []
        for pos, quat, color in [(left_pos, left_quat, '#6495ED'), (right_pos, right_quat, '#FA8072')]:
            frame_data.append(go.Scatter3d(x=pos[:i+1, 0], y=pos[:i+1, 1], z=pos[:i+1, 2],
                              mode='lines', line=dict(color=color, width=3)))
            rot = quat_to_matrix(quat[i])
            for axis_idx, c in enumerate(['red', 'green', 'blue']):
                end = pos[i] + rot[:, axis_idx] * hand_axis_len
                frame_data.append(go.Scatter3d(x=[pos[i, 0], end[0]], y=[pos[i, 1], end[1]], z=[pos[i, 2], end[2]],
                                  mode='lines', line=dict(color=c, width=5)))
        
        lp, rp = left_pos[i], right_pos[i]
        coord_text = f'Left: [{lp[0]:.3f}, {lp[1]:.3f}, {lp[2]:.3f}] | Right: [{rp[0]:.3f}, {rp[1]:.3f}, {rp[2]:.3f}]'
        frame.append(go.Frame(data=frame_data, name=str(idx), traces=list(range(n_static, n_static + len(frame_data))),
                      layout=go.Layout(annotations=[dict(text=coord_text, x=0.5, y=1.02, xref='paper', yref='paper',
                                       showarrow=False, font=dict(size=13))])))
    
    fig.frame = frame
    
    lp0, rp0 = left_pos[0], right_pos[0]
    init_coord = f'Left: [{lp0[0]:.3f}, {lp0[1]:.3f}, {lp0[2]:.3f}] | Right: [{rp0[0]:.3f}, {rp0[1]:.3f}, {rp0[2]:.3f}]'
    
    fig.update_layout(
        title=dict(text='EEF Trajectory Replay', x=0.5, font=dict(size=16)),
        annotations=[
            dict(text=init_coord, x=0.5, y=1.02, xref='paper', yref='paper', showarrow=False, font=dict(size=13)),
        ],
        scene=dict(
            domain=dict(x=[0, 1], y=[0.08, 0.95]),  # 3D view fills entire area
            xaxis=dict(range=x_range, title='X', autorange=False),
            yaxis=dict(range=y_range, title='Y', autorange=False),
            zaxis=dict(range=z_range, title='Z', autorange=False),
            aspectmode='cube',
            dragmode='orbit',
        ),
        dragmode='orbit',
        paper_bgcolor='white', plot_bgcolor='rgba(248,248,248,1)', font=dict(color='black'),
        legend=dict(x=0.01, y=0.98),
        updatemenus=[dict(type='buttons', showactive=False, y=0.02, x=0.35, buttons=[
            dict(label='▶ Play', method='animate', args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
            dict(label='⏸ Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False))]),
        ])],
        sliders=[dict(
            active=0, 
            yanchor='top', 
            xanchor='left', 
            currentvalue=dict(prefix='Frame: ', visible=True, xanchor='right'),
            transition=dict(duration=0),
            pad=dict(b=10, t=50), 
            len=0.5, 
            x=0.25, 
            y=0.0,
            steps=[dict(
                args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))], 
                label=f'{indices[k]}', 
                method='animate'
            ) for k in range(len(indices))]
        )],
        height=800, margin=dict(l=10, r=10, t=80, b=60),
    )
    
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs='cdn', full_html=True)
    print(f"  ✓ Visualization saved: {out_html}")


# ========== Main processing ==========

def process_one(src_h5_path: Path, target_h5_path: Path = None, out_vis_path: Path = None,
                overwrite: bool = False, sg_window: int = 51, sg_poly: int = 2,
                sg_passes: int = 1, downsample: int = 5):
    """
    processsingle HDF5 file
    
    Args:
        src_h5_path: Source data HDF5 file
        target_h5_path: Target HDF5 file (if None, then write to source file)
        out_vis_path: Visualization output path (if None, then don't generate)
        overwrite: Whether to overwrite existing datasets
        sg_window: Savgol smoothingwindow
        sg_poly: Savgol polynomial order
        sg_passes: filteringpasses（passesmore passes means smoother）
        downsample: Downsample factor
    """
    # Read source data
    with h5py.File(str(src_h5_path), "r") as f:
        # breakpoint() 
        if "body_pose" not in f:
            raise KeyError(f"missing dataset: body_pose in {src_h5_path}")
        body_pose_data = f["body_pose"][:]
        dt = float(f.attrs.get("collection_interval_s", 0.01))

    n_src = len(body_pose_data)
    if n_src < 2:
        raise ValueError(f"need at least 2 frame, got {n_src}")

    # Calculate EEF in base coordinate frame (after coordinate frame reorientation and smoothing)
    left_eef, right_eef = compute_eef_in_base(body_pose_data, sg_window, sg_poly, sg_passes)
    
    # Determine target file and target frame count
    write_h5_path = target_h5_path if target_h5_path else src_h5_path
    
    if target_h5_path:
        # Read local_timestamps_ns from target file to determine target frame count
        with h5py.File(str(target_h5_path), "r") as f:
            if "delta_height" not in f:
                raise KeyError(f"missing ''delta_height' in target file: {target_h5_path}")
            target_n = len(f["delta_height"])
        
        # downsample
        left_eef_ds, right_eef_ds = downsample_eef(left_eef, right_eef, target_n, downsample)
        
        # Concatenate as action_eef: (N, 14) = [left: x,y,z,qx,qy,qz,qw, right: x,y,z,qx,qy,qz,qw]
        action_eef = np.hstack([left_eef_ds, right_eef_ds])
        
        # Calculate delta from downsampled EEF
        delta_eef = compute_delta_from_eef(left_eef_ds, right_eef_ds)
        
        print(f"  downsample: {n_src} -> {target_n} (factor={downsample})")
    else:
        # No downsampling, directly use original data
        action_eef = np.hstack([left_eef, right_eef])
        delta_eef = compute_delta_from_eef(left_eef, right_eef)
        target_n = n_src
    
    # Write action_eef
    write_h5_dataset(
        str(write_h5_path), "action_eef", action_eef,
        attrs={
            "description": "EEF pose in base frame: [left: x,y,z,qx,qy,qz,qw, right: x,y,z,qx,qy,qz,qw]",
            "shape": action_eef.shape,
            "left_hand_idx": LEFT_HAND_IDX,
            "right_hand_idx": RIGHT_HAND_IDX,
            "sg_window": sg_window,
            "sg_poly": sg_poly,
            "downsample_factor": downsample if target_h5_path else 1,
        },
        overwrite=overwrite,
    )
    
    # Write action_delta_eef
    write_h5_dataset(
        str(write_h5_path), "action_delta_eef", delta_eef,
        attrs={
            "description": "delta EEF: [left: dx,dy,dz,roll,pitch,yaw, right: dx,dy,dz,roll,pitch,yaw]",
            "shape": delta_eef.shape,
            "dt": dt * (downsample if target_h5_path else 1),
            "units": "[m,m,m,rad,rad,rad] x 2",
        },
        overwrite=overwrite,
    )

    # Visualization (use downsampled data)
    if out_vis_path:
        if target_h5_path:
            generate_plotly_interactive(left_eef_ds, right_eef_ds, out_vis_path, dt * downsample)
        else:
            generate_plotly_interactive(left_eef, right_eef, out_vis_path, dt)
    
    return action_eef.shape, delta_eef.shape


def _process_one_wrapper(args_tuple):
    """Wrapper function for parallel processing"""
    src_path, target_path, out_vis, overwrite, sg_window, sg_poly, sg_passes, downsample = args_tuple
    try:
        eef_shape, delta_shape = process_one(
            src_path, target_path, out_vis, overwrite, sg_window, sg_poly, sg_passes, downsample
        )
        return (src_path.name, "ok", eef_shape, delta_shape)
    except Exception as e:
        return (src_path.name, "fail", type(e).__name__, str(e))


def main():
    parser = argparse.ArgumentParser(description="processhand EEF -> action_eef + action_delta_eef")
    parser.add_argument("h5_file", nargs="?", default=None, help="Single file mode: source HDF5 file path")
    parser.add_argument("--target", type=str, default=None, help="Single file mode: output HDF5 file path (if not specified then write to source file)")
    parser.add_argument("--data-dir", type=str, default=None, help="batch processing mode: source data directory")
    parser.add_argument("--target-dir", type=str, default=None, help="batch processing mode: target data directory (write h5 file with same name)")
    parser.add_argument("--pattern", type=str, default="episode_*.hdf5", help="File matching pattern")
    parser.add_argument("--out", type=str, default=None, help="Single file mode: visualization output path")
    parser.add_argument("--out-dir", type=str, default=None, help="batch processing mode: visualization output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data")
    parser.add_argument("--max-file", type=int, default=0, help="Maximum number of files to process (0=no limit)")
    parser.add_argument("--sg-window", type=int, default=51, help="Savgol smoothing window (odd number, 0=no smoothing)")
    parser.add_argument("--sg-poly", type=int, default=2, help="Savgol polynomial order")
    parser.add_argument("--sg-passes", type=int, default=2, help="Filtering passes (more passes means smoother, recommend 1-3)")
    parser.add_argument("--downsample", type=int, default=5, help="Downsample factor")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel worker processes")
    args = parser.parse_args()

    # single file mode
    if args.h5_file:
        src_path = Path(args.h5_file)
        if not src_path.exists():
            raise FileNotFoundError(str(src_path))
        
        target_path = Path(args.target) if args.target else None
        if target_path and not target_path.exists():
            raise FileNotFoundError(f"target file not found: {target_path}")
        
        out_vis = Path(args.out) if args.out else None
        eef_shape, delta_shape = process_one(src_path, target_path, out_vis, args.overwrite, 
                                              args.sg_window, args.sg_poly, args.sg_passes, args.downsample)
        print(f"✓ processing completed: {src_path}")
        print(f"  Write to: {target_path or src_path}")
        print(f"  action_eef: {eef_shape}, action_delta_eef: {delta_shape}")
        if out_vis:
            print(f"  Visualization: {out_vis}")
        return

    # batch processing mode
    if not args.data_dir:
        parser.error("Please provide h5_file or --data-dir")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(str(data_dir))

    target_dir = Path(args.target_dir) if args.target_dir else None
    if target_dir and not target_dir.exists():
        raise FileNotFoundError(f"target-dir not found: {target_dir}")

    all_file = sorted(data_dir.rglob(args.pattern))
    if args.max_file > 0:
        all_file = all_file[:args.max_file]

    print(f"batch processing: {data_dir} -> {target_dir or '(write to source file)'}")
    print(f"  found {len(all_file)} file, sg_window={args.sg_window}, sg_passes={args.sg_passes}, downsample={args.downsample}")
    print(f"  Number of parallel processes: {args.workers}")

    out_root = Path(args.out_dir) if args.out_dir else None
    
    # Prepare task list
    tasks = []
    skipped = 0
    for src_path in all_file:
        if target_dir:
            rel_path = src_path.relative_to(data_dir)
            target_path = target_dir / rel_path
            if not target_path.exists():
                skipped += 1
                continue
        else:
            target_path = None
        
        out_vis = (out_root / f"{src_path.stem}_traj.html") if out_root else None
        tasks.append((src_path, target_path, out_vis, args.overwrite, 
                      args.sg_window, args.sg_poly, args.sg_passes, args.downsample))
    
    if skipped > 0:
        print(f"  Skipped {skipped} files (target file does not exist)")
    
    if not tasks:
        print("No files to process")
        return

    num_ok, num_fail = 0, 0
    
    # Parallel processing
    with processPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one_wrapper, t): t[0].name for t in tasks}
        
        for future in as_completed(futures):
            result = future.result()
            name, status = result[0], result[1]
            
            if status == "ok":
                eef_shape, delta_shape = result[2], result[3]
                print(f"✓ {name}: action_eef={eef_shape}, action_delta_eef={delta_shape}")
                num_ok += 1
            else:
                err_type, err_msg = result[2], result[3]
                print(f"✗ {name}: [{err_type}] {err_msg}")
                num_fail += 1

    print(f"\nbatch processing completed: ok={num_ok}, fail={num_fail}, skipped={skipped}")


if __name__ == "__main__":
    main()
