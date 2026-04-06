"""
Integrated data processing script（tangent trajectory -> navigation_command -> integration reconstruction -> output comparison PNG）

What it does:
  1) Read original keypoints from HDF5 body_pose
  2) Generate processed trajectory using tangent method:
       - processed/positions_xyz  (N,3)
       - processed/rotation_xy    (N,2)  tangent direction unit vector in xy plane
  3) Generate velocity commands based on processed data:
       - navigation_command       (N,3) [vx, vy, yaw_rate]
  4) Integrate navigation_command to reconstruct trajectory and output a PNG:
       Left: tangent trajectory（xy from processed/positions_xyz）
       Right: integrated trajectory（xy obtained from integrating navigation_command）

python process_navigation_pipeline.py --config configs/human_data_process_config.yaml

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib
import yaml

# Only output png, using non-GUI backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402


# ========== Coordinate system transformation（consistent with replay_pelvis_trajectory_tangent.py） ==========
RX90 = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)
RZ90 = np.array(
    [
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
T_BODY = RZ90 @ RX90


def transform_coordinates_xyz(positions_xyz: np.ndarray) -> np.ndarray:
    """Apply coordinate system transformation, return xyz（with original script's translation offset）."""
    pos = np.asarray(positions_xyz, dtype=np.float64)
    if pos.ndim == 1:
        pose = np.eye(4)
        pose[:3, 3] = pos
        transformed_pose = T_BODY @ pose
        transformed_pos = transformed_pose[:3, 3].copy()
        transformed_pos[2] += 0.7
        transformed_pos[0] += 0.55
        return transformed_pos

    out = np.zeros((len(pos), 3), dtype=np.float64)
    for i in range(len(pos)):
        pose = np.eye(4)
        pose[:3, 3] = pos[i]
        transformed_pose = T_BODY @ pose
        transformed_pos = transformed_pose[:3, 3].copy()
        transformed_pos[2] += 0.7
        transformed_pos[0] += 0.55
        out[i] = transformed_pos
    return out


def _make_valid_savgol_params(n: int, window_length: int, polyorder: int) -> Tuple[int, int]:
    if n <= 0:
        raise ValueError("n must be positive")
    max_window = n if (n % 2 == 1) else (n - 1)
    wl = int(min(window_length, max_window))
    wl = max(3, wl)
    if wl % 2 == 0:
        wl -= 1
    wl = max(3, wl)
    po = int(max(1, polyorder))
    po = min(po, wl - 1)
    return wl, po


def smooth_xyz_savgol(xyz: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64)
    if len(pts) < 3:
        return pts
    wl, po = _make_valid_savgol_params(len(pts), window_length, polyorder)
    out = pts.copy()
    out[:, 0] = savgol_filter(out[:, 0], wl, po, mode="nearest")
    out[:, 1] = savgol_filter(out[:, 1], wl, po, mode="nearest")
    out[:, 2] = savgol_filter(out[:, 2], wl, po, mode="nearest")
    return out


def smooth_xy_savgol(xy: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64)
    if len(pts) < 3:
        return pts
    wl, po = _make_valid_savgol_params(len(pts), window_length, polyorder)
    out = pts.copy()
    out[:, 0] = savgol_filter(out[:, 0], wl, po, mode="nearest")
    out[:, 1] = savgol_filter(out[:, 1], wl, po, mode="nearest")
    return out


def _normalize_xy(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return v / n


def _make_heading_continuous(rot_xy: np.ndarray) -> np.ndarray:
    """Same as export_navigation_command.py: avoid 180° flip in tangent vectors."""
    r = _normalize_xy(rot_xy).copy()
    for i in range(1, len(r)):
        if float(np.dot(r[i - 1], r[i])) < 0.0:
            r[i] *= -1.0
    return r


def estimate_tangent_directions(xy: np.ndarray, lag: int) -> np.ndarray:
    """Local tangent direction d(t) for each frame, estimated using central difference and normalized."""
    pts = np.asarray(xy, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return np.tile(np.array([1.0, 0.0], dtype=np.float64), (n, 1))

    lag = int(max(1, lag))
    d = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        i0 = max(0, i - lag)
        i1 = min(n - 1, i + lag)
        diff = pts[i1] - pts[i0]
        norm = np.linalg.norm(diff)
        if norm < 1e-9:
            d[i] = d[i - 1] if i > 0 else np.array([1.0, 0.0], dtype=np.float64)
        else:
            d[i] = diff / norm
    return d


def compute_processed_tangent(
    body_pose_data: np.ndarray,
    dt: float,
    root: str,
    sg_window: int,
    sg_poly: int,
    baseline_sec: float,
    tangent_lag: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      positions_xyz: (N,3)  x/y=baseline_xy_abs, z=smoothed root z
      rotation_xy:   (N,2)  baseline 's tangent direction unit vector
    """
    n = len(body_pose_data)
    pelvis = body_pose_data[:, 0, :3]
    hip_l = body_pose_data[:, 1, :3]
    hip_r = body_pose_data[:, 2, :3]

    if root == "midhip":
        root_pos = 0.5 * (hip_l + hip_r)
    else:
        root_pos = pelvis

    root_xyz_abs = transform_coordinates_xyz(root_pos)
    root_xyz_abs_smoothed = smooth_xyz_savgol(root_xyz_abs, sg_window, sg_poly)
    root_xy_abs_smoothed = root_xyz_abs_smoothed[:, :2].copy()

    # Baseline path (low frequency): window determined by baseline_sec
    baseline_window = int(max(3, round(baseline_sec / max(dt, 1e-9))))
    if baseline_window % 2 == 0:
        baseline_window += 1
    baseline_xy_abs = smooth_xy_savgol(root_xy_abs_smoothed, baseline_window, 3)

    rotation_xy = estimate_tangent_directions(baseline_xy_abs, lag=tangent_lag)
    rotation_xy = _make_heading_continuous(rotation_xy)

    positions_xyz = np.zeros((n, 3), dtype=np.float64)
    positions_xyz[:, :2] = baseline_xy_abs
    positions_xyz[:, 2] = root_xyz_abs_smoothed[:, 2]
    return positions_xyz, rotation_xy


def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_navigation_command(positions_xyz: np.ndarray, rotation_xy: np.ndarray, dt: float) -> np.ndarray:
    """output (N,3) [vx, vy, yaw_rate],local coordinate frame using frame n's rotation_xy.
    
    For each position point, calculate velocity command from that point to next point.
    Last point's velocity command remains same as previous point (or set to 0).
    """
    pos = np.asarray(positions_xyz, dtype=np.float64)
    rot = _make_heading_continuous(np.asarray(rotation_xy, dtype=np.float64))
    n = int(min(len(pos), len(rot)))
    if n < 1:
        raise ValueError("need at least 1 frame to compute navigation_command")
    
    if n == 1:
        # Only one point, set velocity to 0
        return np.zeros((1, 3), dtype=np.float64)

    pos = pos[:n]
    rot = rot[:n]

    # Calculate displacement difference between adjacent frame (N-1 intervals)
    dp = pos[1:, :2] - pos[:-1, :2]  # world xy
    hx = rot[:-1, 0]
    hy = rot[:-1, 1]
    nx = -hy
    ny = hx

    dx_local = dp[:, 0] * hx + dp[:, 1] * hy
    dy_local = dp[:, 0] * nx + dp[:, 1] * ny

    # Δyaw: strictly calculate by SO(2) relative rotation
    cross = rot[:-1, 0] * rot[1:, 1] - rot[:-1, 1] * rot[1:, 0]
    dot = rot[:-1, 0] * rot[1:, 0] + rot[:-1, 1] * rot[1:, 1]
    dyaw = np.arctan2(cross, dot)

    vx = dx_local / dt
    vy = dy_local / dt
    wz = dyaw / dt
    
    # Extend to N velocity commands: last point uses previous point's velocity (or set to 0)
    vx_full = np.zeros((n,), dtype=np.float64)
    vy_full = np.zeros((n,), dtype=np.float64)
    wz_full = np.zeros((n,), dtype=np.float64)
    
    vx_full[:-1] = vx
    vy_full[:-1] = vy
    wz_full[:-1] = wz
    
    # Last point keeps previous point's velocity (or set to 0, here we choose to keep previous velocity)
    if n > 1:
        vx_full[-1] = vx[-1]
        vy_full[-1] = vy[-1]
        wz_full[-1] = wz[-1]
    
    return np.stack([vx_full, vy_full, wz_full], axis=1)


def integrate_nav_cmd(cmd: np.ndarray, dt: float, x0: float, y0: float, yaw0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate (N,3) velocity commands into (N,) x/y/yaw.
    
    Note: if cmd is (N,3), integration will yield (N,) position points.
    First position point is initial position (x0, y0, yaw0), subsequent N-1 position points obtained from integrating first N-1 velocity commands.
    Last velocity command (cmd[N-1]) will not be used, because only N position points are needed to match positions_xyz.
    """
    cmd = np.asarray(cmd, dtype=np.float64)
    if cmd.ndim != 2 or cmd.shape[1] != 3:
        raise ValueError(f"navigation_command expected shape (N,3), got {cmd.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    vx = cmd[:, 0]
    vy = cmd[:, 1]
    wz = cmd[:, 2]
    n = len(cmd)

    # Only use first n-1 velocity commands to integrate, obtaining n position points
    x = np.zeros((n,), dtype=np.float64)
    y = np.zeros((n,), dtype=np.float64)
    yaw = np.zeros((n,), dtype=np.float64)
    x[0], y[0], yaw[0] = float(x0), float(y0), float(yaw0)

    for k in range(n - 1):
        c = float(np.cos(yaw[k]))
        s = float(np.sin(yaw[k]))
        dx = (c * vx[k] - s * vy[k]) * dt
        dy = (s * vx[k] + c * vy[k]) * dt
        x[k + 1] = x[k] + dx
        y[k + 1] = y[k] + dy
        yaw[k + 1] = yaw[k] + wz[k] * dt

    return x, y, yaw


def to_frame0_xy(xy: np.ndarray, p0_xy: np.ndarray, h0_xy: np.ndarray) -> np.ndarray:
    “””
    Convert world frame xy point list to “first frame coordinate frame”:
      p' = R0^T (p - p0)
    where R0's x-axis is h0_xy, y-axis is its left normal.
    """
    p = np.asarray(xy, dtype=np.float64)
    p0 = np.asarray(p0_xy, dtype=np.float64).reshape(2)
    h = _normalize_xy(np.asarray(h0_xy, dtype=np.float64).reshape(1, 2))[0]
    n = np.array([-h[1], h[0]], dtype=np.float64)
    d = p - p0
    xp = d[:, 0] * h[0] + d[:, 1] * h[1]
    yp = d[:, 0] * n[0] + d[:, 1] * n[1]
    return np.stack([xp, yp], axis=1)


def write_h5_dataset(h5_path: str, key: str, data: np.ndarray, attrs: dict | None = None, overwrite: bool = False):
    """Write HDF5 (supports hierarchical keys with '/')."""
    if key is None or len(str(key).strip()) == 0:
        raise ValueError("HDF5 key is empty")

    key = str(key)
    with h5py.File(h5_path, "r+") as f:
        if "/" in key:
            grp_path, ds_name = key.rsplit("/", 1)
            grp = f.require_group(grp_path)
        else:
            grp = f
            ds_name = key

        if ds_name in grp:
            if not overwrite:
                raise RuntimeError(f"HDF5 dataset already exists: {key} (use --overwrite to replace)")
            del grp[ds_name]

        dset = grp.create_dataset(ds_name, data=np.asarray(data), compression="gzip", compression_opts=4)
        if attrs:
            for k, v in attrs.items():
                try:
                    dset.attrs[k] = v
                except Exception:
                    dset.attrs[k] = str(v)


def load_config(config_path: str) -> dict:
    """Load YAML config file and return dictionary."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config else {}


def main():
    # First parse --config argument (if provided)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="Config file path (YAML format)")
    pre_args, remaining_args = pre_parser.parse_known_args()

    # if config file provided, load config
    config_dict = {}
    if pre_args.config:
        config_dict = load_config(pre_args.config)
        # Convert YAML key names (with hyphens) to argparse parameter names (also with hyphens)
        # true/false in YAML need to be converted to boolean values
        for bool_key in ["overwrite", "dry-run", "no-png"]:
            if bool_key in config_dict:
                config_dict[bool_key] = bool(config_dict[bool_key])

    # Create main parser, using values from config file as defaults
    parser = argparse.ArgumentParser(description="process tangent trajectory -> navigation_command -> compare PNG")
    parser.add_argument("h5_file", nargs="?", default=None, help="Single file mode: HDF5 file path (Optional; if not provided then enable batch processing mode)")
    parser.add_argument("--dataset-dir", type=str, default=config_dict.get("dataset-dir"), help="batch processing mode: data root directory, will process all hdf5 files under {dataset-dir}/hdf5/")
    parser.add_argument("--pattern", type=str, default=config_dict.get("pattern", "episode_*.hdf5"), help="batch processing mode: episode file matching pattern (default episode_*.hdf5)")
    parser.add_argument("--out-dir", type=str, default=config_dict.get("out-dir"), help="batch processing mode: PNG output root directory (default write to same directory as each episode)")
    parser.add_argument("--dry-run", action="store_true", default=config_dict.get("dry-run", False), help="Only print file list to be processed, don't do any write/output")
    parser.add_argument("--max-file", type=int, default=config_dict.get("max-file", 0), help="Maximum number of episodes to process (0 means no limit)")
    parser.add_argument("--root", type=str, default=config_dict.get("root", "midhip"), choices=["pelvis", "midhip"], help="Root point: pelvis or midhip")

    # Trajectory baseline parameters: if config file has values, set as defaults, otherwise still required
    baseline_sec_default = config_dict.get("baseline-sec")
    tangent_lag_default = config_dict.get("tangent-lag")
    parser.add_argument("--baseline-sec", type=float, default=baseline_sec_default, required=baseline_sec_default is None, help="Tangent baseline smoothing window (seconds), e.g. 15")
    parser.add_argument("--tangent-lag", type=int, default=tangent_lag_default, required=tangent_lag_default is None, help="Tangent difference interval (frame), e.g. 5")

    parser.add_argument("--sg-window", type=int, default=config_dict.get("sg-window", 11), help="Savgol smoothing window (frame, odd number)")
    parser.add_argument("--sg-poly", type=int, default=config_dict.get("sg-poly", 3), help="Savgol polynomial order")
    parser.add_argument("--out", type=str, default=config_dict.get("out", "compare_trajectory.png"), help="Single file mode:output PNG path")
    parser.add_argument("--overwrite", action="store_true", default=config_dict.get("overwrite", False), help="Overwrite existing output keys in HDF5")
    parser.add_argument("--no-png", action="store_true", default=config_dict.get("no-png", False), help="Don't generate comparison PNG images")
    
    args = parser.parse_args(remaining_args)

    def process_one(h5_path: Path, out_png: Path, skip_png: bool = False):
        # check if file exists and is readable
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 filedoes not exist: {h5_path}")
        
        if not h5_path.is_file():
            raise ValueError(f"Path is not a file: {h5_path}")
        
        file_size = h5_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"HDF5 file is empty (size is 0): {h5_path}")
        
        # Try to open HDF5 file
        try:
            with h5py.File(str(h5_path), "r") as f:
                if "body_pose" not in f:
                    raise KeyError(f"missing dataset: body_pose in {h5_path}")
                body_pose_data = f["body_pose"][:]
                total_frame_attr = int(f.attrs.get("total_frame", len(body_pose_data)))
                dt = float(f.attrs.get("collection_interval_s", 0.01))
        except (OSError, IOError) as e:
            # HDF5 file corrupted or format error
            raise IOError(f"Cannot open HDF5 file (may be corrupted): {h5_path} (size: {file_size} bytes, Error: {e})")

        n = int(min(total_frame_attr, len(body_pose_data)))
        if n < 2:
            raise ValueError(f"need at least 2 frame, got {n} in {h5_path}")

        body_pose_data = body_pose_data[:n]

        # 1) Tangent processing to get processed trajectory
        positions_xyz, rotation_xy = compute_processed_tangent(
            body_pose_data=body_pose_data,
            dt=dt,
            root=args.root,
            sg_window=args.sg_window,
            sg_poly=args.sg_poly,
            baseline_sec=args.baseline_sec,
            tangent_lag=args.tangent_lag,
        )

        # 2) Generate velocity commands
        nav_cmd = compute_navigation_command(positions_xyz, rotation_xy, dt=dt)

        # 3) Integrate to reconstruct trajectory (first integrate in world frame, then uniformly convert to “first frame coordinate frame” for comparison plotting)
        x0, y0 = float(positions_xyz[0, 0]), float(positions_xyz[0, 1])
        yaw0 = float(np.arctan2(rotation_xy[0, 1], rotation_xy[0, 0]))
        x_int_w, y_int_w, _yaw_int = integrate_nav_cmd(nav_cmd, dt=dt, x0=x0, y0=y0, yaw0=yaw0)

        # 4) Write back to HDF5
        common_attrs = {
            "root": args.root,
            "mode": "tangent",
            "collection_interval_s": float(dt),
            "baseline_sec": float(args.baseline_sec),
            "tangent_lag": int(args.tangent_lag),
            "sg_window": int(args.sg_window),
            "sg_poly": int(args.sg_poly),
            "coordinate": "transformed_xyz",
        }

        write_h5_dataset(
            str(h5_path),
            "processed/positions_xyz",
            positions_xyz,
            attrs={**common_attrs, "description": "tangent baseline positions (xyz): x/y baseline, z smoothed root z"},
            overwrite=args.overwrite,
        )
        write_h5_dataset(
            str(h5_path),
            "processed/rotation_xy",
            rotation_xy,
            attrs={**common_attrs, "description": "tangent baseline heading (xy unit vectors)"},
            overwrite=args.overwrite,
        )
        write_h5_dataset(
            str(h5_path),
            "navigation_command",
            nav_cmd,
            attrs={
                "description": "navigation command for each frame n, local frame of frame n; columns=[vx, vy, yaw_rate]. Shape (N,3) where N matches positions_xyz.",
                "dt": float(dt),
                "units": "[m/s, m/s, rad/s]",
                "frame_convention": "x-axis=rotation_xy[n], y-axis=left-normal",
                "pos_key": "processed/positions_xyz",
                "rot_key": "processed/rotation_xy",
                "shape": str(nav_cmd.shape),
                **common_attrs,
            },
            overwrite=args.overwrite,
        )

        # 5) output PNG (both sides unified to frame0 coords)
        if not skip_png:
            p0_xy = np.array([positions_xyz[0, 0], positions_xyz[0, 1]], dtype=np.float64)
            h0_xy = np.array([rotation_xy[0, 0], rotation_xy[0, 1]], dtype=np.float64)
            processed_xy_f0 = to_frame0_xy(positions_xyz[:, :2], p0_xy=p0_xy, h0_xy=h0_xy)
            integrated_xy_f0 = to_frame0_xy(np.stack([x_int_w, y_int_w], axis=1), p0_xy=p0_xy, h0_xy=h0_xy)

            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            px = processed_xy_f0[:, 0]
            py = processed_xy_f0[:, 1]

            axs[0].plot(px, py, linewidth=2, color="tab:green")
            axs[0].scatter([px[0]], [py[0]], c="green", s=50, label="start")
            axs[0].scatter([px[-1]], [py[-1]], c="red", s=50, label="end")
            axs[0].set_title("tangent trajectory (frame0 coords)")
            axs[0].set_xlabel("x (m)")
            axs[0].set_ylabel("y (m)")
            axs[0].grid(True, alpha=0.3)
            axs[0].set_aspect("equal", adjustable="box")
            axs[0].legend(loc="best")

            axs[1].plot(integrated_xy_f0[:, 0], integrated_xy_f0[:, 1], linewidth=2, color="tab:blue")
            axs[1].scatter([integrated_xy_f0[0, 0]], [integrated_xy_f0[0, 1]], c="green", s=50, label="start")
            axs[1].scatter([integrated_xy_f0[-1, 0]], [integrated_xy_f0[-1, 1]], c="red", s=50, label="end")
            axs[1].set_title("integrated from navigation_command (frame0 coords)")
            axs[1].set_xlabel("x (m)")
            axs[1].set_ylabel("y (m)")
            axs[1].grid(True, alpha=0.3)
            axs[1].set_aspect("equal", adjustable="box")
            axs[1].legend(loc="best")

            fig.suptitle(f"Compare processed vs integrated (frame0 coords, dt={dt:g}s, frame={n})", fontsize=12)
            fig.tight_layout()
            fig.savefig(str(out_png), dpi=200)
            plt.close(fig)

    # ========== single file mode / batch processing mode ==========
    if args.h5_file:
        h5_path = Path(args.h5_file)
        if not h5_path.exists():
            raise FileNotFoundError(str(h5_path))
        if args.dry_run:
            print(f"[dry-run] would process: {h5_path}")
            if not args.no_png:
                print(f"[dry-run] would write PNG: {args.out}")
            return
        process_one(h5_path, Path(args.out), skip_png=args.no_png)
        print("✓ processing completed")
        print(f"  wrote: processed/positions_xyz, processed/rotation_xy, navigation_command (overwrite={args.overwrite})")
        if not args.no_png:
            print(f"  saved PNG: {args.out}")
        return

    # batch processing mode: must provide dataset-dir
    if not args.dataset_dir:
        parser.error("Please provide single file h5_file, or provide --dataset-dir to enable batch processing mode.")

    root_dir = Path(args.dataset_dir) / "hdf5"
    if not root_dir.exists():
        raise FileNotFoundError(str(root_dir))

    # Search for all hdf5 files directly under hdf5 directory
    all_eps = sorted(root_dir.glob(args.pattern))

    if args.max_file and args.max_file > 0:
        all_eps = all_eps[: args.max_file]

    print(f"batch processing:root={root_dir}")
    print(f"  episodes={len(all_eps)}, pattern={args.pattern}")

    if args.dry_run:
        for p in all_eps[:50]:
            print(f"[dry-run] {p}")
        if len(all_eps) > 50:
            print(f"[dry-run] ... and {len(all_eps) - 50} more")
        return

    out_root = Path(args.out_dir) if args.out_dir else None

    num_ok = 0
    num_fail = 0
    for ep_path in all_eps:
        try:
            if out_root:
                # outputto out_dir/episode_x_compare.png
                out_png = out_root / f"{ep_path.stem}_compare.png"
            else:
                # Default output to same directory as episode
                out_png = ep_path.parent / f"{ep_path.stem}_compare.png"

            if args.no_png:
                print(f"-> {ep_path}")
            else:
                print(f"-> {ep_path}  |  png={out_png}")
            process_one(ep_path, out_png, skip_png=args.no_png)
            num_ok += 1
        except Exception as e:
            num_fail += 1
            error_type = type(e).__name__
            print(f"[FAIL] {ep_path}: [{error_type}] {e}")

    print("✓ batch processing completed")
    print(f"  ok={num_ok}, fail={num_fail}, overwrite={args.overwrite}")
    if not args.no_png:
        if out_root:
            print(f"  png out root: {out_root}")
        else:
            print("  png out: alongside each episode file")
    else:
        print("  png: skipped (--no-png)")

if __name__ == "__main__":
    main()


