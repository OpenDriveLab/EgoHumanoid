# Human Data Processing Pipeline

A multi-stage pipeline for processing raw human motion capture data into robot-ready datasets. It reorders raw episodes, computes navigation commands from body pose trajectories, downsamples to a target frequency, integrates ZED camera images, and computes binary hand open/close status into the final HDF5 files.

## Pipeline Overview

```
Raw Data (dated batch folders)
        |
[Step 1: Reorder]       -> Chronologically sorted episode_N.hdf5 + episode_N.svo2
        |
[Step 2: Navigation]    -> Smoothed trajectory + velocity commands (vx, vy, yaw_rate)
        |
[Step 3: Downsample]    -> Downsampled episodes with discrete teleop commands
        |
[Step 4: Merge Camera]  -> Final HDF5 with synchronized left/right camera images
        |
[Step 5: Hand Status]   -> Binary hand open/close status via square wave approximation
        |
Robot-Ready Dataset
```

### Step 1 — Reorder Episodes

Scans `{date}_{batch}/` subdirectories in the input folder, sorts episodes chronologically, and copies them with sequential numbering into `{output_dir}/hdf5/` and `{output_dir}/svo2/`.

### Step 2 — Navigation Pipeline

Reads `body_pose` skeleton keypoints (pelvis + hip landmarks) from each HDF5 file, applies coordinate transforms, smooths the trajectory with a Savitzky-Golay filter, estimates tangent directions, and generates velocity commands `[vx, vy, yaw_rate]` in the local body frame. Optionally produces PNG comparison plots for validation.

### Step 3 — Downsample

Reduces the data frequency by averaging navigation commands over a sliding window (default factor: 5). Also produces discretized `teleop_navigate_command` values via thresholding and computes `delta_height` between frames.

### Step 4 — Merge Camera

Reads binocular frames from ZED SVO2 files, matches them to downsampled HDF5 timestamps via binary search, compresses images as JPEG (quality 95), and writes synchronized left/right images into the final output HDF5 files.

### Step 5 — Hand Status

Computes binary hand open/close status from raw `left_hand_pose` and `right_hand_pose` data using square wave approximation. The pipeline:

1. Computes a weighted combination of finger tip distance and finger curvature metrics
2. Enhances the trajectory to push values toward 0 (open) and 1 (closed)
3. Fits an optimal square wave via `scipy.optimize.minimize`
4. Downsamples to match the target frequency

The result is written as `hand_status` with shape `(M, 2)` (columns: left, right) into the final HDF5 files. Supports per-hand configuration of wave type (`0-1-0`, `1-0`, `0-1`, `0`, `1`, etc.) and manual transition point adjustments. See `add_hand_status.py --help` for all options.

## Dependencies

### System Requirements

- **Python** >= 3.8
- **ZED SDK** (required for Step 4 — reading SVO2 camera files)
- A conda environment with the ZED Python API installed (the shell script assumes a `zed` conda environment)

### Python Packages

```bash
pip install numpy scipy h5py matplotlib pyyaml pillow opencv-python tqdm
```

| Package | Used For |
|---------|----------|
| `numpy` | Array operations and numerical computation |
| `scipy` | Savitzky-Golay filtering, spatial rotations |
| `h5py` | Reading and writing HDF5 datasets |
| `matplotlib` | Trajectory validation plots (PNG export) |
| `pyyaml` | Parsing configuration files |
| `pillow` | Image processing |
| `opencv-python` | JPEG/video encoding (used in camera merge and video export) |
| `tqdm` | Progress bars |

The ZED Python API (`pyzed`) is installed via the ZED SDK and is **not** available through pip:

```bash
# Follow the ZED SDK installation guide:
# https://www.stereolabs.com/docs/installation
# Then install the Python API:
python /usr/local/zed/get_python_api.py
```

### Optional (for additional utilities)

| Package | Used For |
|---------|----------|
| `plotly` | Interactive 3D visualization in `process_human_eef_pipeline.py` |

```bash
pip install plotly
```

## Usage

### Quick Start

```bash
./run_human_data_pipeline.sh \
  --input_dir /path/to/raw_data \
  --output_dir /path/to/intermediate \
  --final-output-dir /path/to/final \
  --file all
```

### Full Options

```
Usage:
  ./run_human_data_pipeline.sh --input_dir <DIR> --output_dir <DIR> --final-output-dir <DIR> [OPTIONS]

Required:
  --input_dir <path>          Raw data directory (containing {date}_{batch}/ subdirectories)
  --output_dir <path>         Intermediate output directory (hdf5/ and svo2/ created here)
  --final-output-dir <path>   Final output directory (merged HDF5 with camera images)

Optional:
  --file <type>               File type filter: hdf5, svo2, or all (default: all)
  --workers <n>               Number of parallel workers (default: 32)
  --baseline-sec <sec>        Trajectory smoothing window in seconds (default: 15)
  --tangent-lag <frames>      Tangent direction estimation lag in frames (default: 5)
  --downsample-rate <rate>    Downsampling factor (default: 5)
  --no-overwrite              Do not overwrite existing output files
  --with-png                  Generate trajectory validation PNG plots
  --skip-reorder              Skip Step 1 (reorder)
  --skip-navigation           Skip Step 2 (navigation command injection)
  --skip-downsample           Skip Step 3 (downsampling)
  --skip-merge                Skip Step 4 (camera image merge)
  --skip-hand-status          Skip Step 5 (hand status computation)
  --dry-run                   Preview mode — print commands without executing
  -h, --help                  Show help message
```

### Examples

Run the full pipeline:

```bash
./run_human_data_pipeline.sh \
  --input_dir /data/raw \
  --output_dir /data/processed \
  --final-output-dir /data/final
```

Skip reorder (already reordered) and generate validation plots:

```bash
./run_human_data_pipeline.sh \
  --input_dir /data/raw \
  --output_dir /data/processed \
  --final-output-dir /data/final \
  --skip-reorder \
  --with-png
```

Dry run to preview what would be executed:

```bash
./run_human_data_pipeline.sh \
  --input_dir /data/raw \
  --output_dir /data/processed \
  --final-output-dir /data/final \
  --dry-run
```

Run only navigation + downsample (skip reorder, merge and hand status):

```bash
./run_human_data_pipeline.sh \
  --input_dir /data/raw \
  --output_dir /data/processed \
  --final-output-dir /data/final \
  --skip-reorder \
  --skip-merge \
  --skip-hand-status
```

Run only hand status on existing final data:

```bash
./run_human_data_pipeline.sh \
  --input_dir /data/raw \
  --output_dir /data/processed \
  --final-output-dir /data/final \
  --skip-reorder \
  --skip-navigation \
  --skip-downsample \
  --skip-merge
```

### Running Individual Scripts

Each stage can also be run independently:

```bash
# Step 1: Reorder
python scripts/reorder_episodes_for_raw.py \
  --input_dir /data/raw --output_dir /data/processed --file all --workers 32

# Step 2: Navigation
python process_navigation_pipeline.py \
  --dataset-dir /data/processed --baseline-sec 15 --tangent-lag 5 --overwrite --no-png

# Step 3: Downsample
python downsample_episode.py \
  --dataset-dir /data/processed --downsample-rate 5 --overwrite

# Step 4: Merge camera
python merge_camera_only.py \
  --dataset-dir /data/processed --output-dir /data/final --num-workers 32

# Step 5: Hand status
python add_hand_status.py \
  --raw /data/processed/hdf5 --mid /data/final --target /data/final \
  --downsample 5 --num_workers 32
```

### Configuration File

Default parameters for the navigation pipeline can be set in `configs/human_data_process_config.yaml`:

```yaml
baseline-sec: 15    # Smoothing window (seconds)
tangent-lag: 5      # Tangent estimation lag (frames)
overwrite: true
no-png: true
```

Command-line arguments override config file values.

## Input Data Format

The pipeline expects raw data organized in dated batch folders:

```
input_dir/
  2025-01-15_batch1/
    episode_0.hdf5
    episode_0.svo2
    episode_1.hdf5
    episode_1.svo2
    ...
  2025-01-15_batch2/
    ...
```

Each raw HDF5 file must contain:
- `body_pose`: `(N, num_joints, 7)` — skeleton keypoints `[x, y, z, qx, qy, qz, qw]`
- `left_hand_pose`: `(N, 26, ≥3)` — left hand joint positions (required for Step 5)
- `right_hand_pose`: `(N, 26, ≥3)` — right hand joint positions (required for Step 5)
- `local_timestamps_ns`: `(N,)` — nanosecond timestamps

## Output Data Format

Final HDF5 files contain:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `body_pose` | `(M, J, 7)` | Original skeleton keypoints (downsampled) |
| `navigation_command` | `(M, 3)` | Velocity commands `[vx, vy, yaw_rate]` |
| `teleop_navigate_command` | `(M, 3)` | Discretized teleop commands |
| `delta_height` | `(M,)` | Height change between consecutive frames |
| `observation_image_left` | `(M,)` | JPEG-compressed left camera frames |
| `observation_image_right` | `(M,)` | JPEG-compressed right camera frames |
| `camera_timestamp` | `(M,)` | Matched camera timestamps |
| `timestamp_diff_ms` | `(M,)` | Camera-data time sync error (ms) |
| `hand_status` | `(M, 2)` | Binary hand status `[left, right]`: 1 = closed, 0 = open |

## Additional Utilities

- **`add_hand_status.py`** — Computes binary hand open/close status from hand pose data via square wave approximation. Supports per-hand wave type configuration and manual transition shifts. Can also be run standalone (see `add_hand_status.py --help`).
- **`process_human_eef_pipeline.py`** — Computes hand end-effector poses (7D) and deltas (6D) from skeleton data. Supports interactive 3D visualization via Plotly.
- **`export_videos.py`** — Exports camera frames from HDF5 files to MP4 videos (20 FPS, half resolution).
- **`scripts/reorder_episodes_for_downsample.py`** — Reorders already-downsampled episodes for sequential numbering.
