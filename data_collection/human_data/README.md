# PICO + ZED Mini Data Collection

A multi-modal data collection system based on PICO VR and ZED Mini depth camera, supporting synchronized full-body tracking, hand tracking, and video recording.

## Features

- **Full-body tracking**: Real-time capture of 24 body joint poses
- **Hand tracking**: Capture 26 joints per hand (left and right)
- **Controller tracking**: Capture left/right controller poses
- **Video recording**: ZED Mini records SVO2 format video (with depth)
- **Real-time visualization**: MeshCat 3D skeleton visualization + optional ZED camera preview
- **Synchronized saving**: PICO tracking data and ZED video saved per episode

## Hardware Requirements

- PICO VR headset (with full-body tracking support)
- ZED Mini depth camera
- PC with Linux (Ubuntu 22.04/24.04)

## Installation

See [install.md](install.md) for full setup instructions.

## Quick Start

```bash
conda activate humandata

# Basic collection
python scripts/human_data_collection.py

# With ZED camera preview window
python scripts/human_data_collection.py --visualize-zed

# Specify save directory and dataset name
python scripts/human_data_collection.py --data-dir my_data --name my_recording
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | `body_data` | Dataset name |
| `--data-dir` | `data_collection` | Data save directory |
| `--dataset-name` | same as `--name` | Dataset subdirectory name |
| `--interval` | `0.01` | Collection interval (seconds), ~100Hz |
| `--visualize-zed`, `-v` | off | Enable ZED camera live preview |
| `--no-depth` | off | Disable depth recording |
| `--episode` | `0` | Starting episode index |

## Collection Workflow

1. Launch the program
2. System initializes (PICO SDK + ZED Mini + MeshCat)
3. Enter an episode index (e.g., 0, 1, 2...)
4. Synchronized collection begins (PICO data + ZED video + MeshCat visualization)
5. Press **Space** to finish the current episode
6. HDF5 + SVO2 files are saved automatically
7. Continue to the next episode or press **Ctrl+C** to exit

## Data Format

### Directory Structure

```
data_collection/
└── body_data/
    ├── episode_0.hdf5    # PICO tracking data
    ├── episode_0.svo2    # ZED Mini video
    ├── episode_1.hdf5
    ├── episode_1.svo2
    └── ...
```

### HDF5 File Structure

| Dataset | Shape | Description |
|---------|-------|-------------|
| `body_pose` | (frames, 24, 7) | Body joint poses [x, y, z, qx, qy, qz, qw] |
| `left_controller_pose` | (frames, 7) | Left controller pose |
| `right_controller_pose` | (frames, 7) | Right controller pose |
| `left_hand_pose` | (frames, 26, 7) | Left hand joint poses |
| `right_hand_pose` | (frames, 26, 7) | Right hand joint poses |
| `left_hand_active` | (frames,) | Left hand tracking active status |
| `right_hand_active` | (frames,) | Right hand tracking active status |
| `local_timestamps_ns` | (frames,) | Local PC timestamps (nanoseconds) |

### Body Joint Index (24 joints)

```
 0: Pelvis              12: NECK
 1: LEFT_HIP            13: LEFT_COLLAR
 2: RIGHT_HIP           14: RIGHT_COLLAR
 3: SPINE1              15: HEAD
 4: LEFT_KNEE           16: LEFT_SHOULDER
 5: RIGHT_KNEE          17: RIGHT_SHOULDER
 6: SPINE2              18: LEFT_ELBOW
 7: LEFT_ANKLE          19: RIGHT_ELBOW
 8: RIGHT_ANKLE         20: LEFT_WRIST
 9: SPINE3              21: RIGHT_WRIST
10: LEFT_FOOT            22: LEFT_HAND
11: RIGHT_FOOT           23: RIGHT_HAND
```

## Utility Scripts

### Convert SVO2 to MP4

```bash
# Default: extract left camera view
python scripts/svo2_to_mp4.py episode_0.svo2

# Specify output path
python scripts/svo2_to_mp4.py episode_0.svo2 --output output.mp4

# Side-by-side left+right view
python scripts/svo2_to_mp4.py episode_0.svo2 --view side_by_side
```

## MeshCat Visualization

After launching the collection script, open a browser and visit:

```
http://localhost:7000/static/
```

The 3D viewer displays:
- Green sphere: Pelvis (root joint)
- Red sphere: Head
- Blue: Left side controller and hand
- Orange: Right side controller and hand
- White lines: Skeleton connections

## Troubleshooting

### ZED camera fails to open
- Check that the camera is properly connected
- Ensure you are using a USB 3.0 port
- Run `ZED_Explorer` to test the camera

### PICO data is all zeros
- Ensure the PICO device is connected and the XRoboToolkit PC Service is running
- Make sure full-body tracking is activated on the headset
- Stand in front of the device and wait for tracking initialization
