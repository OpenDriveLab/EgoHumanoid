# Decoupled WBC

Software stack for loco-manipulation experiments across multiple humanoid platforms, with primary support for the Unitree G1. This repository provides whole-body control policies, a teleoperation stack, and a data exporter.

---

## System Installation

### Prerequisites
- Ubuntu 22.04
- NVIDIA GPU with a recent driver
- Docker and NVIDIA Container Toolkit (required for GPU access inside the container)

### Repository Setup

Install Git and Git LFS:

```bash
sudo apt update
sudo apt install git git-lfs
git lfs install
```

Clone the repository:

```bash
cd data
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl/decoupled_wbc
cp  -r ../../teleop/ control/main/teleop

```

### Docker Environment

We provide a Docker image with all dependencies pre-installed.

To mount the `src/openpi` directory into the Docker container, apply the following changes to `decoupled_wbc/docker/run_docker.sh`:

1. After the `PROJECT_DIR` variable (~line 144), add:

```bash
# EgoHumanoid repository root (3 levels up from GR00T-WholeBodyControl)
REPO_ROOT_DIR="$(cd "$PROJECT_DIR/../../.." && pwd)"
OPENPI_DIR="$REPO_ROOT_DIR/src/openpi"
```

2. In the `DOCKER_RUN_ARGS` volume mount section (~line 412), add the following line after the `$PROJECT_DIR` mount:

```bash
    -v $OPENPI_DIR:$DOCKER_HOME_DIR/Projects/openpi
```

This expects the following directory structure:

```
EgoHumanoid/
├── src/
│   └── openpi/                         # Mounted to /root/Projects/openpi
├── data_collection/
│   └── robot_data/
│       └── GR00T-WholeBodyControl/
│           └── decoupled_wbc/
│               └── docker/
│                   └── run_docker.sh
```

Make sure `src/openpi` is present under the EgoHumanoid repo root before starting the container.

Install a fresh image and start a container:

```bash
./docker/run_docker.sh --install --root
```

This pulls the latest `decoupled_wbc` image from `docker.io/nvgear`.

Start or re-enter a container:

```bash
./docker/run_docker.sh --root
```

Inside the container, both projects will be available under `/root/Projects/`:
- `/root/Projects/GR00T-WholeBodyControl`
- `/root/Projects/openpi`

Use `--root` to run as the `root` user. To run as a normal user, build the image locally:

```bash
./docker/run_docker.sh --build
```


---

## Running the Control Test

Once inside the container, the control policies can be launched directly.

- Simulation:

```bash
python decoupled_wbc/control/main/teleop/run_g1_control_loop.py
```

- Real robot: Ensure the host machine network is configured per the [G1 SDK Development Guide](https://support.unitree.com/home/en/G1_developer) and set a static IP at `192.168.123.222`, subnet mask `255.255.255.0`:

```bash
python decoupled_wbc/control/main/teleop/run_g1_control_loop.py --interface real
```

Keyboard shortcuts (terminal window):
- `]`: Activate policy
- `o`: Deactivate policy
- `9`: Release / Hold the robot
- `w` / `s`: Move forward / backward
- `a` / `d`: Strafe left / right
- `q` / `e`: Rotate left / right
- `z`: Zero navigation commands
- `1` / `2`: Raise / lower the base height
- `backspace` (viewer): Reset the robot in the visualizer

---

## Running the Teleoperation Test

The teleoperation policy primarily uses Pico controllers for coordinated hand and body control. It also supports other teleoperation devices, including LeapMotion and HTC Vive with Nintendo Switch Joy-Con controllers.

Keep `run_g1_control_loop.py` running, and in another terminal run:

```bash
python decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py --hand_control_device=pico --body_control_device=pico
```

### Pico Setup and Controls

Configure the teleop app on your Pico headset by following the [XR Robotics guidelines](https://github.com/XR-Robotics).

The necessary PC software is pre-installed in the Docker container. Only the [XRoboToolkit-PC-Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service) component is needed.

Prerequisites: Connect the Pico to the same network as the host computer.

Controller bindings:
- `menu + left trigger`: Toggle lower-body policy
- `menu + right trigger`: Toggle upper-body policy
- `Left stick`: X/Y translation
- `Right stick`: Yaw rotation
- `L/R triggers`: Control hand grippers

Pico unit test:

```bash
python decoupled_wbc/control/teleop/streamers/pico_streamer.py
```

---

## Running the Data Collection Stack


Running the G1 Control
```bash
python decoupled_wbc/control/main/teleop/run_g1_control_loop.py --interface real --control-frequency 50 --with_hands
```

Running the G1 Teleoperation
```bash
python decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py --body-control-device pico --hand_control_device=pico --enable_real_device
```

Running the G1 Data Collection

**Before this step,you need to install the zed python binding in this docker environment**
[ZED SDK install guide](https://www.stereolabs.com/docs/development/zed-sdk)

```bash
python decoupled_wbc/control/main/teleop/zed_mini_run_g1_data_exporter.py --dataset-name {my_task}_{my_dataset} --visualize
```

Controller bindings:
- `menu + left trigger`: Toggle lower-body policy
- `menu + right trigger`: Toggle upper-body policy
- `Left stick`: X/Y translation
- `Right stick`: Yaw rotation
- `L/R triggers`: Control hand grippers
- `X/Y buttons`: Control robot square
- `A/B buttons`: Collect/Discard data