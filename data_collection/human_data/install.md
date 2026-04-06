# Installation Guide

## 1. Conda Environment

```bash
conda create -n humandata python=3.12 -y
conda activate humandata
```

## 2. ZED SDK

Download and install the ZED SDK (CUDA 12, Ubuntu 22.04):

```bash
wget https://download.stereolabs.com/zedsdk/5.2/cu12/ubuntu22 -O zed_sdk.run
chmod +x zed_sdk.run
./zed_sdk.run
```

After installation, install the ZED Python API into the conda environment:

```bash
conda activate humandata
cd /usr/local/zed
python3 get_python_api.py
```

## 3. Python Dependencies

```bash
pip install -r requirements.txt
```

## 4. XRoboToolkit-PC-Service-Pybind (Python SDK)

Follow the instructions at: https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind

## 5. XRoboToolkit-PC-Service (PC)

Download and install the deb package:

```bash
wget https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_24.04_amd64.deb
```

## 6. XRoboToolkit-PICO (PICO Device)

Download and install the APK on your PICO device:

https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk
