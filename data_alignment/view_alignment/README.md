# Viewport Transform

Single-image viewport transformation with hole filling, based on [GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/).

This tool shifts the camera viewpoint of egocentric images to create augmented training data. It is used to generate viewpoint-diverse versions of human demonstration data for humanoid robot learning.

## Pipeline

1. **MoGe** - Monocular depth prediction
2. **Cache3D** - 3D point cloud warping to new viewpoint
3. **Stable Diffusion Inpainting** - Fill disoccluded regions (holes)

## Dependencies

```bash
pip install torch numpy opencv-python h5py pillow tqdm einops warp-lang diffusers transformers moge-model psutil
```

## Usage

### Single H5 file
```bash
python viewport_transform_batch_h5.py \
    --h5_file /path/to/input.h5 \
    --image_key "observation_image_left" \
    --trajectory "down" \
    --movement_distance 0.07 \
    --output_dir ./output
```

### Directory of H5 files (multi-GPU)
```bash
python viewport_transform_batch_h5.py \
    --h5_dir /path/to/h5_directory \
    --batch_size 32 \
    --trajectory "down" \
    --movement_distance 0.07 \
    --num_gpus 4 \
    --output_dir /path/to/output
```

### Parallel batch processing

For processing multiple batches in parallel across multiple GPUs, you can run separate processes with different GPU assignments:

```bash
# GPU 0,1,2,3 process batch_000
CUDA_VISIBLE_DEVICES=0,1,2,3 python viewport_transform_batch_h5.py \
    --h5_dir /path/to/data/batch_000 \
    --batch_size 32 \
    --trajectory "down" \
    --movement_distance 0.07 \
    --num_gpus 4 \
    --output_dir /path/to/output/batch_000 &

# GPU 4,5,6,7 process batch_001
CUDA_VISIBLE_DEVICES=4,5,6,7 python viewport_transform_batch_h5.py \
    --h5_dir /path/to/data/batch_001 \
    --batch_size 32 \
    --trajectory "down" \
    --movement_distance 0.07 \
    --num_gpus 4 \
    --output_dir /path/to/output/batch_001 &

# Wait for all tasks to complete
wait
```

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--h5_file` / `--h5_dir` | Input H5 file or directory | - |
| `--image_key` | Key for image data in HDF5 | `observation_image_left` |
| `--trajectory` | Camera direction: `left`, `right`, `up`, `down`, `forward`, `backward` | `down` |
| `--movement_distance` | Camera movement distance | `0.1` |
| `--movement_distance_noise` | Random perturbation per sample | `0.02` |
| `--batch_size` | Frames per batch | `1` |
| `--num_gpus` | Number of GPUs | `1` |
| `--sd_model` | SD Inpainting model | `stabilityai/stable-diffusion-2-inpainting` |
| `--save_h5` | Save as H5 (replacing original images) | `false` |

## Acknowledgement

The 3D warping code (Cache3D, camera utilities, forward warping) is adapted from [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) under the Apache 2.0 License.
