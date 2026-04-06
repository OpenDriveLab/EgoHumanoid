"""
Single-image viewport transformation with hole filling using Stable Diffusion Inpainting.

This script combines:
1. MoGe for monocular depth prediction
2. Cache3D for 3D warping
3. Stable Diffusion Inpainting for hole filling (higher quality than OpenCV, faster than video generation)
"""

import argparse
import atexit
import os
import signal
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Filter multiprocessing resource_tracker warnings
warnings.filterwarnings("ignore", message="resource_tracker: process died unexpectedly")

# Global variables for tracking child processes for cleanup
_executor = None
_child_pids = []

# Add parent directory to path so viewport_transform package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from moge.model.v1 import MoGeModel
from viewport_transform.cache_3d import Cache3D_Buffer
from viewport_transform.camera_utils import generate_camera_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description="Viewport transformation with SD Inpainting")
    # Input: single file or directory (choose one)
    parser.add_argument("--h5_file", type=str, default=None, help="Path to a single HDF5 file")
    parser.add_argument("--h5_dir", type=str, default=None, help="Directory of HDF5 files (processes all .h5 files)")
    parser.add_argument("--image_key", type=str, default="observation_image_left", help="Key for image data in HDF5")
    parser.add_argument("--frame_index", type=int, default=None, help="Frame index to process (processes all if not specified)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (only effective when frame_index is not set)")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index (processes to last frame if not specified)")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of frames per batch")
    parser.add_argument("--output_dir", type=str, default="./output_sd_inpaint", help="Output directory")
    parser.add_argument("--save_h5", action="store_true", help="Save as H5 file (replacing the original image key)")
    parser.add_argument("--moge_model", type=str, default="Ruicheng/moge-vitl",
                       help="MoGe model path (HuggingFace ID or local path)")
    parser.add_argument("--trajectory", type=str, default="down",
                       choices=["left", "right", "up", "down", "forward", "backward"],
                       help="Camera movement direction")
    parser.add_argument("--movement_distance", type=float, default=0.1, help="Camera movement distance")
    parser.add_argument("--movement_distance_noise", type=float, default=0.02,
                       help="Random perturbation range for movement distance (independent per sample)")
    parser.add_argument("--sd_model", type=str, default="stabilityai/stable-diffusion-2-inpainting",
                       help="Stable Diffusion Inpainting model")
    parser.add_argument("--prompt", type=str, default="",
                       help="Inpainting prompt (text describing the scene)")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, distorted",
                       help="Negative prompt")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="SD inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show_time", action="store_true", help="Show timing statistics for each step")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    return parser.parse_args()


def get_h5_frame_count(h5_file: str, image_key: str) -> int:
    """Get the total number of frames in an HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        if image_key not in f:
            available_keys = list(f.keys())
            raise KeyError(f"Key '{image_key}' not found. Available keys: {available_keys}")
        return len(f[image_key])


def read_h5_image(h5_file: str, image_key: str, frame_index: int) -> np.ndarray:
    """Read an image from an HDF5 file."""
    with h5py.File(h5_file, 'r') as f:
        if image_key not in f:
            available_keys = list(f.keys())
            raise KeyError(f"Key '{image_key}' not found. Available keys: {available_keys}")

        data = f[image_key]
        if frame_index >= len(data):
            raise IndexError(f"Frame index {frame_index} out of range (max: {len(data)-1})")

        image_data = data[frame_index]

        # Check if data is compressed JPEG
        if len(image_data.shape) == 1:
            image_bgr = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            image_bgr = image_data

        # BGR -> RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb


def get_h5_files_in_dir(h5_dir: str) -> list:
    """Get all H5 files in a directory."""
    h5_files = []
    for filename in sorted(os.listdir(h5_dir)):
        if filename.endswith('.h5') or filename.endswith('.hdf5'):
            h5_files.append(os.path.join(h5_dir, filename))
    return h5_files


def copy_h5_with_replaced_images(
    src_h5_file: str,
    dst_h5_file: str,
    image_key: str,
    new_images: list,
):
    """
    Copy an H5 file and replace image data for the specified key.

    Args:
        src_h5_file: Source H5 file path
        dst_h5_file: Destination H5 file path
        image_key: Key of the image data to replace
        new_images: List of new images [N, H, W, 3] in RGB format
    """
    import shutil

    # Copy the entire file first
    shutil.copy2(src_h5_file, dst_h5_file)

    # Open and replace the specified key
    with h5py.File(dst_h5_file, 'r+') as f:
        if image_key not in f:
            raise KeyError(f"Key '{image_key}' not found in H5 file")

        # Check original data format (compressed or not)
        original_data = f[image_key]
        is_compressed = len(original_data[0].shape) == 1

        # Delete old data
        del f[image_key]

        if is_compressed:
            # Keep JPEG compression if original was compressed
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            new_dataset = f.create_dataset(image_key, (len(new_images),), dtype=dt)
            for i, img_rgb in enumerate(new_images):
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                _, encoded = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                new_dataset[i] = encoded.flatten()
        else:
            # Keep uncompressed format (BGR)
            new_data = np.stack([cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in new_images])
            f.create_dataset(image_key, data=new_data)


def predict_depth(image_rgb: np.ndarray, moge_model: MoGeModel, device: str = "cuda"):
    """Predict depth using MoGe."""
    h, w = image_rgb.shape[:2]

    # MoGe works best at 720x1280
    depth_pred_h, depth_pred_w = 720, 1280

    # Resize image for depth prediction
    image_resized = cv2.resize(image_rgb, (depth_pred_w, depth_pred_h))
    image_tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

    with torch.no_grad():
        output = moge_model.infer(image_tensor)

    depth = output["depth"]
    intrinsics = output["intrinsics"]
    mask = output.get("mask")

    # Handle invalid depth values
    if mask is not None:
        depth = torch.where(mask == 0, torch.tensor(1000.0, device=depth.device), depth)

    # Convert intrinsics to pixel coordinates
    intrinsics_pixel = intrinsics.clone()
    intrinsics_pixel[0, 0] *= depth_pred_w
    intrinsics_pixel[1, 1] *= depth_pred_h
    intrinsics_pixel[0, 2] *= depth_pred_w
    intrinsics_pixel[1, 2] *= depth_pred_h

    # Resize back to original dimensions
    height_scale = h / depth_pred_h
    width_scale = w / depth_pred_w

    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
    ).squeeze()

    # Adjust intrinsics
    intrinsics_final = intrinsics_pixel.clone()
    intrinsics_final[0, 0] *= width_scale
    intrinsics_final[1, 1] *= height_scale
    intrinsics_final[0, 2] *= width_scale
    intrinsics_final[1, 2] *= height_scale

    return depth_resized, mask, intrinsics_final


def predict_depth_batch(images_rgb: list, moge_model: MoGeModel, device: str = "cuda"):
    """
    Batch depth prediction using MoGe.

    Args:
        images_rgb: list of numpy arrays, each [H, W, 3]
        moge_model: MoGe model
        device: device string

    Returns:
        depths: list of depth tensors
        masks: list of mask tensors (or None)
        intrinsics_list: list of intrinsics tensors
    """
    if len(images_rgb) == 0:
        return [], [], []

    # Assume all images have the same dimensions
    h, w = images_rgb[0].shape[:2]
    batch_size = len(images_rgb)

    # MoGe works best at 720x1280
    depth_pred_h, depth_pred_w = 720, 1280

    # Resize and stack all images: [B, 3, H, W]
    images_resized = []
    for img in images_rgb:
        img_resized = cv2.resize(img, (depth_pred_w, depth_pred_h))
        img_tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1)
        images_resized.append(img_tensor)

    batch_tensor = torch.stack(images_resized, dim=0).to(device)  # [B, 3, H, W]

    with torch.no_grad():
        output = moge_model.infer(batch_tensor)

    depths_batch = output["depth"]          # [B, H, W]
    intrinsics_batch = output["intrinsics"] # [B, 3, 3]
    masks_batch = output.get("mask")        # [B, H, W] or None

    # Process per-frame results
    depths = []
    masks = []
    intrinsics_list = []

    height_scale = h / depth_pred_h
    width_scale = w / depth_pred_w

    for i in range(batch_size):
        depth = depths_batch[i]
        intrinsics = intrinsics_batch[i]
        mask = masks_batch[i] if masks_batch is not None else None

        # Handle invalid depth values
        if mask is not None:
            depth = torch.where(mask == 0, torch.tensor(1000.0, device=depth.device), depth)

        # Convert intrinsics to pixel coordinates
        intrinsics_pixel = intrinsics.clone()
        intrinsics_pixel[0, 0] *= depth_pred_w
        intrinsics_pixel[1, 1] *= depth_pred_h
        intrinsics_pixel[0, 2] *= depth_pred_w
        intrinsics_pixel[1, 2] *= depth_pred_h

        # Resize back to original dimensions
        depth_resized = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
        ).squeeze()

        # Adjust intrinsics
        intrinsics_final = intrinsics_pixel.clone()
        intrinsics_final[0, 0] *= width_scale
        intrinsics_final[1, 1] *= height_scale
        intrinsics_final[0, 2] *= width_scale
        intrinsics_final[1, 2] *= height_scale

        depths.append(depth_resized)
        masks.append(mask)
        intrinsics_list.append(intrinsics_final)

    return depths, masks, intrinsics_list


def warp_image_3d(
    image_rgb: np.ndarray,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    trajectory_type: str,
    movement_distance: float,
    device: str = "cuda"
):
    """Warp image using 3D Cache."""
    H, W = image_rgb.shape[:2]

    # Create initial camera matrix (identity)
    initial_w2c = torch.eye(4, dtype=torch.float32, device=device)

    # Prepare image tensor: [0,1] -> [-1,1], shape [B, C, H, W]
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    image_bchw = (image_tensor.unsqueeze(0) * 2 - 1).to(device)  # [-1, 1]

    # Prepare depth tensor: [B, 1, H, W]
    if depth.dim() == 2:
        depth_b1hw = depth.unsqueeze(0).unsqueeze(0)
    else:
        depth_b1hw = depth
    depth_b1hw = torch.nan_to_num(depth_b1hw, nan=1e4)
    depth_b1hw = torch.clamp(depth_b1hw, min=0, max=1e4)

    # Prepare intrinsics: [B, 3, 3]
    if intrinsics.dim() == 2:
        intrinsics_b33 = intrinsics.unsqueeze(0)
    else:
        intrinsics_b33 = intrinsics

    # Prepare w2c: [B, 4, 4]
    w2c_b44 = initial_w2c.unsqueeze(0)

    # Create 3D cache
    generator = torch.Generator(device=device).manual_seed(42)
    cache = Cache3D_Buffer(
        frame_buffer_max=2,
        generator=generator,
        noise_aug_strength=0.0,
        input_image=image_bchw,
        input_depth=depth_b1hw,
        input_w2c=w2c_b44,
        input_intrinsics=intrinsics_b33,
        filter_points_threshold=0.05,
        foreground_masking=False,
    )

    # Generate target camera position (only need 2 frames: start and target)
    generated_w2cs, generated_intrinsics = generate_camera_trajectory(
        trajectory_type=trajectory_type,
        initial_w2c=initial_w2c,
        initial_intrinsics=intrinsics,
        num_frames=2,
        movement_distance=movement_distance,
        camera_rotation="center_facing",
        center_depth=1.0,
        device=device if isinstance(device, str) else device.type,
    )

    # Render target viewpoint
    rendered_images, rendered_masks = cache.render_cache(
        generated_w2cs[:, 1:2],  # Take only target position
        generated_intrinsics[:, 1:2],
    )

    # Extract warped image and mask
    warped_image = rendered_images[0, 0, 0]  # [C, H, W], range [-1, 1]
    warped_mask = rendered_masks[0, 0, 0]    # [1, H, W]

    # Convert warped image: [-1,1] -> [0,255]
    warped_np = ((warped_image.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    # mask: 1 = has content, 0 = hole
    # output mask_np: 255 = has content, 0 = hole
    if warped_mask.dim() == 3:
        warped_mask = warped_mask[0]
    mask_np = (warped_mask.cpu().numpy() * 255).astype(np.uint8)

    return warped_np, mask_np


def inpaint_with_sd(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    sd_pipeline,
    prompt: str = "",
    negative_prompt: str = "blurry, low quality, distorted",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int = 42,
):
    """Inpaint using Stable Diffusion."""
    # SD Inpainting expects mask: white (255) = region to inpaint
    # Our mask: 255 = has content, 0 = hole
    # So we need to invert
    inpaint_mask = 255 - mask

    # Save original dimensions
    orig_H, orig_W = image_rgb.shape[:2]

    # Ensure image dimensions are multiples of 8 (SD requirement)
    new_H = (orig_H // 8) * 8
    new_W = (orig_W // 8) * 8

    image_resized = cv2.resize(image_rgb, (new_W, new_H))
    mask_resized = cv2.resize(inpaint_mask, (new_W, new_H))

    # Convert to PIL Image
    image_pil = Image.fromarray(image_resized)
    mask_pil = Image.fromarray(mask_resized)

    # Set random seed
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Run inpainting with specified output size
    result = sd_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_pil,
        mask_image=mask_pil,
        height=new_H,
        width=new_W,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Convert back to numpy
    result_np = np.array(result)

    # Restore original dimensions
    if result_np.shape[0] != orig_H or result_np.shape[1] != orig_W:
        result_np = cv2.resize(result_np, (orig_W, orig_H))

    return result_np


def inpaint_with_sd_batch(
    images_rgb: list,
    masks: list,
    sd_pipeline,
    prompt: str = "",
    negative_prompt: str = "blurry, low quality, distorted",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seeds: list = None,
):
    """
    Batch inpainting using Stable Diffusion.

    Args:
        images_rgb: list of numpy arrays [H, W, 3]
        masks: list of numpy arrays [H, W], 255=content, 0=hole
        sd_pipeline: diffusers pipeline
        prompt: text prompt
        negative_prompt: negative prompt
        num_inference_steps: number of inference steps
        guidance_scale: CFG scale
        seeds: list of seeds, one per image

    Returns:
        results: list of numpy arrays [H, W, 3]
    """
    batch_size = len(images_rgb)
    if batch_size == 0:
        return []

    # Save original dimensions (assume all images have the same size)
    orig_H, orig_W = images_rgb[0].shape[:2]

    # Ensure image dimensions are multiples of 8 (SD requirement)
    new_H = (orig_H // 8) * 8
    new_W = (orig_W // 8) * 8

    # Prepare batch inputs
    images_pil = []
    masks_pil = []
    generators = []

    for i in range(batch_size):
        # Invert mask: our mask 255=content, SD expects 255=inpaint region
        inpaint_mask = 255 - masks[i]

        # Resize
        image_resized = cv2.resize(images_rgb[i], (new_W, new_H))
        mask_resized = cv2.resize(inpaint_mask, (new_W, new_H))

        # Convert to PIL Image
        images_pil.append(Image.fromarray(image_resized))
        masks_pil.append(Image.fromarray(mask_resized))

        # Generator
        seed = seeds[i] if seeds is not None else 42 + i
        generators.append(torch.Generator(device="cuda").manual_seed(seed))

    # Batch inpainting
    results = sd_pipeline(
        prompt=[prompt] * batch_size,
        negative_prompt=[negative_prompt] * batch_size,
        image=images_pil,
        mask_image=masks_pil,
        height=new_H,
        width=new_W,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generators,
    ).images

    # Convert back to numpy and restore original dimensions
    results_np = []
    for result in results:
        result_np = np.array(result)
        if result_np.shape[0] != orig_H or result_np.shape[1] != orig_W:
            result_np = cv2.resize(result_np, (orig_W, orig_H))
        results_np.append(result_np)

    return results_np


def process_single_frame(
    frame_idx: int,
    h5_file: str,
    image_key: str,
    output_dir: str,
    moge_model,
    sd_pipeline,
    args,
    device: str,
):
    """Full pipeline for processing a single frame."""
    # 1. Read image
    image_rgb = read_h5_image(h5_file, image_key, frame_idx)

    # 2. Predict depth
    depth, _, intrinsics = predict_depth(image_rgb, moge_model, device)

    # 3. 3D Warping (with random perturbation)
    random_offset = np.random.uniform(-args.movement_distance_noise, args.movement_distance_noise)
    actual_movement_distance = args.movement_distance + random_offset

    warped_rgb, mask = warp_image_3d(
        image_rgb, depth, intrinsics,
        args.trajectory, actual_movement_distance, device
    )

    # 4. Inpainting
    if sd_pipeline is not None:
        result_rgb = inpaint_with_sd(
            warped_rgb, mask, sd_pipeline,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed + frame_idx,  # Different seed per frame
        )
    else:
        # Fallback to OpenCV inpainting
        inpaint_mask = 255 - mask
        result_bgr = cv2.inpaint(
            cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2BGR),
            inpaint_mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA
        )
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # 5. Save results (filename includes frame index)
    frame_prefix = f"frame_{frame_idx:06d}"

    # Save final result
    cv2.imwrite(
        os.path.join(output_dir, f"{frame_prefix}_result.jpg"),
        cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    )

    # Save comparison: original | warped | result
    comparison = np.hstack([image_rgb, warped_rgb, result_rgb])
    cv2.imwrite(
        os.path.join(output_dir, f"{frame_prefix}_comparison.jpg"),
        cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    )

    return result_rgb


def process_batch(
    batch_frames: list,
    h5_file: str,
    image_key: str,
    output_dir: str,
    moge_model,
    sd_pipeline,
    args,
    device: str,
    save_images: bool = True,
):
    """
    Process a batch of frames.

    Args:
        save_images: Whether to save images to files. When False, returns processed image list.

    Returns:
        time_stats: dict with timing for each step (moge, warp, inpaint)
        results_rgb: (only when save_images=False) list of processed images
    """
    batch_size = len(batch_frames)
    time_stats = {"moge": 0.0, "warp": 0.0, "inpaint": 0.0}

    # 1. Batch read images
    images_rgb = []
    for frame_idx in batch_frames:
        img = read_h5_image(h5_file, image_key, frame_idx)
        images_rgb.append(img)

    # 2. Batch depth prediction
    t0 = time.time()
    depths, _, intrinsics_list = predict_depth_batch(images_rgb, moge_model, device)
    time_stats["moge"] = time.time() - t0

    # 3. Per-frame 3D warping (independent random perturbation per sample)
    t0 = time.time()
    warped_images = []
    masks = []
    for i in range(batch_size):
        # Generate random movement_distance per sample
        random_offset = np.random.uniform(-args.movement_distance_noise, args.movement_distance_noise)
        actual_movement_distance = args.movement_distance + random_offset

        warped_rgb, mask = warp_image_3d(
            images_rgb[i], depths[i], intrinsics_list[i],
            args.trajectory, actual_movement_distance, device
        )
        warped_images.append(warped_rgb)
        masks.append(mask)
    time_stats["warp"] = time.time() - t0

    # 4. Batch SD Inpainting
    t0 = time.time()
    if sd_pipeline is not None:
        seeds = [args.seed + frame_idx for frame_idx in batch_frames]
        results_rgb = inpaint_with_sd_batch(
            warped_images, masks, sd_pipeline,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seeds=seeds,
        )
    else:
        results_rgb = []
        for i in range(batch_size):
            inpaint_mask = 255 - masks[i]
            result_bgr = cv2.inpaint(
                cv2.cvtColor(warped_images[i], cv2.COLOR_RGB2BGR),
                inpaint_mask,
                inpaintRadius=5,
                flags=cv2.INPAINT_TELEA
            )
            results_rgb.append(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    time_stats["inpaint"] = time.time() - t0

    # 5. Save results or return images
    if save_images:
        for i, frame_idx in enumerate(batch_frames):
            frame_prefix = f"frame_{frame_idx:06d}"

            cv2.imwrite(
                os.path.join(output_dir, f"{frame_prefix}_result.jpg"),
                cv2.cvtColor(results_rgb[i], cv2.COLOR_RGB2BGR)
            )

            comparison = np.hstack([images_rgb[i], warped_images[i], results_rgb[i]])
            cv2.imwrite(
                os.path.join(output_dir, f"{frame_prefix}_comparison.jpg"),
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            )
        return time_stats
    else:
        # Return processed images (for H5 saving)
        return time_stats, results_rgb


def cleanup_workers(graceful_timeout=2):
    """
    Clean up all child processes.

    Args:
        graceful_timeout: Wait time in seconds for graceful shutdown
    """
    global _executor, _child_pids

    # Try graceful executor shutdown first
    if _executor is not None:
        try:
            _executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            try:
                _executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        _executor = None

    # Give child processes time to exit
    time.sleep(0.5)

    # Force kill remaining child processes
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)

        # Send SIGTERM for graceful exit
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Wait briefly
        gone, alive = psutil.wait_procs(children, timeout=graceful_timeout)

        # Send SIGKILL to remaining processes
        for child in alive:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except ImportError:
        # psutil not installed, use os.kill
        import subprocess
        try:
            result = subprocess.run(
                ['pgrep', '-P', str(os.getpid())],
                capture_output=True, text=True
            )
            for pid in result.stdout.strip().split('\n'):
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        pass

            time.sleep(graceful_timeout)

            # SIGKILL remaining
            result = subprocess.run(
                ['pgrep', '-P', str(os.getpid())],
                capture_output=True, text=True
            )
            for pid in result.stdout.strip().split('\n'):
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception:
            pass

    # Clean up CUDA cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def signal_handler(signum, frame):
    """Signal handler."""
    print("\n\nInterrupt received, cleaning up child processes...")
    cleanup_workers()
    print("Cleanup complete, exiting")
    sys.exit(1)


# Global variables for multi-process workers
_worker_moge_model = None
_worker_sd_pipeline = None
_worker_device = None


def init_gpu_worker(gpu_id, moge_model_path, sd_model_path):
    """
    Initialize a GPU worker by loading models on the specified GPU.
    Called once per worker process at startup.
    """
    global _worker_moge_model, _worker_sd_pipeline, _worker_device

    _worker_device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    # Load MoGe model
    _worker_moge_model = MoGeModel.from_pretrained(moge_model_path).to(_worker_device)
    _worker_moge_model.eval()

    # Load SD Inpainting model
    try:
        from diffusers import StableDiffusionInpaintPipeline
        _worker_sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_path,
            torch_dtype=torch.float16,
        ).to(_worker_device)
        _worker_sd_pipeline.set_progress_bar_config(disable=True)
    except ImportError:
        _worker_sd_pipeline = None


def gpu_worker_process_batch(task):
    """
    GPU worker function for processing a single batch.
    task: (batch_frames, h5_file, image_key, output_dir, args_dict, save_images)
    """
    global _worker_moge_model, _worker_sd_pipeline, _worker_device

    # Support two task formats (backward compatibility)
    if len(task) == 5:
        batch_frames, h5_file, image_key, output_dir, args_dict = task
        save_images = True
    else:
        batch_frames, h5_file, image_key, output_dir, args_dict, save_images = task

    # Convert args_dict to a simple object
    class Args:
        pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)

    batch_size = len(batch_frames)
    time_stats = {"moge": 0.0, "warp": 0.0, "inpaint": 0.0}

    # 1. Batch read images
    images_rgb = []
    for frame_idx in batch_frames:
        img = read_h5_image(h5_file, image_key, frame_idx)
        images_rgb.append(img)

    # 2. Batch depth prediction
    t0 = time.time()
    depths, _, intrinsics_list = predict_depth_batch(images_rgb, _worker_moge_model, _worker_device)
    time_stats["moge"] = time.time() - t0

    # 3. Per-frame 3D warping (independent random perturbation per sample)
    t0 = time.time()
    warped_images = []
    masks = []
    for i in range(batch_size):
        # Generate random movement_distance per sample
        random_offset = np.random.uniform(-args.movement_distance_noise, args.movement_distance_noise)
        actual_movement_distance = args.movement_distance + random_offset

        warped_rgb, mask = warp_image_3d(
            images_rgb[i], depths[i], intrinsics_list[i],
            args.trajectory, actual_movement_distance, _worker_device
        )
        warped_images.append(warped_rgb)
        masks.append(mask)
    time_stats["warp"] = time.time() - t0

    # 4. Batch SD Inpainting
    t0 = time.time()
    if _worker_sd_pipeline is not None:
        seeds = [args.seed + frame_idx for frame_idx in batch_frames]
        results_rgb = inpaint_with_sd_batch(
            warped_images, masks, _worker_sd_pipeline,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seeds=seeds,
        )
    else:
        results_rgb = []
        for i in range(batch_size):
            inpaint_mask = 255 - masks[i]
            result_bgr = cv2.inpaint(
                cv2.cvtColor(warped_images[i], cv2.COLOR_RGB2BGR),
                inpaint_mask,
                inpaintRadius=5,
                flags=cv2.INPAINT_TELEA
            )
            results_rgb.append(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    time_stats["inpaint"] = time.time() - t0

    # 5. Save results or return images
    if save_images:
        for i, frame_idx in enumerate(batch_frames):
            frame_prefix = f"frame_{frame_idx:06d}"

            cv2.imwrite(
                os.path.join(output_dir, f"{frame_prefix}_result.jpg"),
                cv2.cvtColor(results_rgb[i], cv2.COLOR_RGB2BGR)
            )

            comparison = np.hstack([images_rgb[i], warped_images[i], results_rgb[i]])
            cv2.imwrite(
                os.path.join(output_dir, f"{frame_prefix}_comparison.jpg"),
                cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
            )
        return len(batch_frames), time_stats
    else:
        # Return processed images (for H5 saving)
        return batch_frames, results_rgb, time_stats


def process_h5_file_to_h5(
    src_h5_file: str,
    dst_h5_file: str,
    image_key: str,
    moge_model,
    sd_pipeline,
    args,
    device: str,
):
    """
    Process all frames in a single H5 file and save as a new H5 file.

    Args:
        src_h5_file: Source H5 file path
        dst_h5_file: Destination H5 file path
        image_key: Key for image data
        moge_model: MoGe model
        sd_pipeline: SD Inpainting pipeline
        args: Command-line arguments
        device: Device string

    Returns:
        total_time: Total processing time
        time_stats: Timing statistics
    """
    total_start_time = time.time()
    time_stats = {"moge": 0.0, "warp": 0.0, "inpaint": 0.0}

    # Get frame count
    total_frames = get_h5_frame_count(src_h5_file, image_key)

    if args.frame_index is not None:
        frame_indices = [args.frame_index]
    else:
        start = args.start_frame
        end = args.end_frame if args.end_frame is not None else total_frames
        end = min(end, total_frames)
        frame_indices = list(range(start, end))

    # Collect all processed images
    all_results = [None] * total_frames  # Pre-allocate

    # Process in batches
    num_batches = (len(frame_indices) + args.batch_size - 1) // args.batch_size

    with tqdm(total=len(frame_indices), desc=f"Processing {os.path.basename(src_h5_file)}", unit="frame") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(frame_indices))
            batch_frames = frame_indices[batch_start:batch_end]

            # Process batch without saving image files
            batch_time_stats, batch_results = process_batch(
                batch_frames=batch_frames,
                h5_file=src_h5_file,
                image_key=image_key,
                output_dir=args.output_dir,
                moge_model=moge_model,
                sd_pipeline=sd_pipeline,
                args=args,
                device=device,
                save_images=False,
            )

            # Collect results
            for i, frame_idx in enumerate(batch_frames):
                all_results[frame_idx] = batch_results[i]

            # Accumulate timing statistics
            for key in ["moge", "warp", "inpaint"]:
                time_stats[key] += batch_time_stats[key]

            pbar.update(len(batch_frames))

    # For unprocessed frames (if any), read from original file
    for i in range(total_frames):
        if all_results[i] is None:
            all_results[i] = read_h5_image(src_h5_file, image_key, i)

    # Save as new H5 file
    os.makedirs(os.path.dirname(dst_h5_file), exist_ok=True)
    copy_h5_with_replaced_images(src_h5_file, dst_h5_file, image_key, all_results)

    total_time = time.time() - total_start_time
    return total_time, time_stats


def process_single_h5_with_executor(
    executor,
    src_h5_file: str,
    dst_h5_file: str,
    image_key: str,
    args,
    num_gpus: int,
):
    """
    Process a single H5 file using an already-initialized executor.
    """
    total_start_time = time.time()
    time_stats = {"moge": 0.0, "warp": 0.0, "inpaint": 0.0}

    # Get frame count
    total_frames = get_h5_frame_count(src_h5_file, image_key)

    if args.frame_index is not None:
        frame_indices = [args.frame_index]
    else:
        start = args.start_frame
        end = args.end_frame if args.end_frame is not None else total_frames
        end = min(end, total_frames)
        frame_indices = list(range(start, end))

    # Prepare all batch tasks
    num_batches = (len(frame_indices) + args.batch_size - 1) // args.batch_size
    tasks = []

    args_dict = {
        'trajectory': args.trajectory,
        'movement_distance': args.movement_distance,
        'movement_distance_noise': args.movement_distance_noise,
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
    }

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(frame_indices))
        batch_frames = frame_indices[batch_start:batch_end]
        tasks.append((batch_frames, src_h5_file, image_key, args.output_dir, args_dict, False))

    # Collect all processed images
    all_results = [None] * total_frames

    # Submit all tasks
    with tqdm(total=len(frame_indices), desc=f"  {os.path.basename(src_h5_file)}", unit="frame") as pbar:
        futures = [executor.submit(gpu_worker_process_batch, task) for task in tasks]

        for future in as_completed(futures):
            batch_frames, batch_results, batch_time_stats = future.result()

            # Collect results
            for i, frame_idx in enumerate(batch_frames):
                all_results[frame_idx] = batch_results[i]

            for key in ["moge", "warp", "inpaint"]:
                time_stats[key] += batch_time_stats[key]
            pbar.update(len(batch_frames))

    # For unprocessed frames, read from original file
    for i in range(total_frames):
        if all_results[i] is None:
            all_results[i] = read_h5_image(src_h5_file, image_key, i)

    # Save as new H5 file
    os.makedirs(os.path.dirname(dst_h5_file) if os.path.dirname(dst_h5_file) else '.', exist_ok=True)
    copy_h5_with_replaced_images(src_h5_file, dst_h5_file, image_key, all_results)

    total_time = time.time() - total_start_time
    return total_time, time_stats


def process_h5_dir_multi_gpu(h5_files: list, args, num_gpus: int):
    """
    Process multiple H5 files using multiple GPUs (models loaded once).
    """
    global _executor

    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Register signal handling and exit cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_workers)

    total_start_time = time.time()
    total_time_stats = {"moge": 0.0, "warp": 0.0, "inpaint": 0.0}
    total_frames_processed = 0

    try:
        # Initialize executor (once only)
        print(f"\nInitializing {num_gpus} GPU workers...")
        _executor = ProcessPoolExecutor(max_workers=num_gpus)

        init_futures = []
        for gpu_id in range(num_gpus):
            future = _executor.submit(init_gpu_worker, gpu_id, args.moge_model, args.sd_model)
            init_futures.append(future)

        for future in init_futures:
            future.result()
        print(f"GPU workers initialized\n")

        # Process each H5 file (reusing initialized workers)
        for h5_idx, src_h5_file in enumerate(h5_files):
            filename = os.path.basename(src_h5_file)
            dst_h5_file = os.path.join(args.output_dir, filename)

            print(f"[{h5_idx+1}/{len(h5_files)}] Processing: {filename}")

            file_time, file_stats = process_single_h5_with_executor(
                executor=_executor,
                src_h5_file=src_h5_file,
                dst_h5_file=dst_h5_file,
                image_key=args.image_key,
                args=args,
                num_gpus=num_gpus,
            )

            # Accumulate statistics
            for key in ["moge", "warp", "inpaint"]:
                total_time_stats[key] += file_stats[key]
            total_frames_processed += get_h5_frame_count(src_h5_file, args.image_key)

            print(f"  Saved to: {dst_h5_file}")

    except KeyboardInterrupt:
        print("\nInterrupt received, cleaning up...")
        raise
    finally:
        if _executor is not None:
            _executor.shutdown(wait=True, cancel_futures=True)
            _executor = None
        cleanup_workers()

    total_time = time.time() - total_start_time
    return total_time, total_time_stats, total_frames_processed


def run_single_gpu(args, frame_indices):
    """Single GPU mode."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    time_stats = {
        "load_moge": 0.0,
        "load_sd": 0.0,
        "moge": 0.0,
        "warp": 0.0,
        "inpaint": 0.0,
    }
    total_start_time = time.time()

    # Load MoGe model
    print(f"Loading MoGe model: {args.moge_model}...")
    t0 = time.time()
    moge_model = MoGeModel.from_pretrained(args.moge_model).to(device)
    moge_model.eval()
    time_stats["load_moge"] = time.time() - t0
    print(f"MoGe model loaded in {time_stats['load_moge']:>8.2f}s")

    # Load SD Inpainting model
    print(f"Loading SD Inpainting model: {args.sd_model}...")
    sd_pipeline = None
    t0 = time.time()
    try:
        from diffusers import StableDiffusionInpaintPipeline
        sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            args.sd_model,
            torch_dtype=torch.float16,
        ).to(device)
        sd_pipeline.set_progress_bar_config(disable=True)
    except ImportError:
        print("Warning: diffusers not installed, falling back to OpenCV inpainting")
    time_stats["load_sd"] = time.time() - t0
    print(f"SD Inpainting model loaded in {time_stats['load_sd']:>8.2f}s")

    # Process frames in batches
    num_batches = (len(frame_indices) + args.batch_size - 1) // args.batch_size

    with tqdm(total=len(frame_indices), desc="Processing", unit="frame") as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(frame_indices))
            batch_frames = frame_indices[batch_start:batch_end]

            batch_time_stats = process_batch(
                batch_frames=batch_frames,
                h5_file=args.h5_file,
                image_key=args.image_key,
                output_dir=args.output_dir,
                moge_model=moge_model,
                sd_pipeline=sd_pipeline,
                args=args,
                device=device,
            )

            for key in ["moge", "warp", "inpaint"]:
                time_stats[key] += batch_time_stats[key]

            pbar.update(len(batch_frames))

    total_time = time.time() - total_start_time
    return total_time, time_stats


def run_multi_gpu(args, frame_indices):
    """Multi-GPU parallel mode."""
    global _executor

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # Register signal handling and exit cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_workers)

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs for parallel processing")

    total_start_time = time.time()

    # Prepare all batch tasks
    num_batches = (len(frame_indices) + args.batch_size - 1) // args.batch_size
    tasks = []

    # Convert args to serializable dict
    args_dict = {
        'trajectory': args.trajectory,
        'movement_distance': args.movement_distance,
        'movement_distance_noise': args.movement_distance_noise,
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
    }

    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(frame_indices))
        batch_frames = frame_indices[batch_start:batch_end]
        tasks.append((batch_frames, args.h5_file, args.image_key, args.output_dir, args_dict))

    # Timing statistics
    time_stats = {"moge": 0.0, "warp": 0.0, "inpaint": 0.0}

    # Create process pool, each process bound to a GPU
    print(f"Initializing {num_gpus} GPU workers...")

    try:
        _executor = ProcessPoolExecutor(max_workers=num_gpus)

        # Initialize each worker
        init_futures = []
        for gpu_id in range(num_gpus):
            future = _executor.submit(init_gpu_worker, gpu_id, args.moge_model, args.sd_model)
            init_futures.append(future)

        # Wait for initialization
        for future in init_futures:
            future.result()
        print(f"GPU workers initialized")

        # Submit all tasks
        with tqdm(total=len(frame_indices), desc=f"Processing ({num_gpus} GPUs)", unit="frame") as pbar:
            futures = [_executor.submit(gpu_worker_process_batch, task) for task in tasks]

            for future in as_completed(futures):
                num_frames, batch_time_stats = future.result()
                for key in ["moge", "warp", "inpaint"]:
                    time_stats[key] += batch_time_stats[key]
                pbar.update(num_frames)

    except KeyboardInterrupt:
        print("\nInterrupt received, cleaning up...")
        raise
    finally:
        # Ensure cleanup
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=True)
            _executor = None
        cleanup_workers()

    total_time = time.time() - total_start_time
    return total_time, time_stats


def main():
    args = parse_args()

    # Validate input arguments
    if args.h5_file is None and args.h5_dir is None:
        print("Error: must specify either --h5_file or --h5_dir")
        sys.exit(1)
    if args.h5_file is not None and args.h5_dir is not None:
        print("Error: --h5_file and --h5_dir cannot be specified at the same time")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== Directory mode: process multiple H5 files and save as new H5 ==========
    if args.h5_dir is not None:
        h5_files = get_h5_files_in_dir(args.h5_dir)
        if len(h5_files) == 0:
            print(f"Error: no H5 files found in directory {args.h5_dir}")
            sys.exit(1)

        # Determine number of GPUs to use
        num_gpus = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        use_multi_gpu = num_gpus > 1

        print(f"Found {len(h5_files)} H5 files")
        print(f"Input directory: {args.h5_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Image key: {args.image_key}")
        print(f"Batch size: {args.batch_size}")
        print(f"Number of GPUs: {num_gpus}")

        if use_multi_gpu:
            # Multi-GPU mode: process all files with unified function (models loaded once)
            total_time, total_time_stats, total_frames_processed = process_h5_dir_multi_gpu(
                h5_files=h5_files,
                args=args,
                num_gpus=num_gpus,
            )
        else:
            # Single-GPU mode: pre-load models, process files sequentially
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\nUsing device: {device}")

            print(f"Loading MoGe model: {args.moge_model}...")
            moge_model = MoGeModel.from_pretrained(args.moge_model).to(device)
            moge_model.eval()

            print(f"Loading SD Inpainting model: {args.sd_model}...")
            sd_pipeline = None
            try:
                from diffusers import StableDiffusionInpaintPipeline
                sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    args.sd_model,
                    torch_dtype=torch.float16,
                ).to(device)
                sd_pipeline.set_progress_bar_config(disable=True)
            except ImportError:
                print("Warning: diffusers not installed, falling back to OpenCV inpainting")

            # Process each H5 file
            total_start_time = time.time()
            total_time_stats = {"moge": 0.0, "warp": 0.0, "inpaint": 0.0}
            total_frames_processed = 0

            for h5_idx, src_h5_file in enumerate(h5_files):
                filename = os.path.basename(src_h5_file)
                dst_h5_file = os.path.join(args.output_dir, filename)

                print(f"\n[{h5_idx+1}/{len(h5_files)}] Processing: {filename}")

                file_time, file_stats = process_h5_file_to_h5(
                    src_h5_file=src_h5_file,
                    dst_h5_file=dst_h5_file,
                    image_key=args.image_key,
                    moge_model=moge_model,
                    sd_pipeline=sd_pipeline,
                    args=args,
                    device=device,
                )

                for key in ["moge", "warp", "inpaint"]:
                    total_time_stats[key] += file_stats[key]
                total_frames_processed += get_h5_frame_count(src_h5_file, args.image_key)

                print(f"  Saved to: {dst_h5_file}")

            total_time = time.time() - total_start_time

        print(f"\n{'='*60}")
        print(f"Done! Processed {len(h5_files)} files, {total_frames_processed} frames")
        print(f"Output directory: {args.output_dir}")
        print(f"{'='*60}")

        if args.show_time:
            print(f"\nTiming statistics:")
            print(f"  MoGe depth prediction: {total_time_stats['moge']:>8.2f}s")
            print(f"  3D Warping:            {total_time_stats['warp']:>8.2f}s")
            print(f"  SD Inpainting:         {total_time_stats['inpaint']:>8.2f}s")
            print(f"  Total time:            {total_time:>8.2f}s")
            print(f"  Average per frame:     {total_time/total_frames_processed:>8.2f}s")

        return

    # ========== Single file mode ==========
    # Determine frame range
    total_frames = get_h5_frame_count(args.h5_file, args.image_key)
    print(f"Total frames in HDF5 file: {total_frames}")

    if args.frame_index is not None:
        frame_indices = [args.frame_index]
    else:
        start = args.start_frame
        end = args.end_frame if args.end_frame is not None else total_frames
        end = min(end, total_frames)
        frame_indices = list(range(start, end))

    print(f"Processing {len(frame_indices)} frames (batch_size={args.batch_size})")

    # If --save_h5 specified, save as H5 file
    if args.save_h5:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print(f"Loading MoGe model: {args.moge_model}...")
        moge_model = MoGeModel.from_pretrained(args.moge_model).to(device)
        moge_model.eval()

        print(f"Loading SD Inpainting model: {args.sd_model}...")
        sd_pipeline = None
        try:
            from diffusers import StableDiffusionInpaintPipeline
            sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                args.sd_model,
                torch_dtype=torch.float16,
            ).to(device)
            sd_pipeline.set_progress_bar_config(disable=True)
        except ImportError:
            print("Warning: diffusers not installed, falling back to OpenCV inpainting")

        filename = os.path.basename(args.h5_file)
        dst_h5_file = os.path.join(args.output_dir, filename)

        total_time, time_stats = process_h5_file_to_h5(
            src_h5_file=args.h5_file,
            dst_h5_file=dst_h5_file,
            image_key=args.image_key,
            moge_model=moge_model,
            sd_pipeline=sd_pipeline,
            args=args,
            device=device,
        )

        print(f"\nDone! Output: {dst_h5_file}")
    else:
        # Default mode: save as image files
        if args.num_gpus > 1 and torch.cuda.device_count() > 1:
            total_time, time_stats = run_multi_gpu(args, frame_indices)
        else:
            total_time, time_stats = run_single_gpu(args, frame_indices)

        print(f"\nDone! Output: {args.output_dir}")

    # Show timing statistics
    if args.show_time:
        print(f"\n{'='*50}")
        print("Timing statistics:")
        print(f"{'='*50}")
        if "load_moge" in time_stats:
            print(f"  Load MoGe model:       {time_stats['load_moge']:>8.2f}s")
            print(f"  Load SD model:         {time_stats['load_sd']:>8.2f}s")
        print(f"  MoGe depth prediction: {time_stats['moge']:>8.2f}s")
        print(f"  3D Warping:            {time_stats['warp']:>8.2f}s")
        print(f"  SD Inpainting:         {time_stats['inpaint']:>8.2f}s")
        print(f"{'='*50}")
        print(f"  Total time:            {total_time:>8.2f}s")
        print(f"  Average per frame:     {total_time/len(frame_indices):>8.2f}s")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
