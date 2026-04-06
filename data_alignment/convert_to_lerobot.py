"""
Convert G1 HDF5 dataset to LeRobot format.

Dataset structure from HDF5:
- action: (N, 43) float64 - full action state
- action_delta_eef: (N, 12) float64 - delta end effector action
- action_eef: (N, 14) float64 - end effector action
- observations/images/left: (N, 540, 960, 3) uint8 - left camera image
- teleop_navigate_command: (N, 3) float64 - navigation command
- delta_height: (N,) float32 - delta height
- hand_status: (N, 2) int32 - hand status


## usage
# for debug mode, only process the first episode
python scripts/ego2lerobot/g1_to_lerobot.py --debug

# for full conversion with custom parameters
python scripts/ego2lerobot/g1_to_lerobot.py \
    --src-path /cpfs01/shared/shimodi/data/g1/source/1224_combined \
    --output-path /cpfs01/shared/shimodi/data/g1/lerobot \
    --repo-id 1224_combined \
    --fps 20 \
    --task "grasp the toy and put it on the table"

# for parallel conversion (faster)
python scripts/ego2lerobot/g1_to_lerobot.py \
    --src-path /cpfs01/shared/shimodi/data/g1/source/toy/0103_copy_hand_status \
    --output-path /cpfs01/shared/shimodi/data/g1/lerobot/toy \
    --repo-id 0103_copy_hand_status \
    --num-workers 16 \
    --fps 20 \
    --task "grasp the toy and put it on the table"

python scripts/ego2lerobot/g1_to_lerobot.py \
    --src-path  /cpfs01/shared/shimodi/data/human/source/toy/version_5/final_split/batch_000 \
    --output-path /cpfs01/shared/shimodi/data/human/lerobot/toy \
    --repo-id v5_cleaned \
    --num-workers 16 \
    --fps 20 \
    --task "grasp the toy and put it on the table"

python scripts/ego2lerobot/g1_to_lerobot.py \
    --src-path /cpfs01/shared/shimodi/data/human/source/toy/version_5/final_split/batch_001 \
    --output-path /cpfs01/shared/shimodi/data/human/lerobot/toy \
    --repo-id v5_cleaned_1 \
    --num-workers 16 \
    --fps 20 \
    --task "grasp the toy and put it on the table"

python scripts/ego2lerobot/g1_to_lerobot.py \
    --src-path  /cpfs01/shared/shimodi/data/human/source/bed/version_3/final_split/batch_000 \
    --output-path /cpfs01/shared/shimodi/data/human/lerobot/bed \
    --repo-id v3_cleaned \
    --num-workers 16 \
    --fps 20 \
    --task "put the pillow on the bed"

python scripts/ego2lerobot/g1_to_lerobot.py \
    --src-path /cpfs01/shared/shimodi/data/human/source/bed/version_3/final_split/batch_001 \
    --output-path /cpfs01/shared/shimodi/data/human/lerobot/bed \
    --repo-id v3_cleaned_1 \
    --num-workers 16 \
    --fps 20 \
    --task "put the pillow on the bed"

python scripts/ego2lerobot/g1_to_lerobot.py \
    --src-path /cpfs01/shared/shimodi/data/human/source/bed/version_1/final_split/batch_002 \
    --output-path /cpfs01/shared/shimodi/data/human/lerobot/bed \
    --repo-id 0113_2 \
    --num-workers 16 \
    --fps 20 \
    --task "put the pillow on the bed"

python scripts/ego2lerobot/g1_to_lerobot.py \
    --src-path /cpfs01/shared/shimodi/data/g1/source/cart/version_2/merged \
    --output-path /cpfs01/shared/shimodi/data/g1/lerobot/cart \
    --repo-id version_2 \
    --num-workers 16 \
    --fps 20 \
    --task "push the cart and put a toy in it"

python scripts/ego2lerobot/g1_to_lerobot.py \
    --src-path /cpfs01/shared/shimodi/data/g1/source/garbage/version_1/merged \
    --output-path /cpfs01/shared/shimodi/data/g1/lerobot/garbage \
    --repo-id version_1 \
    --num-workers 16 \
    --fps 20 \
    --task "put the garbage into the bin"

"""

import argparse
import gc
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


# Configuration for G1 dataset
G1_CONFIG = {
    "images": {
        "left": {
            "dtype": "image",
            "shape": (600, 960, 3),
            "names": ["height", "width", "channel"],
        },
    },
    "states": {
    },
    "actions": {
        "action_eef": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["action_eef"],
        },
        "action_delta_eef": {
            "dtype": "float32",
            "shape": (12,),
            "names": ["action_delta_eef"],
        },
        "teleop_navigate": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["teleop_navigate_command"],
        },
        "delta_height": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["delta_height"],
        },
        "hand_status": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["hand_status"],
        },
    },
}


def generate_features_from_config(config: dict) -> dict:
    """Generate LeRobot features from config."""
    features = {}

    # Add image features
    for key, value in config["images"].items():
        features[f"observation.images.{key}"] = {
            "dtype": value["dtype"],
            "shape": value["shape"],
            "names": value["names"],
        }

    # Add state features
    for key, value in config["states"].items():
        features[f"observation.{key}"] = {
            "dtype": value["dtype"],
            "shape": value["shape"],
            "names": value["names"],
        }

    # Add action features
    for key, value in config["actions"].items():
        features[key] = {
            "dtype": value["dtype"],
            "shape": value["shape"],
            "names": value["names"],
        }

    return features


def _decode_image(img_data, cv2_module):
    """Decode image from various encoded formats to RGB numpy array."""
    # Already decoded as 3D array
    if isinstance(img_data, np.ndarray) and img_data.ndim == 3:
        return img_data
    
    # Encoded as bytes
    if isinstance(img_data, bytes):
        img = cv2_module.imdecode(np.frombuffer(img_data, np.uint8), cv2_module.IMREAD_COLOR)
        return cv2_module.cvtColor(img, cv2_module.COLOR_BGR2RGB)
    
    # Encoded as 1D numpy array (common in HDF5 object dtype)
    if isinstance(img_data, np.ndarray) and img_data.ndim == 1:
        img = cv2_module.imdecode(img_data.astype(np.uint8), cv2_module.IMREAD_COLOR)
        return cv2_module.cvtColor(img, cv2_module.COLOR_BGR2RGB)
    
    # Numpy void type (from HDF5 variable-length data)
    if isinstance(img_data, np.void):
        img_bytes = bytes(img_data)
        img = cv2_module.imdecode(np.frombuffer(img_bytes, np.uint8), cv2_module.IMREAD_COLOR)
        return cv2_module.cvtColor(img, cv2_module.COLOR_BGR2RGB)
    
    raise ValueError(f"Unknown image format: type={type(img_data)}, shape={getattr(img_data, 'shape', 'N/A')}")


def load_episode_from_hdf5_streaming(hdf5_path: Path, task_instruction: str):
    """
    Load episode data from HDF5 file using streaming/chunked approach.
    Yields frames one by one to reduce memory usage.

    Args:
        hdf5_path: Path to HDF5 file
        task_instruction: Task instruction string to add to each frame

    Yields:
        frame: dict containing frame data
    """
    with h5py.File(hdf5_path, "r") as f:
        num_frames = f["action_eef"].shape[0]

        # Get dataset references (don't load into memory yet)
        teleop_navigate = f["teleop_navigate_command"]
        delta_height_data = f["delta_height"]
        hand_status_data = f["hand_status"]
        action_eef = f["action_eef"]
        action_delta_eef = f["action_delta_eef"]
        images_left = f["observation_image_left"]

        # Stream frames one by one
        import cv2
        for i in range(num_frames):
            # Decode images if stored as object (encoded bytes)
            img_left = images_left[i]
            
            # Handle various encoded image formats
            img_left = _decode_image(img_left, cv2)
            
            # delta_height is (N,), reshape to (1,)
            delta_height = np.array(delta_height_data[i], dtype=np.float32).reshape(1,)
            # # hand_status: int32 in HDF5 -> float32 in LeRobot
            # hand_status = np.array(hand_status_data[i], dtype=np.float32).reshape(1,)
            
            frame = {
                # Images - read directly from HDF5
                "observation.images.left": np.array(img_left, dtype=np.uint8),
                # Actions
                "action_eef": np.array(action_eef[i], dtype=np.float32),
                "action_delta_eef": np.array(action_delta_eef[i], dtype=np.float32),
                "teleop_navigate": np.array(teleop_navigate[i], dtype=np.float32),
                "delta_height": delta_height,
                "hand_status": np.array(hand_status_data[i], dtype=np.float32),
                # "hand_status": hand_status,
                # Task
                "task": task_instruction,
            }
            yield frame

        return num_frames


def load_episode_from_hdf5_batch(hdf5_path: Path, task_instruction: str, batch_size: int = 100):
    """
    Load episode data from HDF5 file in batches for better I/O performance.

    Args:
        hdf5_path: Path to HDF5 file
        task_instruction: Task instruction string
        batch_size: Number of frames to load at once

    Yields:
        frames: list of frame dicts
    """
    with h5py.File(hdf5_path, "r") as f:
        num_frames = f["action_eef"].shape[0]

        for start_idx in range(0, num_frames, batch_size):
            end_idx = min(start_idx + batch_size, num_frames)

            # Load batch of data
            teleop_navigate = np.array(f["teleop_navigate_command"][start_idx:end_idx], dtype=np.float32)
            delta_height_data = np.array(f["delta_height"][start_idx:end_idx], dtype=np.float32)
            hand_status_data = np.array(f["hand_status"][start_idx:end_idx], dtype=np.float32)  # int32 -> float32
            action_eef = np.array(f["action_eef"][start_idx:end_idx], dtype=np.float32)
            action_delta_eef = np.array(f["action_delta_eef"][start_idx:end_idx], dtype=np.float32)
            
            # Load and decode images
            images_left_raw = f["observation_image_left"][start_idx:end_idx]
            
            import cv2
            images_left = [_decode_image(img, cv2) for img in images_left_raw]

            # Build frames from batch
            frames = []
            for i in range(end_idx - start_idx):
                # delta_height is (N,), reshape to (1,)
                delta_height = delta_height_data[i].reshape(1,)
                # # hand_status is (N, 1), already float32
                # hand_status = hand_status_data[i].reshape(1,)
                
                frame = {
                    "observation.images.left": np.array(images_left[i], dtype=np.uint8),
                    "action_eef": action_eef[i],
                    "action_delta_eef": action_delta_eef[i],
                    "teleop_navigate": teleop_navigate[i],
                    "delta_height": delta_height,
                    "hand_status": hand_status_data[i],
                    "task": task_instruction,
                }
                frames.append(frame)

            yield frames

            # Free memory
            del teleop_navigate, delta_height_data, hand_status_data
            del action_eef, action_delta_eef
            del images_left, images_left_raw


def get_episode_info(hdf5_path: Path) -> tuple[int, int]:
    """Get episode ID and number of frames without loading data."""
    episode_id = int(hdf5_path.stem.split("_")[-1])
    with h5py.File(hdf5_path, "r") as f:
        num_frames = f["action_eef"].shape[0]
    return episode_id, num_frames


def get_episode_files(src_path: Path) -> list[Path]:
    """Get all episode HDF5 files from source directory, sorted by episode number."""
    episode_files = list(src_path.glob("episode_*.hdf5"))
    # Sort numerically by episode number (e.g., 0, 1, 2, ..., 10, 11, 12)
    episode_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
    return episode_files


def process_single_episode(
    episode_file: Path,
    task_instruction: str,
    batch_size: int = 100,
) -> tuple[int, int, str | None]:
    """
    Process a single episode - used for parallel processing info gathering.

    Returns:
        episode_id, num_frames, error_message
    """
    try:
        episode_id, num_frames = get_episode_info(episode_file)
        return episode_id, num_frames, None
    except Exception as e:
        episode_id = int(episode_file.stem.split("_")[-1])
        return episode_id, 0, str(e)


def convert_to_lerobot(
    src_path: str,
    output_path: str,
    repo_id: str = "g1_dataset",
    fps: int = 30,
    task_instruction: str = "G1 robot manipulation task",
    image_writer_threads: int = 8,
    image_writer_processes: int = 4,
    batch_size: int = 100,
    num_workers: int = 1,
    debug: bool = False,
):
    """
    Convert G1 HDF5 dataset to LeRobot format.

    Args:
        src_path: Path to source directory containing episode_*.hdf5 files
        output_path: Path to output directory for LeRobot dataset
        repo_id: Repository ID for the dataset
        fps: Frames per second
        task_instruction: Task instruction string
        image_writer_threads: Number of threads for image writing
        image_writer_processes: Number of processes for image writing
        batch_size: Number of frames to load at once from HDF5
        num_workers: Number of parallel workers for episode info gathering
        debug: If True, only process first episode
    """
    src_path = Path(src_path)
    output_path = Path(output_path)

    # Get all episode files
    episode_files = get_episode_files(src_path)
    if not episode_files:
        raise ValueError(f"No episode files found in {src_path}")

    print(f"Found {len(episode_files)} episodes in {src_path}")

    if debug:
        episode_files = episode_files[:1]
        print("Debug mode: only processing first episode")

    # Generate features
    features = generate_features_from_config(G1_CONFIG)
    print("Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")

    # Create output directory
    local_dir = output_path / repo_id
    if local_dir.exists():
        print(f"Removing existing directory: {local_dir}")
        shutil.rmtree(local_dir)

    # Create LeRobot dataset with optimized settings
    print(f"\nCreating LeRobot dataset with {image_writer_threads} threads, {image_writer_processes} processes...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=local_dir,
        fps=fps,
        robot_type="g1",
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    # Gather episode info in parallel if num_workers > 1
    if num_workers > 1 and len(episode_files) > 1:
        print(f"\nGathering episode info with {num_workers} workers...")
        episode_info = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(get_episode_info, f): f for f in episode_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning episodes"):
                try:
                    ep_id, num_frames = future.result()
                    episode_info.append((futures[future], ep_id, num_frames))
                except Exception as e:
                    f = futures[future]
                    print(f"Error scanning {f}: {e}")
        # Sort by episode ID
        episode_info.sort(key=lambda x: x[1])
        total_frames_estimate = sum(x[2] for x in episode_info)
        print(f"Total frames to process: ~{total_frames_estimate}")
    else:
        episode_info = [(f, int(f.stem.split("_")[-1]), None) for f in episode_files]

    # Process each episode using batch loading
    total_frames = 0
    successful_episodes = 0

    print(f"\nProcessing episodes (batch_size={batch_size})...")
    for episode_file, episode_id, _ in tqdm(episode_info, desc="Converting episodes"):
        try:
            episode_frames = 0

            # Use batch loading for better I/O performance
            for frames_batch in load_episode_from_hdf5_batch(episode_file, task_instruction, batch_size):
                for frame in frames_batch:
                    dataset.add_frame(frame)
                    episode_frames += 1

            # Save episode
            dataset.save_episode()
            total_frames += episode_frames
            successful_episodes += 1

            if debug or (successful_episodes % 10 == 0):
                print(f"  Episode {episode_id}: {episode_frames} frames (total: {total_frames})")

        except Exception as e:
            print(f"  Error processing episode {episode_id}: {e}")
            dataset.episode_buffer = None
            continue

        # Periodic garbage collection
        if successful_episodes % 5 == 0:
            gc.collect()

    print(f"\n{'='*50}")
    print(f"Dataset saved to {local_dir}")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Total frames: {dataset.num_frames}")
    print(f"Successful: {successful_episodes}/{len(episode_info)}")


def main():
    parser = argparse.ArgumentParser(description="Convert G1 HDF5 dataset to LeRobot format")
    parser.add_argument(
        "--src-path",
        type=str,
        default="/cpfs01/shared/shimodi/data/g1/test",
        help="Path to source directory containing episode_*.hdf5 files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/cpfs01/shared/shimodi/data/g1/lerobot",
        help="Path to output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="g1_dataset",
        help="Repository ID for the dataset",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="G1 robot manipulation task",
        help="Task instruction string",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=8,
        help="Number of threads for image writing (default: 8)",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=4,
        help="Number of processes for image writing (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of frames to load at once from HDF5 (default: 100)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for episode scanning (default: 4)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only process first episode",
    )

    args = parser.parse_args()

    convert_to_lerobot(
        src_path=args.src_path,
        output_path=args.output_path,
        repo_id=args.repo_id,
        fps=args.fps,
        task_instruction=args.task,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
