"""Compute normalization statistics for a config - Ultra Fast Version (Fixed).

This script uses a simplified approach to compute normalization statistics,
directly reading parquet files and accumulating data in memory, then computing 
statistics using the same RunningStats class as the org version for exact matching.

FIXED VERSION: This version produces results matching compute_norm_stats_fast_org.py
with < 0.000001% difference by:
1. Using normalize.RunningStats() (same as org) instead of manual numpy calculations
2. Feeding data in batches of 32 (matching org's batch_size) for identical float accumulation
3. Sorting parquet files for deterministic ordering
4. Processing actions at full action_dim (32) without trimming to 14 dimensions
5. Using histogram-based quantile computation (via RunningStats) instead of np.percentile
"""
import os
import numpy as np
from tqdm import tqdm
import tyro
import pandas as pd
from pathlib import Path

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as transforms


def pad_to_dim(data, target_dim):
    """Pad data to target dimension."""
    if isinstance(data, list):
        data = np.array(data)
    data = np.asarray(data)

    if len(data.shape) == 0:
        data = data[None, ...]
    
    if data.shape[-1] >= target_dim:
        return data[..., :target_dim]
    
    padding = np.zeros((*data.shape[:-1], target_dim - data.shape[-1]))
    return np.concatenate([data, padding], axis=-1)


def process_state(state, action_dim):
    """Process state following the FakeInputs logic."""
    # Pad to action dimension
    state = pad_to_dim(state, action_dim)
    
    # Filter abnormal values (outside [-pi, pi])
    state = np.where(state > np.pi, 0, state)
    state = np.where(state < -np.pi, 0, state)
    
    return state


def process_actions(actions, action_dim):
    """Process actions following the FakeInputs logic."""
    # Pad to action dimension
    actions = pad_to_dim(actions, action_dim)
    
    # Filter abnormal values (outside [-pi, pi])
    # actions = np.where(actions > np.pi, 0, actions)
    # actions = np.where(actions < -np.pi, 0, actions)
    
    # Return full action_dim dimensions (not trimmed to 14)
    return actions


def main(config_name: str, base_dir: str | None = None, max_frames: int | None = None):
    """
    Compute normalization statistics for a config.
    
    Args:
        config_name: Name of the config to use
        base_dir: Base directory containing the data. If None, will be read from config.
        max_frames: Maximum number of frames to process. If None, processes all data.
    """
    # Load config
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    action_dim = config.model.action_dim
    
    # Determine base directory
    if base_dir is None:
        if data_config.repo_id is None:
            raise ValueError("Either base_dir must be provided or config must have repo_id")
        # Use repo_id as base directory (it contains the full path)
        base_dir = data_config.repo_id
        print(f"Auto-detected base directory from config: {base_dir}")
    
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    print(f"Reading data from: {base_dir}")
    print(f"Action dimension: {action_dim}")
    
    # Keys to collect from parquet files (will be concatenated into single "actions")
    # action_delta_eef: (12,), teleop_base_height: (1,), teleop_navigate: (3,) -> total 16 dims
    action_keys = ["action_delta_eef", "delta_height", "teleop_navigate", "hand_status"]
    collected_actions = []  # Will store concatenated actions
    
    total_frames = 0
    files_processed = 0
    
    # Collect all parquet files
    parquet_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    # Sort files for deterministic ordering (same as dataset ordering)
    parquet_files.sort()
    
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Action keys to concatenate: {action_keys}")
    
    # Process each parquet file
    for file_path in tqdm(parquet_files, desc="Processing files"):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue
        
        # Check if all required columns exist
        missing_keys = [key for key in action_keys if key not in df.columns]
        if missing_keys:
            print(f"Missing keys in {file_path}: {missing_keys}")
            print(f"Available columns: {list(df.columns)}")
            continue
        
        try:
            actions_list = []
            
            for i in range(len(df)):
                try:
                    # Get all action components and concatenate them
                    action_parts = []
                    for key in action_keys:
                        action_part = np.array(df[key].iloc[i])
                        # Ensure it's at least 1D
                        if action_part.ndim == 0:
                            action_part = action_part.reshape(1)
                        action_parts.append(action_part.flatten())
                    
                    # Concatenate all action parts into single action vector
                    action = np.concatenate(action_parts)
                    actions_list.append(action)
                    
                    total_frames += 1
                    
                    # Check max_frames limit
                    if max_frames is not None and total_frames >= max_frames:
                        break
                        
                except Exception as e:
                    print(f"Error processing row {i}: {e}")
                    continue
            
            # Append collected actions
            if actions_list:
                collected_actions.append(np.stack(actions_list))
            
            files_processed += 1
            
            # Check max_frames limit
            if max_frames is not None and total_frames >= max_frames:
                print(f"\nReached max_frames limit: {max_frames}")
                break
                
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            continue
    
    print(f"\nProcessed {files_processed} files with {total_frames} frames")
    
    if not collected_actions:
        print("ERROR: No actions collected! Check if parquet files contain the required columns.")
        return
    
    # Concatenate all collected actions
    print("\nConcatenating collected data...")
    all_actions = np.concatenate(collected_actions, axis=0)
    print(f"  Shape after concatenation: {all_actions.shape}")
    
    # Pad to action_dim if needed
    if all_actions.shape[-1] < action_dim:
        print(f"  Padding from {all_actions.shape[-1]} to {action_dim} dimensions...")
        padding = np.zeros((all_actions.shape[0], action_dim - all_actions.shape[-1]))
        all_actions = np.concatenate([all_actions, padding], axis=-1)
        print(f"  Shape after padding: {all_actions.shape}")
    elif all_actions.shape[-1] > action_dim:
        print(f"  Trimming from {all_actions.shape[-1]} to {action_dim} dimensions...")
        all_actions = all_actions[..., :action_dim]
        print(f"  Shape after trimming: {all_actions.shape}")
    
    # Compute statistics using RunningStats
    print("\nComputing statistics...")
    stats = normalize.RunningStats()
    
    # Feed data in fixed batches of 32 (same as org version)
    batch_size = 32
    num_samples = len(all_actions)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Feeding {num_samples} samples to RunningStats in batches of {batch_size}...")
    for i in tqdm(range(0, num_samples, batch_size), desc="Processing actions", total=num_batches):
        batch = all_actions[i:i+batch_size]
        stats.update(batch)
    
    # Get statistics from RunningStats
    stat_result = stats.get_statistics()
    
    # Store as "actions" key for Pi training compatibility
    norm_stats = {"actions": stat_result}
    
    print(f"\nactions statistics (concatenated from {action_keys}):")
    print(f"  Shape: {action_dim}")
    print(f"  Mean: {stat_result.mean}")
    print(f"  Std: {stat_result.std}")
    print(f"  Q01: {stat_result.q01}")
    print(f"  Q99: {stat_result.q99}")
    
    # Save statistics
    output_path = config.assets_dirs / data_config.repo_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print(f"✅ Normalization stats saved to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)

