#!/usr/bin/env python3
"""
Reorder downsample_episode files script

Function: 
  Copy and rename downsample_episode_{index}.hdf5 files to output directory in chronological order (date_batch)

Input directory structure:
  input_dir/
    ├── 0101_1/          # Naming convention: {date}_{batch_number}
    │   ├── downsample_episode_0.hdf5
    │   └── downsample_episode_1.hdf5
    ├── 0102_1/
    │   └── downsample_episode_5.hdf5
    └── 0102_2/
        └── downsample_episode_10.hdf5

output directoryStructure:
  output_dir/
    ├── episode_0.hdf5   # From 0101_1/downsample_episode_0.hdf5
    ├── episode_1.hdf5   # From 0101_1/downsample_episode_1.hdf5
    ├── episode_2.hdf5   # From 0102_1/downsample_episode_5.hdf5
    └── episode_3.hdf5   # From 0102_2/downsample_episode_10.hdf5

Usage:
  python reorder_episodes.py --input_dir <input_dir> --output_dir <output_dir>
  python reorder_episodes.py --input_dir <input_dir> --output_dir <output_dir> --dry-run
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Tuple


def parse_batch_folder(folder_name: str) -> Tuple[int, int]:
    """
    Parse batch folder name, return (date, batch_number) tuple for sorting.
    
    Format: {date}_{batch_number}, e.g.: 0101_1 -> (101, 1), 0102_2 -> (102, 2)
    """
    match = re.match(r'^(\d+)_(\d+)$', folder_name)
    if match:
        date = int(match.group(1))
        batch = int(match.group(2))
        return (date, batch)
    # if does not match, return a very large value, place at end
    return (999999, 999999)


def extract_episode_index(filename: str) -> int:
    """
    Extract episode index from file name.
    
    format:downsample_episode_{index}.hdf5
    """
    match = re.search(r'downsample_episode_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def collect_episodes(input_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Collect all episode files, sort in chronological order.
    
    Returns: [(filepath, batch folder name, original file name), ...]
    """
    episodes = []
    
    # Get all batch folders
    batch_folders = []
    for item in input_dir.iterdir():
        if item.is_dir():
            batch_folders.append(item)
    
    # Sort by (date, batch_number)
    batch_folders.sort(key=lambda x: parse_batch_folder(x.name))
    
    # Collect episode files in each batch folder
    for batch_folder in batch_folders:
        # Find downsample_episode_*.hdf5 files
        episode_file = list(batch_folder.glob("downsample_episode_*.hdf5"))
        
        # Sort by episode index
        episode_file.sort(key=lambda x: extract_episode_index(x.name))
        
        # Add to result list
        for ep_file in episode_file:
            episodes.append((ep_file, batch_folder.name, ep_file.name))
    
    return episodes


def reorder_episodes(input_dir: Path, output_dir: Path, dry_run: bool = False):
    """
    Reorder episode files in chronological order.
    """
    print("=" * 70)
    print("📁 Reorder episode files")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"output directory: {output_dir}")
    print(f"Mode: {'dry-run (no actual operations)' if dry_run else 'execute'}")
    print()
    
    # checkInput directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directorydoes not exist: {input_dir}")
    
    # Collect all episode files
    episodes = collect_episodes(input_dir)
    
    if len(episodes) == 0:
        print("⚠️  No downsample_episode_*.hdf5 file")
        return
    
    print(f"found {len(episodes)}  episode file")
    print()
    
    # Createoutput directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reorder and copy
    print("Start processing...")
    print("-" * 70)
    
    for new_index, (src_path, batch_name, orig_filename) in enumerate(episodes):
        # New file name: episode_{new_index}.hdf5
        new_filename = f"episode_{new_index}.hdf5"
        dst_path = output_dir / new_filename
        
        # Print information
        print(f"[{new_index:4d}] {new_filename} <- {batch_name}/{orig_filename}")
        
        # Execute copy
        if not dry_run:
            shutil.copy2(src_path, dst_path)
    
    print("-" * 70)
    print()
    
    # Statistics information
    print("📊 Statistics:")
    
    # Statistics by batch
    batch_stats = {}
    for src_path, batch_name, orig_filename in episodes:
        if batch_name not in batch_stats:
            batch_stats[batch_name] = 0
        batch_stats[batch_name] += 1
    
    # Sort output by date
    sorted_batches = sorted(batch_stats.keys(), key=lambda x: parse_batch_folder(x))
    for batch_name in sorted_batches:
        count = batch_stats[batch_name]
        print(f"  {batch_name}: {count}  episode")
    
    print()
    print(f"Total: {len(episodes)}  episode")
    
    if dry_run:
        print()
        print("⚠️  dry-run mode, no actual copy operations performed")
    else:
        print()
        print(f"✅ Completed! Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Reorder downsample_episode files in chronological order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Execute reordering
  python reorder_episodes.py --input_dir /path/to/input --output_dir /path/to/output
  
  # Preview mode (no actual operations)
  python reorder_episodes.py --input_dir /path/to/input --output_dir /path/to/output --dry-run
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory path (contains {date}_{batch} subdirectories)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='output directory path'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview mode, only print operations, don't perform actual copying'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    reorder_episodes(input_dir, output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

