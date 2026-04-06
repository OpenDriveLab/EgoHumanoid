#!/usr/bin/env python3
"""
Reorder episode files script

Function: 
  Copy and rename episode_{index}.hdf5 or episode_{index}.svo2 files to output directory in chronological order (date_batch_number)

Input directory structure:
  input_dir/
    ├── 0101_1/          # Naming convention: {date}_{batch_number}
    │   ├── episode_0.hdf5
    │   └── episode_1.hdf5
    ├── 0102_1/
    │   └── episode_5.hdf5
    └── 0102_2/
        └── episode_10.hdf5

output directoryStructure:
  output_dir/
    ├── hdf5/
    │   ├── episode_0.hdf5   # From 0101_1/episode_0.hdf5
    │   ├── episode_1.hdf5   # From 0101_1/episode_1.hdf5
    │   ├── episode_2.hdf5   # From 0102_1/episode_5.hdf5
    │   └── episode_3.hdf5   # From 0102_2/episode_10.hdf5
    └── svo2/
        └── ...

Usage:
  python reorder_episodes.py --input_dir <input_dir> --output_dir <output_dir> --file hdf5
  python reorder_episodes.py --input_dir <input_dir> --output_dir <output_dir> --file svo2
  python reorder_episodes.py --input_dir <input_dir> --output_dir <output_dir> --file all
  python reorder_episodes.py --input_dir <input_dir> --output_dir <output_dir> --file hdf5 --dry-run
"""

import argparse
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple


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
    
    format:episode_{index}.hdf5
    """
    match = re.search(r'episode_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def collect_episodes(input_dir: Path, file_type: str) -> List[Tuple[Path, str, str]]:
    """
    Collect all episode files, sort in chronological order.
    
    Args:
        input_dir: Input directory
        file_type: File type ('hdf5' or 'svo2')
    
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
        # Find files with corresponding suffix based on file type
        episode_file = list(batch_folder.glob(f"episode_*.{file_type}"))
        
        # Sort by episode index
        episode_file.sort(key=lambda x: extract_episode_index(x.name))
        
        # Add to result list
        for ep_file in episode_file:
            episodes.append((ep_file, batch_folder.name, ep_file.name))
    
    return episodes


def count_episodes_per_batch(input_dir: Path, file_type: str) -> Dict[str, int]:
    """
    Count number of specified type files in each batch folder.
    
    Args:
        input_dir: Input directory
        file_type: File type ('hdf5' or 'svo2')
    
    Returns:
        {batch folder name: file count, ...}
    """
    batch_counts = {}
    
    for item in input_dir.iterdir():
        if item.is_dir():
            episode_file = list(item.glob(f"episode_*.{file_type}"))
            batch_counts[item.name] = len(episode_file)
    
    return batch_counts


def check_file_count_consistency(input_dir: Path) -> bool:
    """
    check if hdf5 and svo2 file counts are consistent for each batch.
    
    Args:
        input_dir: Input directory
    
    Returns:
        True if consistent, otherwise print error message and return False
    """
    print("=" * 70)
    print("🔍 check hdf5 and svo2 file count consistency")
    print("=" * 70)
    print()
    
    hdf5_counts = count_episodes_per_batch(input_dir, 'hdf5')
    svo2_counts = count_episodes_per_batch(input_dir, 'svo2')
    
    # Get all batch folders
    all_batches = set(hdf5_counts.keys()) | set(svo2_counts.keys())
    sorted_batches = sorted(all_batches, key=lambda x: parse_batch_folder(x))
    
    has_error = False
    error_batches = []
    
    for batch_name in sorted_batches:
        hdf5_count = hdf5_counts.get(batch_name, 0)
        svo2_count = svo2_counts.get(batch_name, 0)
        
        if hdf5_count != svo2_count:
            has_error = True
            error_batches.append((batch_name, hdf5_count, svo2_count))
    
    if has_error:
        print("❌ Found batches with inconsistent file counts:")
        print("-" * 70)
        print(f"{'batch':<15} {'hdf5 count':>12} {'svo2 count':>12} {'Difference':>10}")
        print("-" * 70)
        for batch_name, hdf5_count, svo2_count in error_batches:
            diff = hdf5_count - svo2_count
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            print(f"{batch_name:<15} {hdf5_count:>12} {svo2_count:>12} {diff_str:>10}")
        print("-" * 70)
        print()
        print(f"Total {len(error_batches)} batches hsaved inconsistent file counts, please check data!")
        return False
    else:
        total_hdf5 = sum(hdf5_counts.values())
        total_svo2 = sum(svo2_counts.values())
        print(f"✅ All {len(sorted_batches)} batches have consistent file counts")
        print(f"   Total hdf5: {total_hdf5}")
        print(f"   Total svo2: {total_svo2}")
        print()
        return True


def copy_file(args: Tuple[Path, Path, int, str, str, str]) -> Tuple[int, str]:
    """
    Worker function for copying single file.
    
    Args:
        args: (src_path, dst_path, new_index, new_filename, batch_name, orig_filename)
    
    Returns:
        (new_index, Status message)
    """
    src_path, dst_path, new_index, new_filename, batch_name, orig_filename = args
    shutil.copy2(src_path, dst_path)
    return (new_index, f"[{new_index:4d}] {new_filename} <- {batch_name}/{orig_filename}")


def reorder_episodes_for_type(input_dir: Path, output_dir: Path, file_type: str, dry_run: bool = False, num_workers: int = 8):
    """
    Reorder episode files of specified type in chronological order.
    
    Args:
        input_dir: Input directory
        output_dir: output root directory
        file_type: File type ('hdf5' or 'svo2')
        dry_run: whether preview mode
        num_workers: number of threads for parallel copying
    """
    # actual output directory is output_dir/{file_type}/
    actual_output_dir = output_dir / file_type
    
    print("=" * 70)
    print(f"📁 Reorder episode files ({file_type})")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"output directory: {actual_output_dir}")
    print(f"File type: {file_type}")
    print(f"Number of parallel threads: {num_workers}")
    print(f"Mode: {'dry-run (no actual operations)' if dry_run else 'execute'}")
    print()
    
    # checkInput directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directorydoes not exist: {input_dir}")
    
    # Collect all episode files
    episodes = collect_episodes(input_dir, file_type)
    
    if len(episodes) == 0:
        print(f"⚠️  No episode_*.{file_type} file")
        return
    
    print(f"found {len(episodes)}  episode file")
    print()
    
    # Create output directory (if exists then delete and recreate, to overwrite)
    if not dry_run:
        if actual_output_dir.exists():
            print(f"⚠️  output directory exists, will be overwritten: {actual_output_dir}")
            shutil.rmtree(actual_output_dir)
        actual_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare copy tasks
    copy_tasks = []
    for new_index, (src_path, batch_name, orig_filename) in enumerate(episodes):
        new_filename = f"episode_{new_index}.{file_type}"
        dst_path = actual_output_dir / new_filename
        copy_tasks.append((src_path, dst_path, new_index, new_filename, batch_name, orig_filename))
    
    # Reorder and copy
    print("Start processing...")
    print("-" * 70)
    
    if dry_run:
        # dry-run mode: only print information
        for src_path, dst_path, new_index, new_filename, batch_name, orig_filename in copy_tasks:
            print(f"[{new_index:4d}] {new_filename} <- {batch_name}/{orig_filename}")
    else:
        # use multithreading for parallel copying
        total = len(copy_tasks)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(copy_file, task): task for task in copy_tasks}
            
            for future in as_completed(futures):
                completed += 1
                new_index, msg = future.result()
                # print progress
                print(f"\rProgress: {completed}/{total} ({completed*100//total}%)", end="", flush=True)
        
        print()  # newline
    
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
        print(f"✅ Completed! Output directory: {actual_output_dir}")


def reorder_episodes_all(input_dir: Path, output_dir: Path, dry_run: bool = False, num_workers: int = 8):
    """
    Reorder both hdf5 and svo2 files simultaneously.
    
    Args:
        input_dir: Input directory
        output_dir: output directory
        dry_run: whether preview mode
        num_workers: number of threads for parallel copying
    """
    # First check if hdf5 and svo2 file counts are consistent
    if not check_file_count_consistency(input_dir):
        print()
        print("⛔ Program exits due to inconsistent file counts. Please fix data first before running.")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("📁 Reorder episode files simultaneously (hdf5 + svo2)")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"output directory: {output_dir}")
    print(f"Number of parallel threads: {num_workers}")
    print(f"Mode: {'dry-run (no actual operations)' if dry_run else 'execute'}")
    print()
    
    # checkInput directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directorydoes not exist: {input_dir}")
    
    # Collect all episode files
    all_copy_tasks = []
    batch_stats = {'hdf5': {}, 'svo2': {}}
    
    for file_type in ['hdf5', 'svo2']:
        actual_output_dir = output_dir / file_type
        episodes = collect_episodes(input_dir, file_type)
        
        if len(episodes) == 0:
            print(f"⚠️  No episode_*.{file_type} file")
            continue
        
        print(f"found {len(episodes)}  {file_type} file")
        
        # Create output directory (if exists then delete and recreate, to overwrite)
        if not dry_run:
            if actual_output_dir.exists():
                print(f"⚠️  output directory exists, will be overwritten: {actual_output_dir}")
                shutil.rmtree(actual_output_dir)
            actual_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare copy tasks
        for new_index, (src_path, batch_name, orig_filename) in enumerate(episodes):
            new_filename = f"episode_{new_index}.{file_type}"
            dst_path = actual_output_dir / new_filename
            all_copy_tasks.append((src_path, dst_path, new_index, new_filename, batch_name, orig_filename))
            
            # Statistics
            if batch_name not in batch_stats[file_type]:
                batch_stats[file_type][batch_name] = 0
            batch_stats[file_type][batch_name] += 1
    
    print()
    
    if len(all_copy_tasks) == 0:
        print("⚠️  No files to process")
        return
    
    # Reorder and copy
    print("Start processing...")
    print("-" * 70)
    
    if dry_run:
        # dry-run mode: only print information
        for src_path, dst_path, new_index, new_filename, batch_name, orig_filename in all_copy_tasks:
            print(f"[{new_index:4d}] {new_filename} <- {batch_name}/{orig_filename}")
    else:
        # use multithreading for parallel copying
        total = len(all_copy_tasks)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(copy_file, task): task for task in all_copy_tasks}
            
            for future in as_completed(futures):
                completed += 1
                new_index, msg = future.result()
                # print progress
                print(f"\rProgress: {completed}/{total} ({completed*100//total}%)", end="", flush=True)
        
        print()  # newline
    
    print("-" * 70)
    print()
    
    # Statistics information
    print("📊 Statistics:")
    
    for file_type in ['hdf5', 'svo2']:
        if batch_stats[file_type]:
            print(f"\n  [{file_type}]")
            sorted_batches = sorted(batch_stats[file_type].keys(), key=lambda x: parse_batch_folder(x))
            for batch_name in sorted_batches:
                count = batch_stats[file_type][batch_name]
                print(f"    {batch_name}: {count}  episode")
            print(f"    Total: {sum(batch_stats[file_type].values())}  episode")
    
    print()
    print(f"Total copied: {len(all_copy_tasks)} file")
    
    if dry_run:
        print()
        print("⚠️  dry-run mode, no actual copy operations performed")
    else:
        print()
        print(f"✅ Completed! Output directory:")
        print(f"   hdf5: {output_dir / 'hdf5'}")
        print(f"   svo2: {output_dir / 'svo2'}")


def reorder_episodes(input_dir: Path, output_dir: Path, file_type: str, dry_run: bool = False, num_workers: int = 8):
    """
    Reorder episode files in chronological order.
    
    Args:
        input_dir: Input directory
        output_dir: output directory
        file_type: File type ('hdf5', 'svo2' or 'all')
        dry_run: whether preview mode
        num_workers: number of threads for parallel copying
    """
    if file_type == 'all':
        # Process both hdf5 and svo2 simultaneously
        reorder_episodes_all(input_dir, output_dir, dry_run, num_workers)
    else:
        reorder_episodes_for_type(input_dir, output_dir, file_type, dry_run, num_workers)


def main():
    # Get CPU core count as default number of threads
    default_workers = min(os.cpu_count() or 8, 16)
    
    parser = argparse.ArgumentParser(
        description="Reorder episode files in chronological order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Execute reordering of hdf5 files (output to output_dir/hdf5/)
  python reorder_episodes.py --input_dir /path/to/input --output_dir /path/to/output --file hdf5
  
  # Execute reordering of svo2 files (output to output_dir/svo2/)
  python reorder_episodes.py --input_dir /path/to/input --output_dir /path/to/output --file svo2
  
  # Reorder hdf5 and svo2 files simultaneously (output to output_dir/hdf5/ and output_dir/svo2/)
  python reorder_episodes.py --input_dir /path/to/input --output_dir /path/to/output --file all
  
  # use 16 threads for parallel copying
  python reorder_episodes.py --input_dir /path/to/input --output_dir /path/to/output --file all --workers 16
  
  # Preview mode (no actual operations)
  python reorder_episodes.py --input_dir /path/to/input --output_dir /path/to/output --file hdf5 --dry-run
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
        help='Output directory path (will create hdf5/ or svo2/ subdirectories under this directory)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        choices=['hdf5', 'svo2', 'all'],
        help='File type: hdf5, svo2 or all (process both types simultaneously)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=default_workers,
        help=f'Number of threads for parallel copying (default: {default_workers}）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview mode, only print operations, don't perform actual copying'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    reorder_episodes(input_dir, output_dir, file_type=args.file, dry_run=args.dry_run, num_workers=args.workers)


if __name__ == "__main__":
    main()

