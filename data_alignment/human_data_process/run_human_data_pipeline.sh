#!/bin/bash
#
# Human data processing pipeline script (conda environment with ZED)
#
# Functions:
#   1. Run reorder_episodes_for_raw.py to reorder raw data
#   2. Run process_navigation_pipeline.py to inject navigation_command into HDF5 files
#   3. Run downsample_episode.py to downsample data
#   4. Run merge_camera_only.py to integrate camera images into HDF5 files
#   5. Run add_hand_status.py to compute hand open/close status from raw hand poses and write to final files
#
# Usage:
#   ./run_human_data_pipeline.sh --input_dir <input_dir> --output_dir <output_dir> [options]
#
# Example:

# ./run_human_data_pipeline.sh  --input_dir /home/admins/psj_ws/new_zed_mini_ws/toy/ --output_dir /home/admins/psj_ws/new_zed_mini_ws/toy/reorder --file all --final-output-dir ../output
# --input_dir /home/admins/psj_ws/new_zed_mini_ws/toy/     \
# --output_dir /home/admins/psj_ws/new_zed_mini_ws/toy/reorder   \
# --file all  \
# --final-output-dir ../output

set -e  # Exit immediately on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default parameters
INPUT_DIR=""
OUTPUT_DIR=""
FINAL_OUTPUT_DIR=""
FILE_TYPE="all"
DRY_RUN=""
WORKERS_NUM="32"
BASELINE_SEC="15"
TANGENT_LAG="5"
OVERWRITE="--overwrite"
NO_PNG="--no-png"
SKIP_REORDER=""
SKIP_NAVIGATION=""
SKIP_DOWNSAMPLE=""
SKIP_MERGE=""
SKIP_HAND_STATUS=""
DOWNSAMPLE_RATE="5"
# Print usage instructions
usage() {
    echo "Usage: $0 --input_dir <input_dir> --output_dir <output_dir> --final-output-dir <final_output_dir> [options]"
    echo ""
    echo "Required parameters:"
    echo "  --input_dir <path>          Raw data input directory (containing {date}_{batch} subdirectories)"
    echo "  --output_dir <path>         Intermediate output directory (hdf5/ and svo2/ subdirectories will be created here)"
    echo "  --final-output-dir <path>   Final output directory (for files with integrated camera images)"
    echo ""
    echo "Optional parameters:"
    echo "  --file <type>           File type: hdf5, svo2, or all (default: all)"
    echo "  --workers <number>      Number of parallel threads (default: 16)"
    echo "  --baseline-sec <sec>    Tangent baseline smoothing window in seconds (default: 15)"
    echo "  --tangent-lag <frames>  Tangent direction estimation lag in frames (default: 5)"
    echo "  --no-overwrite          Do not overwrite existing data"
    echo "  --with-png              Generate comparison PNG images"
    echo "  --skip-reorder          Skip reorder step"
    echo "  --skip-navigation       Skip navigation injection step"
    echo "  --skip-downsample       Skip downsample step"
    echo "  --skip-merge            Skip camera image integration step"
    echo "  --skip-hand-status      Skip hand status computation step"
    echo "  --downsample-rate <rate> Downsample rate (default: 5)"
    echo "  --dry-run               Preview mode, do not execute actual operations"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Full pipeline"
    echo "  $0 --input_dir /data/raw --output_dir /data/processed --final-output-dir /data/final"
    echo ""
    echo "  # Skip reorder (already have reordered data)"
    echo "  $0 --input_dir /data/raw --output_dir /data/processed --final-output-dir /data/final --skip-reorder"
    echo ""
    echo "  # Preview mode"
    echo "  $0 --input_dir /data/raw --output_dir /data/processed --final-output-dir /data/final --dry-run"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --final-output-dir)
            FINAL_OUTPUT_DIR="$2"
            shift 2
            ;;
        --file)
            FILE_TYPE="$2"
            shift 2
            ;;
        --workers)
            WORKERS_NUM="$2"
            shift 2
            ;;
        --baseline-sec)
            BASELINE_SEC="$2"
            shift 2
            ;;
        --tangent-lag)
            TANGENT_LAG="$2"
            shift 2
            ;;
        --no-overwrite)
            OVERWRITE=""
            shift
            ;;
        --with-png)
            NO_PNG=""
            shift
            ;;
        --skip-reorder)
            SKIP_REORDER="true"
            shift
            ;;
        --skip-navigation)
            SKIP_NAVIGATION="true"
            shift
            ;;
        --skip-downsample)
            SKIP_DOWNSAMPLE="true"
            shift
            ;;
        --skip-merge)
            SKIP_MERGE="true"
            shift
            ;;
        --skip-hand-status)
            SKIP_HAND_STATUS="true"
            shift
            ;;
        --downsample-rate)
            DOWNSAMPLE_RATE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown parameter '$1'"
            usage
            ;;
    esac
done

# Check required parameters
if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: Missing --input_dir parameter"
    usage
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing --output_dir parameter"
    usage
fi

if [[ -z "$FINAL_OUTPUT_DIR" ]]; then
    echo "Error: Missing --final-output-dir parameter"
    usage
fi

# Build WORKERS parameter
WORKERS="--workers $WORKERS_NUM"

# Print configuration
echo "========================================"
echo "🚀 Human Data Processing Pipeline"
echo "========================================"
echo "Input directory:       $INPUT_DIR"
echo "Intermediate output:   $OUTPUT_DIR"
echo "Final output:          $FINAL_OUTPUT_DIR"
echo "File type:             $FILE_TYPE"
echo "Parallel threads:      $WORKERS_NUM"
echo "baseline-sec:          $BASELINE_SEC"
echo "tangent-lag:           $TANGENT_LAG"
echo "Overwrite mode:        $([ -n "$OVERWRITE" ] && echo 'Yes' || echo 'No')"
echo "Generate PNG:          $([ -z "$NO_PNG" ] && echo 'Yes' || echo 'No')"
echo "Skip reorder:          $([ -n "$SKIP_REORDER" ] && echo 'Yes' || echo 'No')"
echo "Skip navigation:       $([ -n "$SKIP_NAVIGATION" ] && echo 'Yes' || echo 'No')"
echo "Skip downsample:       $([ -n "$SKIP_DOWNSAMPLE" ] && echo 'Yes' || echo 'No')"
echo "Skip image merge:      $([ -n "$SKIP_MERGE" ] && echo 'Yes' || echo 'No')"
echo "Skip hand status:      $([ -n "$SKIP_HAND_STATUS" ] && echo 'Yes' || echo 'No')"
echo "Downsample rate:       $DOWNSAMPLE_RATE"
echo "Preview mode:          $([ -n "$DRY_RUN" ] && echo 'Yes' || echo 'No')"
echo "========================================"
echo ""

# Step 1: Run reorder script
if [[ -z "$SKIP_REORDER" ]]; then
    echo "========================================"
    echo "📋 Step 1: Run reorder script"
    echo "========================================"
    echo ""

    REORDER_CMD="python ${SCRIPT_DIR}/scripts/reorder_episodes_for_raw.py \
        --input_dir \"$INPUT_DIR\" \
        --output_dir \"$OUTPUT_DIR\" \
        --file $FILE_TYPE \
        $WORKERS \
        $DRY_RUN"

    echo "Executing command:"
    echo "$REORDER_CMD"
    echo ""

    eval $REORDER_CMD

    echo ""
    echo "✅ Step 1 completed"
    echo ""
else
    echo "⏭️  Skipping Step 1 (reorder)"
    echo ""
fi

# Step 2: Run navigation pipeline
if [[ -z "$SKIP_NAVIGATION" ]]; then
    echo "========================================"
    echo "📋 Step 2: Run Navigation Pipeline"
    echo "========================================"
    echo ""

    # Check if hdf5 directory exists
    HDF5_DIR="${OUTPUT_DIR}/hdf5"
    if [[ ! -d "$HDF5_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "Error: hdf5 directory does not exist: $HDF5_DIR"
        echo "Please run the reorder step first, or check if the output directory path is correct"
        exit 1
    fi

    NAV_CMD="python ${SCRIPT_DIR}/process_navigation_pipeline.py \
        --dataset-dir \"$OUTPUT_DIR\" \
        --baseline-sec $BASELINE_SEC \
        --tangent-lag $TANGENT_LAG \
        $OVERWRITE \
        $NO_PNG \
        $DRY_RUN"

    echo "Executing command:"
    echo "$NAV_CMD"
    echo ""

    eval $NAV_CMD

    echo ""
    echo "✅ Step 2 completed"
    echo ""
else
    echo "⏭️  Skipping Step 2 (Navigation Pipeline)"
    echo ""
fi

# Step 3: Run downsample script
if [[ -z "$SKIP_DOWNSAMPLE" ]]; then
    echo "========================================"
    echo "📋 Step 3: Run downsample script"
    echo "========================================"
    echo ""

    # Check if hdf5 directory exists
    HDF5_DIR="${OUTPUT_DIR}/hdf5"
    if [[ ! -d "$HDF5_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "Error: hdf5 directory does not exist: $HDF5_DIR"
        echo "Please run the previous steps first, or check if the output directory path is correct"
        exit 1
    fi

    DOWNSAMPLE_CMD="python ${SCRIPT_DIR}/downsample_episode.py \
        --dataset-dir \"$OUTPUT_DIR\" \
        --downsample-rate $DOWNSAMPLE_RATE \
        $OVERWRITE \
        $DRY_RUN"

    echo "Executing command:"
    echo "$DOWNSAMPLE_CMD"
    echo ""

    eval $DOWNSAMPLE_CMD

    echo ""
    echo "✅ Step 3 completed"
    echo ""
else
    echo "⏭️  Skipping Step 3 (downsample)"
    echo ""
fi

# Step 4: Run camera image integration script
if [[ -z "$SKIP_MERGE" ]]; then
    echo "========================================"
    echo "📋 Step 4: Run camera image integration script"
    echo "========================================"
    echo ""

    # Check if hdf5 and svo2 directories exist
    HDF5_DIR="${OUTPUT_DIR}/hdf5"
    SVO2_DIR="${OUTPUT_DIR}/svo2"
    if [[ ! -d "$HDF5_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "Error: hdf5 directory does not exist: $HDF5_DIR"
        echo "Please run the previous steps first, or check if the output directory path is correct"
        exit 1
    fi
    if [[ ! -d "$SVO2_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "Error: svo2 directory does not exist: $SVO2_DIR"
        echo "Please run the previous steps first, or check if the output directory path is correct"
        exit 1
    fi

    MERGE_CMD="python ${SCRIPT_DIR}/merge_camera_only.py \
        --dataset-dir \"$OUTPUT_DIR\" \
        --output-dir \"$FINAL_OUTPUT_DIR\" \
        --num-workers $WORKERS_NUM"

    echo "Executing command:"
    echo "$MERGE_CMD"
    echo ""

    if [[ -z "$DRY_RUN" ]]; then
        eval $MERGE_CMD
    else
        echo "[dry-run] Will merge ${HDF5_DIR}/downsample_episode_*.hdf5 and ${SVO2_DIR}/episode_*.svo2"
        echo "[dry-run] Output to: $FINAL_OUTPUT_DIR"
    fi

    echo ""
    echo "✅ Step 4 completed"
    echo ""
else
    echo "⏭️  Skipping Step 4 (camera image integration)"
    echo ""
fi

# Step 5: Run hand status computation script
if [[ -z "$SKIP_HAND_STATUS" ]]; then
    echo "========================================"
    echo "📋 Step 5: Run hand status computation script"
    echo "========================================"
    echo ""

    # Check if raw hdf5 directory and final output directory exist
    HDF5_DIR="${OUTPUT_DIR}/hdf5"
    if [[ ! -d "$HDF5_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "Error: hdf5 directory does not exist: $HDF5_DIR"
        echo "Please run the previous steps first, or check if the output directory path is correct"
        exit 1
    fi
    if [[ ! -d "$FINAL_OUTPUT_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "Error: final output directory does not exist: $FINAL_OUTPUT_DIR"
        echo "Please run Step 4 (camera image integration) first, or check if the path is correct"
        exit 1
    fi

    HAND_STATUS_CMD="python ${SCRIPT_DIR}/add_hand_status.py \
        --raw \"$HDF5_DIR\" \
        --mid \"$FINAL_OUTPUT_DIR\" \
        --target \"$FINAL_OUTPUT_DIR\" \
        --downsample $DOWNSAMPLE_RATE \
        --num_workers $WORKERS_NUM"

    echo "Executing command:"
    echo "$HAND_STATUS_CMD"
    echo ""

    if [[ -z "$DRY_RUN" ]]; then
        eval $HAND_STATUS_CMD
    else
        echo "[dry-run] Will read hand pose data from ${HDF5_DIR}/episode_*.hdf5"
        echo "[dry-run] Will write hand_status to ${FINAL_OUTPUT_DIR}/episode_*.hdf5 (in-place mode)"
    fi

    echo ""
    echo "✅ Step 5 completed"
    echo ""
else
    echo "⏭️  Skipping Step 5 (hand status computation)"
    echo ""
fi

echo "========================================"
echo "🎉 Pipeline execution completed!"
echo "========================================"
echo "Intermediate output directory: $OUTPUT_DIR"
if [[ "$FILE_TYPE" == "all" ]] || [[ "$FILE_TYPE" == "hdf5" ]]; then
    echo "  - hdf5: ${OUTPUT_DIR}/hdf5/"
    echo "    - episode_*.hdf5 (raw + navigation_command)"
    if [[ -z "$SKIP_DOWNSAMPLE" ]]; then
        echo "    - downsample_episode_*.hdf5 (after downsampling)"
    fi
fi
if [[ "$FILE_TYPE" == "all" ]] || [[ "$FILE_TYPE" == "svo2" ]]; then
    echo "  - svo2: ${OUTPUT_DIR}/svo2/"
fi
if [[ -z "$SKIP_MERGE" ]] || [[ -z "$SKIP_HAND_STATUS" ]]; then
    echo ""
    echo "Final output directory: $FINAL_OUTPUT_DIR"
    if [[ -z "$SKIP_HAND_STATUS" ]]; then
        echo "  - episode_*.hdf5 (with camera images + hand_status)"
    else
        echo "  - episode_*.hdf5 (with camera images)"
    fi
fi
echo "========================================"

