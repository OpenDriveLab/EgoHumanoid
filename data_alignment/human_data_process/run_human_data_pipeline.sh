#!/bin/bash
#
# 人体数据处理流水线脚本(caonda 环境 zed)
#
# 功能：
#   1. 运行 reorder_episodes_for_raw.py 对原始数据进行重排序
#   2. 运行 process_navigation_pipeline.py 将 navigation_command 注入 HDF5 文件
#   3. 运行 downsample_episode.py 对数据进行降采样处理
#   4. 运行 merge_camera_only.py 将相机图像数据整合到 HDF5 文件
#   5. 运行 add_hand_status.py 从原始手部姿态计算手部开合状态并写入最终文件
#
# 用法：
#   ./run_human_data_pipeline.sh --input_dir <输入目录> --output_dir <输出目录> [选项]
#
# 示例：

# ./run_human_data_pipeline.sh  --input_dir /home/admins/psj_ws/new_zed_mini_ws/toy/ --output_dir /home/admins/psj_ws/new_zed_mini_ws/toy/reorder --file all --final-output-dir ../output
# --input_dir /home/admins/psj_ws/new_zed_mini_ws/toy/     \ 
# --output_dir /home/admins/psj_ws/new_zed_mini_ws/toy/reorder   \ 
# --file all  \
# --final-output-dir ../output

set -e  # 遇到错误立即退出

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认参数
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
# 打印使用说明
usage() {
    echo "用法: $0 --input_dir <输入目录> --output_dir <输出目录> --final-output-dir <最终输出目录> [选项]"
    echo ""
    echo "必需参数:"
    echo "  --input_dir <路径>          原始数据输入目录（包含 {日期}_{批次} 子目录）"
    echo "  --output_dir <路径>         中间输出目录（会在此目录下创建 hdf5/ svo2/ 子目录）"
    echo "  --final-output-dir <路径>   最终输出目录（整合相机图像后的文件）"
    echo ""
    echo "可选参数:"
    echo "  --file <类型>           文件类型: hdf5, svo2, all (默认: all)"
    echo "  --workers <数量>        并行线程数 (默认: 16)"
    echo "  --baseline-sec <秒>     tangent 基准平滑窗口秒数 (默认: 15)"
    echo "  --tangent-lag <帧>      tangent 切向差分间隔帧数 (默认: 5)"
    echo "  --no-overwrite          不覆盖已存在的数据"
    echo "  --with-png              生成对比 PNG 图片"
    echo "  --skip-reorder          跳过重排序步骤"
    echo "  --skip-navigation       跳过 navigation 注入步骤"
    echo "  --skip-downsample       跳过降采样步骤"
    echo "  --skip-merge            跳过相机图像整合步骤"
    echo "  --skip-hand-status      跳过手部开合状态计算步骤"
    echo "  --downsample-rate <率>  降采样率 (默认: 5)"
    echo "  --dry-run               预览模式，不执行实际操作"
    echo "  -h, --help              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 完整流水线"
    echo "  $0 --input_dir /data/raw --output_dir /data/processed --final-output-dir /data/final"
    echo ""
    echo "  # 跳过重排序（已有重排序数据）"
    echo "  $0 --input_dir /data/raw --output_dir /data/processed --final-output-dir /data/final --skip-reorder"
    echo ""
    echo "  # 预览模式"
    echo "  $0 --input_dir /data/raw --output_dir /data/processed --final-output-dir /data/final --dry-run"
    exit 1
}

# 解析命令行参数
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
            echo "错误: 未知参数 '$1'"
            usage
            ;;
    esac
done

# 检查必需参数
if [[ -z "$INPUT_DIR" ]]; then
    echo "错误: 缺少 --input_dir 参数"
    usage
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误: 缺少 --output_dir 参数"
    usage
fi

if [[ -z "$FINAL_OUTPUT_DIR" ]]; then
    echo "错误: 缺少 --final-output-dir 参数"
    usage
fi

# 构建 WORKERS 参数
WORKERS="--workers $WORKERS_NUM"

# 打印配置信息
echo "========================================"
echo "🚀 人体数据处理流水线"
echo "========================================"
echo "输入目录:       $INPUT_DIR"
echo "中间输出目录:   $OUTPUT_DIR"
echo "最终输出目录:   $FINAL_OUTPUT_DIR"
echo "文件类型:       $FILE_TYPE"
echo "并行线程数:     $WORKERS_NUM"
echo "baseline-sec:   $BASELINE_SEC"
echo "tangent-lag:    $TANGENT_LAG"
echo "覆盖模式:       $([ -n "$OVERWRITE" ] && echo '是' || echo '否')"
echo "生成PNG:        $([ -z "$NO_PNG" ] && echo '是' || echo '否')"
echo "跳过重排序:     $([ -n "$SKIP_REORDER" ] && echo '是' || echo '否')"
echo "跳过navigation: $([ -n "$SKIP_NAVIGATION" ] && echo '是' || echo '否')"
echo "跳过降采样:     $([ -n "$SKIP_DOWNSAMPLE" ] && echo '是' || echo '否')"
echo "跳过图像整合:   $([ -n "$SKIP_MERGE" ] && echo '是' || echo '否')"
echo "跳过手部状态:   $([ -n "$SKIP_HAND_STATUS" ] && echo '是' || echo '否')"
echo "降采样率:       $DOWNSAMPLE_RATE"
echo "预览模式:       $([ -n "$DRY_RUN" ] && echo '是' || echo '否')"
echo "========================================"
echo ""

# Step 1: 运行重排序脚本
if [[ -z "$SKIP_REORDER" ]]; then
    echo "========================================"
    echo "📋 Step 1: 运行重排序脚本"
    echo "========================================"
    echo ""
    
    REORDER_CMD="python ${SCRIPT_DIR}/scripts/reorder_episodes_for_raw.py \
        --input_dir \"$INPUT_DIR\" \
        --output_dir \"$OUTPUT_DIR\" \
        --file $FILE_TYPE \
        $WORKERS \
        $DRY_RUN"
    
    echo "执行命令:"
    echo "$REORDER_CMD"
    echo ""
    
    eval $REORDER_CMD
    
    echo ""
    echo "✅ Step 1 完成"
    echo ""
else
    echo "⏭️  跳过 Step 1 (重排序)"
    echo ""
fi

# Step 2: 运行 navigation pipeline
if [[ -z "$SKIP_NAVIGATION" ]]; then
    echo "========================================"
    echo "📋 Step 2: 运行 Navigation Pipeline"
    echo "========================================"
    echo ""
    
    # 检查 hdf5 目录是否存在
    HDF5_DIR="${OUTPUT_DIR}/hdf5"
    if [[ ! -d "$HDF5_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "错误: hdf5 目录不存在: $HDF5_DIR"
        echo "请先运行重排序步骤，或检查输出目录路径是否正确"
        exit 1
    fi
    
    NAV_CMD="python ${SCRIPT_DIR}/process_navigation_pipeline.py \
        --dataset-dir \"$OUTPUT_DIR\" \
        --baseline-sec $BASELINE_SEC \
        --tangent-lag $TANGENT_LAG \
        $OVERWRITE \
        $NO_PNG \
        $DRY_RUN"
    
    echo "执行命令:"
    echo "$NAV_CMD"
    echo ""
    
    eval $NAV_CMD
    
    echo ""
    echo "✅ Step 2 完成"
    echo ""
else
    echo "⏭️  跳过 Step 2 (Navigation Pipeline)"
    echo ""
fi

# Step 3: 运行降采样脚本
if [[ -z "$SKIP_DOWNSAMPLE" ]]; then
    echo "========================================"
    echo "📋 Step 3: 运行降采样脚本"
    echo "========================================"
    echo ""
    
    # 检查 hdf5 目录是否存在
    HDF5_DIR="${OUTPUT_DIR}/hdf5"
    if [[ ! -d "$HDF5_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "错误: hdf5 目录不存在: $HDF5_DIR"
        echo "请先运行前面的步骤，或检查输出目录路径是否正确"
        exit 1
    fi
    
    DOWNSAMPLE_CMD="python ${SCRIPT_DIR}/downsample_episode.py \
        --dataset-dir \"$OUTPUT_DIR\" \
        --downsample-rate $DOWNSAMPLE_RATE \
        $OVERWRITE \
        $DRY_RUN"
    
    echo "执行命令:"
    echo "$DOWNSAMPLE_CMD"
    echo ""
    
    eval $DOWNSAMPLE_CMD
    
    echo ""
    echo "✅ Step 3 完成"
    echo ""
else
    echo "⏭️  跳过 Step 3 (降采样)"
    echo ""
fi

# Step 4: 运行相机图像整合脚本
if [[ -z "$SKIP_MERGE" ]]; then
    echo "========================================"
    echo "📋 Step 4: 运行相机图像整合脚本"
    echo "========================================"
    echo ""
    
    # 检查 hdf5 和 svo2 目录是否存在
    HDF5_DIR="${OUTPUT_DIR}/hdf5"
    SVO2_DIR="${OUTPUT_DIR}/svo2"
    if [[ ! -d "$HDF5_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "错误: hdf5 目录不存在: $HDF5_DIR"
        echo "请先运行前面的步骤，或检查输出目录路径是否正确"
        exit 1
    fi
    if [[ ! -d "$SVO2_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "错误: svo2 目录不存在: $SVO2_DIR"
        echo "请先运行前面的步骤，或检查输出目录路径是否正确"
        exit 1
    fi
    
    MERGE_CMD="python ${SCRIPT_DIR}/merge_camera_only.py \
        --dataset-dir \"$OUTPUT_DIR\" \
        --output-dir \"$FINAL_OUTPUT_DIR\" \
        --num-workers $WORKERS_NUM"
    
    echo "执行命令:"
    echo "$MERGE_CMD"
    echo ""
    
    if [[ -z "$DRY_RUN" ]]; then
        eval $MERGE_CMD
    else
        echo "[dry-run] 将整合 ${HDF5_DIR}/downsample_episode_*.hdf5 和 ${SVO2_DIR}/episode_*.svo2"
        echo "[dry-run] 输出到: $FINAL_OUTPUT_DIR"
    fi
    
    echo ""
    echo "✅ Step 4 完成"
    echo ""
else
    echo "⏭️  跳过 Step 4 (相机图像整合)"
    echo ""
fi

# Step 5: 运行手部开合状态计算脚本
if [[ -z "$SKIP_HAND_STATUS" ]]; then
    echo "========================================"
    echo "📋 Step 5: 运行手部开合状态计算脚本"
    echo "========================================"
    echo ""
    
    # 检查原始 hdf5 目录和最终输出目录是否存在
    HDF5_DIR="${OUTPUT_DIR}/hdf5"
    if [[ ! -d "$HDF5_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "错误: hdf5 目录不存在: $HDF5_DIR"
        echo "请先运行前面的步骤，或检查输出目录路径是否正确"
        exit 1
    fi
    if [[ ! -d "$FINAL_OUTPUT_DIR" ]] && [[ -z "$DRY_RUN" ]]; then
        echo "错误: 最终输出目录不存在: $FINAL_OUTPUT_DIR"
        echo "请先运行 Step 4 (相机图像整合)，或检查路径是否正确"
        exit 1
    fi
    
    HAND_STATUS_CMD="python ${SCRIPT_DIR}/add_hand_status.py \
        --raw \"$HDF5_DIR\" \
        --mid \"$FINAL_OUTPUT_DIR\" \
        --target \"$FINAL_OUTPUT_DIR\" \
        --downsample $DOWNSAMPLE_RATE \
        --num_workers $WORKERS_NUM"
    
    echo "执行命令:"
    echo "$HAND_STATUS_CMD"
    echo ""
    
    if [[ -z "$DRY_RUN" ]]; then
        eval $HAND_STATUS_CMD
    else
        echo "[dry-run] 将从 ${HDF5_DIR}/episode_*.hdf5 读取手部姿态数据"
        echo "[dry-run] 将 hand_status 写入 ${FINAL_OUTPUT_DIR}/episode_*.hdf5 (原地模式)"
    fi
    
    echo ""
    echo "✅ Step 5 完成"
    echo ""
else
    echo "⏭️  跳过 Step 5 (手部开合状态计算)"
    echo ""
fi

echo "========================================"
echo "🎉 流水线执行完成！"
echo "========================================"
echo "中间输出目录: $OUTPUT_DIR"
if [[ "$FILE_TYPE" == "all" ]] || [[ "$FILE_TYPE" == "hdf5" ]]; then
    echo "  - hdf5: ${OUTPUT_DIR}/hdf5/"
    echo "    - episode_*.hdf5 (原始+navigation_command)"
    if [[ -z "$SKIP_DOWNSAMPLE" ]]; then
        echo "    - downsample_episode_*.hdf5 (降采样后)"
    fi
fi
if [[ "$FILE_TYPE" == "all" ]] || [[ "$FILE_TYPE" == "svo2" ]]; then
    echo "  - svo2: ${OUTPUT_DIR}/svo2/"
fi
if [[ -z "$SKIP_MERGE" ]] || [[ -z "$SKIP_HAND_STATUS" ]]; then
    echo ""
    echo "最终输出目录: $FINAL_OUTPUT_DIR"
    if [[ -z "$SKIP_HAND_STATUS" ]]; then
        echo "  - episode_*.hdf5 (包含相机图像 + hand_status)"
    else
        echo "  - episode_*.hdf5 (包含相机图像)"
    fi
fi
echo "========================================"

