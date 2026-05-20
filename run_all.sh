#!/usr/bin/env bash
set -euo pipefail

# 进入脚本所在目录，避免从别处启动时相对路径混乱
cd "$(dirname "$0")"

# 使用 1,2 号 GPU
export CUDA_VISIBLE_DEVICES=1,2

CONFIG_DIR="./config"
NPROC=2

# 为每次 torchrun 分配不同端口，避免端口残留冲突
BASE_PORT=29500
RUN_IDX=0

# 重试配置
# MAX_RETRIES=-1 表示无限重试；=3 表示最多重试 3 次（总共最多执行 4 次：1次初始 + 3次重试）
MAX_RETRIES=1
RETRY_WAIT_SECONDS=10

# 日志目录
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

# 临时目录：希望 torchelastic 不要创建到项目根目录，而是放到 ./temp
TEMP_DIR="./temp"
mkdir -p "${TEMP_DIR}"

# 尝试引导 Python / torch / tempfile 将临时文件写入 ./temp
export TMPDIR="${TEMP_DIR}"
export TMP="${TEMP_DIR}"
export TEMP="${TEMP_DIR}"

# 总日志文件
TIMESTAMP=$(date '+%F_%H-%M-%S')
MASTER_LOG="${LOG_DIR}/run_all_${TIMESTAMP}.log"

cleanup() {
    echo "[$(date '+%F %T')] Cleaning temporary files..."
    find "${TEMP_DIR}" -maxdepth 1 -type d -name 'torchelastic_*' -exec rm -rf {} + 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_exp() {
    local expid="$1"
    local exp_log="${LOG_DIR}/${expid}_${TIMESTAMP}.log"
    local attempt=0

    while true; do
        attempt=$((attempt + 1))
        local port=$((BASE_PORT + RUN_IDX))
        RUN_IDX=$((RUN_IDX + 1))

        echo "=================================================="
        echo "[$(date '+%F %T')] Starting experiment: ${expid} | attempt=${attempt} | master_port=${port}"
        echo "=================================================="

        # 每次实验前清理一次旧的 torchelastic 临时目录
        find "${TEMP_DIR}" -maxdepth 1 -type d -name 'torchelastic_*' -exec rm -rf {} + 2>/dev/null || true

        # 注意：使用 if 包裹失败命令，可以避免 set -e 直接退出脚本
        if torchrun \
            --standalone \
            --master_port="${port}" \
            --nproc_per_node="${NPROC}" \
            run_expid.py \
            --config "${CONFIG_DIR}" \
            --expid "${expid}" 2>&1 | tee -a "${exp_log}"; then

            echo "[$(date '+%F %T')] Finished experiment: ${expid} (attempt=${attempt})"

            # 每次实验结束后清理 torchelastic 临时目录
            find "${TEMP_DIR}" -maxdepth 1 -type d -name 'torchelastic_*' -exec rm -rf {} + 2>/dev/null || true

            echo
            break
        else
            rc=$?
            echo "[$(date '+%F %T')] ERROR: experiment ${expid} failed (attempt=${attempt}, exit_code=${rc})"

            # 失败后也清理一次，避免重试前残留
            find "${TEMP_DIR}" -maxdepth 1 -type d -name 'torchelastic_*' -exec rm -rf {} + 2>/dev/null || true

            # 达到最大重试次数则退出整个脚本
            if [[ "${MAX_RETRIES}" -ge 0 && "${attempt}" -gt "${MAX_RETRIES}" ]]; then
                echo "[$(date '+%F %T')] ERROR: experiment ${expid} exceeded max retries (${MAX_RETRIES}). Abort."
                exit "${rc}"
            fi

            echo "[$(date '+%F %T')] Retrying ${expid} after ${RETRY_WAIT_SECONDS}s..."
            sleep "${RETRY_WAIT_SECONDS}"
        fi
    done
}

# 整个脚本输出也写入总日志
exec > >(tee -a "${MASTER_LOG}") 2>&1

# 依次运行

# run_exp "LONGER_QK_Video_Action"
# run_exp "LONGER_KuaiRand_Video_Action"
# run_exp "LONGER_TencentGR_10M_Action"

# run_exp "Alloy_KuaiRand_Video_Action"
# run_exp "Alloy_QK_Video_Action"
# run_exp "Alloy_TencentGR_10M_Action"

# run_exp "UltraHSTU_KuaiRand_Video_Action"
# run_exp "UltraHSTU_QK_Video_Action"
# run_exp "UltraHSTU_TencentGR_10M_Action"

# run_exp "DIN_QK_Video_Action"
# run_exp "DIN_KuaiRand_Video_Action"
# run_exp "DIN_TencentGR_10M_Action"

# run_exp "RankMixer_QK_Video_Action"
# run_exp "RankMixer_KuaiRand_Video_Action"
# run_exp "RankMixer_TencentGR_10M_Action"

# run_exp "HeMix_KuaiRand_Video_Action"
# run_exp "HeMix_QK_Video_Action"
# run_exp "HeMix_TencentGR_10M_Action"

# run_exp "HiFormer_QK_Video_Action"
# run_exp "HiFormer_KuaiRand_Video_Action"
# run_exp "HiFormer_TencentGR_10M_Action"

# run_exp "Zenith_KuaiRand_Video_Action"
# run_exp "Zenith_QK_Video_Action"
# run_exp "Zenith_TencentGR_10M_Action"

# run_exp "OneTrans_QK_Video_Action"
# run_exp "OneTrans_KuaiRand_Video_Action"
# run_exp "OneTrans_TencentGR_10M_Action"

# run_exp "HyFormer_QK_Video_Action"
# run_exp "HyFormer_KuaiRand_Video_Action"
# run_exp "HyFormer_TencentGR_10M_Action"

# run_exp "MixFormer_QK_Video_Action"
# run_exp "MixFormer_KuaiRand_Video_Action"
# run_exp "MixFormer_TencentGR_10M_Action"

# run_exp "INFNet_QK_Video_Action"
# run_exp "INFNet_KuaiRand_Video_Action"
# run_exp "INFNet_TencentGR_10M_Action"

# run_exp "EST_KuaiRand_Video_Action"
# run_exp "EST_QK_Video_Action"
# run_exp "EST_TencentGR_10M_Action"

# run_exp "UniMixer_KuaiRand_Video_Action"
# run_exp "UniMixer_QK_Video_Action"
# run_exp "UniMixer_TencentGR_10M_Action"

# run_exp "TokenFormer_QK_Video_Action"
 run_exp "TokenFormer_KuaiRand_Video_Action"
 run_exp "TokenFormer_TencentGR_10M_Action"

echo "All experiments completed successfully."