#!/bin/bash
#SBATCH --job-name=miles-opd-qwen3
#SBATCH --nodes=3
#SBATCH --partition=h200-hi-preempt
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/miles-opd-qwen3-%j.out
#SBATCH --error=./logs/miles-opd-qwen3-%j.err
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --wait-all-nodes=1
# Nodes with broken pyxis: --container-mounts silently no-ops on these hosts.
#SBATCH --exclude=ip-10-1-8-7,ip-10-1-7-180,ip-10-1-38-198,ip-10-1-142-67,ip-10-1-126-226,ip-10-1-141-192,ip-10-1-94-38

# =========================================================================
# Qwen3-8B student + Qwen3-32B teacher online policy distillation (OPD).
#
# Default layout  NON-COLOCATE 2+1:
#   Node 0   — Ray head + actor training (Qwen3-8B Megatron, TP=2)
#   Node 1   — Rollout sglang (student, 8 GPUs)
#   Node 2   — Teacher sglang (Qwen3-32B, TP=8) — NOT in the Ray cluster
#
# The teacher node runs a standalone sglang server. miles queries it via
# --rm-url for per-token log-probs; the OPD-KL penalty is applied to GRPO
# advantages. Same-tokenizer OPD by default (Qwen3-8B and Qwen3-32B share
# the same tokenizer). For cross-tokenizer distillation (e.g. Llama-3-8B
# student + Qwen3-32B teacher) uncomment the --opd-cross-* block below.
#
# Resubmit knobs (env vars):
#   IMAGE=/fsx/peng/containers/miles-kimi-k26-main-20260605.sqsh
#   MILES_HOST=/fsx/peng/workspace/miles-oss-workspace-1/miles
#   STUDENT_CHECKPOINT=/fsx/peng/models/Qwen3-8B
#   STUDENT_REF_CHECKPOINT=<torch_dist dir>   (default: same as HF checkpoint)
#   TEACHER_CHECKPOINT=/fsx/peng/models/Qwen3-30B-A3B
#   ACTOR_NODES=1  TEACHER_TP=8  OPD_KL_COEF=1.0
#   NUM_ROLLOUT=300  MAX_RESPONSE_LEN=16384
# =========================================================================

set -euxo pipefail
mkdir -p ./logs

# ---------- Image ----------
IMAGE=${IMAGE:-/fsx/peng/containers/miles-kimi-k26-main-20260605.sqsh}
if [[ ! -f "${IMAGE}" ]]; then echo "ERROR: IMAGE not found: ${IMAGE}" >&2; exit 1; fi

# ---------- Checkpoints ----------
# Student (Qwen3-8B): HF format for hf-checkpoint and sglang rollout;
#   torch_dist for ref (set to HF dir if not yet converted).
STUDENT_CHECKPOINT=${STUDENT_CHECKPOINT:-/fsx/peng/models/Qwen3-8B}
STUDENT_REF_CHECKPOINT=${STUDENT_REF_CHECKPOINT:-${STUDENT_CHECKPOINT}}
# Teacher: served by a standalone sglang on the dedicated teacher node.
# Same-tokenizer default: Qwen3-30B-A3B (same Qwen3 vocab as student).
# Cross-tokenizer example: GLM4.5-Air (GLM tokenizer, different from Qwen3).
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-/fsx/peng/models/Qwen3-30B-A3B}
TEACHER_PORT=${TEACHER_PORT:-13141}
TEACHER_TP=${TEACHER_TP:-8}
# Cross-tokenizer mode: set CROSS_TOKENIZER=1 to enable character-span alignment.
# The student rollout is decoded to text, re-tokenized with the teacher vocab,
# scored by the teacher server, and aligned back to student tokens.
CROSS_TOKENIZER=${CROSS_TOKENIZER:-0}
TEACHER_MEM_FRAC=${TEACHER_MEM_FRAC:-0.85}

# ---------- Paths ----------
MILES_HOST=${MILES_HOST:-/fsx/peng/workspace/miles-oss-workspace-1/miles}
DATASETS_HOST=${DATASETS_HOST:-/fsx/peng/workspace/kimi-26-training/datasets}
OUTPUTS_HOST=${OUTPUTS_HOST:-${MILES_HOST}/outputs}
mkdir -p "${OUTPUTS_HOST}"

MILES_ROOT=/workspace/miles
DATASETS_ROOT=/root/datasets
OUTPUTS_ROOT=/workspace/outputs
STUDENT_HF_ROOT=/root/student_hf
TEACHER_HF_ROOT=/root/teacher_hf

# Ray session dir must be on a local filesystem (Lustre does not support Unix sockets).
# /tmp is always local on EC2 nodes (root EBS or tmpfs, never Lustre). Use NVMe only
# if it's explicitly overridden, since not all node types have /opt/dlami/nvme.
RAY_TMP_HOST=${RAY_TMP_HOST:-/tmp/ray-tmp-${USER:-peng}-${SLURM_JOB_ID:-manual}}
RAY_TMP_ROOT=/tmp/ray-session
srun --overlap --ntasks-per-node=1 bash -c "mkdir -p ${RAY_TMP_HOST}" || {
  echo "ERROR: mkdir -p ${RAY_TMP_HOST} failed on one or more nodes (/tmp full?). Aborting before teacher load."
  exit 1
}

# If student ref is the same path as HF, reuse the same container mount point to
# avoid duplicate-source mounts which some pyxis/enroot versions reject.
if [[ "${STUDENT_REF_CHECKPOINT}" == "${STUDENT_CHECKPOINT}" ]]; then
  STUDENT_REF_ROOT=${STUDENT_HF_ROOT}
  REF_MOUNT=""
else
  STUDENT_REF_ROOT=/root/student_ref
  REF_MOUNT=",${STUDENT_REF_CHECKPOINT}:${STUDENT_REF_ROOT}"
fi

MILES_MOUNTS="${MILES_HOST}:${MILES_ROOT},\
${DATASETS_HOST}:${DATASETS_ROOT},\
${OUTPUTS_HOST}:${OUTPUTS_ROOT},\
${STUDENT_CHECKPOINT}:${STUDENT_HF_ROOT}${REF_MOUNT},\
${TEACHER_CHECKPOINT}:${TEACHER_HF_ROOT},\
${RAY_TMP_HOST}:${RAY_TMP_ROOT},\
/var/log/aws/clusters:/var/log/aws/clusters"

TEACHER_MOUNTS="${TEACHER_CHECKPOINT}:${TEACHER_HF_ROOT},\
/var/log/aws/clusters:/var/log/aws/clusters"

# Build cross-tokenizer args string (expanded by outer shell, injected into heredoc).
if [[ "${CROSS_TOKENIZER}" == "1" ]]; then
  CROSS_TOKENIZER_ARGS="--opd-cross-tokenizer --opd-cross-teacher-tokenizer ${TEACHER_HF_ROOT} --opd-cross-student-tokenizer ${STUDENT_HF_ROOT}"
  echo "== Cross-tokenizer OPD: student=${STUDENT_CHECKPOINT} teacher=${TEACHER_CHECKPOINT} =="
else
  CROSS_TOKENIZER_ARGS=""
  echo "== Same-tokenizer OPD: student=${STUDENT_CHECKPOINT} teacher=${TEACHER_CHECKPOINT} =="
fi

# Model architecture script — override for different student sizes (e.g. qwen3-4B.sh)
MODEL_SCRIPT=${MODEL_SCRIPT:-scripts/models/qwen3-8B.sh}
# Run name used for checkpoint save/load dir under OUTPUTS_ROOT
RUN_NAME=${RUN_NAME:-qwen3-8B-opd}

# ---------- Run knobs ----------
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
NUM_ROLLOUT=${NUM_ROLLOUT:-300}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-16384}
ROLLOUT_BATCH=${ROLLOUT_BATCH:-16}
N_SAMPLES=${N_SAMPLES:-4}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
OPD_KL_COEF=${OPD_KL_COEF:-1.0}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}

# ---------- Topology ----------
# Miles nodes: first MILES_NODES nodes join the Ray cluster.
# Teacher node: last node runs a standalone sglang server only.
NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-3}}
TEACHER_NODES=${TEACHER_NODES:-1}
MILES_NODES=$(( NNODES - TEACHER_NODES ))
ACTOR_NODES=${ACTOR_NODES:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$(( (MILES_NODES - ACTOR_NODES) * GPUS_PER_NODE ))}

if [[ "${NNODES}" -ne "${SLURM_JOB_NUM_NODES}" ]]; then
  echo "ERROR: NNODES=${NNODES} but slurm allocated ${SLURM_JOB_NUM_NODES} nodes." >&2; exit 1
fi
if [[ "${MILES_NODES}" -lt 1 ]]; then
  echo "ERROR: MILES_NODES=${MILES_NODES} must be at least 1." >&2; exit 1
fi

# Parallelism (Qwen3-8B is dense; TP=2 fits comfortably on 2 H200s per actor node).
TP=${TP:-2}
PP=${PP:-1}

# ---------- Cluster discovery ----------
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
# Miles nodes: indices 0..MILES_NODES-1; teacher node(s): last TEACHER_NODES nodes
MILES_NODE_ARRAY=("${nodes_array[@]:0:${MILES_NODES}}")
TEACHER_NODE_ARRAY=("${nodes_array[@]: -${TEACHER_NODES}}")
HEAD_NODE=${MILES_NODE_ARRAY[0]}
TEACHER_NODE=${TEACHER_NODE_ARRAY[0]}

cluster_ip_for_node() {
  local node=$1; local iface=$2
  srun --overlap --nodes=1 --ntasks=1 -w "${node}" \
    bash -lc "ip -o -4 addr show dev ${iface} | awk '{print \$4}' | cut -d/ -f1 | head -n1"
}
detect_cluster_iface_for_node() {
  local node=$1
  srun --overlap --nodes=1 --ntasks=1 -w "${node}" \
    bash -lc "ip route show default 0.0.0.0/0 | awk 'NR==1 {print \$5}'"
}
common_prefix() {
  local prefix=$1; shift; local value
  for value in "$@"; do
    while [[ -n "${prefix}" && "${value#"$prefix"}" == "${value}" ]]; do prefix=${prefix%?}; done
  done
  printf '%s' "${prefix}"
}

ALL_NODE_IPS=()
ALL_NODE_IFACES=()
if [[ -n "${CLUSTER_NET_IFACE:-}" ]]; then
  for _node in "${nodes_array[@]}"; do
    ALL_NODE_IFACES+=("${CLUSTER_NET_IFACE}")
    ALL_NODE_IPS+=("$(cluster_ip_for_node "${_node}" "${CLUSTER_NET_IFACE}")")
  done
  NCCL_SOCKET_IFNAME_VALUE="${CLUSTER_NET_IFACE}"
else
  for node in "${nodes_array[@]}"; do
    iface=$(detect_cluster_iface_for_node "${node}")
    [[ -z "${iface}" ]] && { echo "Failed to detect iface for ${node}" >&2; exit 1; }
    ALL_NODE_IFACES+=("${iface}")
    ip=$(cluster_ip_for_node "${node}" "${iface}")
    [[ -z "${ip}" ]] && { echo "Failed to resolve IP for ${node}" >&2; exit 1; }
    ALL_NODE_IPS+=("${ip}")
  done
  NCCL_SOCKET_IFNAME_VALUE=$(common_prefix "${ALL_NODE_IFACES[0]}" "${ALL_NODE_IFACES[@]:1}")
  [[ -z "${NCCL_SOCKET_IFNAME_VALUE}" ]] && NCCL_SOCKET_IFNAME_VALUE="^lo,docker,veth,tailscale"
fi

MASTER_ADDR=${ALL_NODE_IPS[0]}
MASTER_PORT=6379
TEACHER_IP=${ALL_NODE_IPS[-1]}
ALL_HOSTS_CSV=$(echo "${nodes}" | tr '\n' ',' | sed 's/,$//')
MILES_IPS_CSV=$(IFS=,; echo "${ALL_NODE_IPS[*]:0:${MILES_NODES}}")

RAY_LAUNCHER_PIDS=()

NVLINK_COUNT=$(srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  bash -lc "nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l" || echo 0)
[[ "${NVLINK_COUNT}" -gt 0 ]] && HAS_NVLINK=1 || HAS_NVLINK=0

echo "== head: ${HEAD_NODE} (${MASTER_ADDR}) | teacher: ${TEACHER_NODE} (${TEACHER_IP}:${TEACHER_PORT}) | miles_nodes=${MILES_NODES} actor_nodes=${ACTOR_NODES} rollout_gpus=${ROLLOUT_GPUS} | TP${TP} PP${PP} | nvlink=${HAS_NVLINK} =="

# ---------- Network / env ----------
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_PROTO="simple"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_VALUE}"
export NCCL_SOCKET_FAMILY="AF_INET"
export RDMAV_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG="WARN"
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export no_proxy="127.0.0.1,localhost,0.0.0.0,${MASTER_ADDR},${TEACHER_IP},${MILES_IPS_CSV},${ALL_HOSTS_CSV}"
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=${WANDB_API_KEY:-}

# ---------- Cleanup ----------
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM; set +e
  echo "Cleaning up (exit=${exit_code})..."
  for node in "${MILES_NODE_ARRAY[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${node}" \
      --container-image "${IMAGE}" --container-mounts "${MILES_MOUNTS}" --container-remap-root \
      bash -lc "ray stop --force || true" || true
  done
  for pid in "${RAY_LAUNCHER_PIDS[@]}"; do
    kill -0 "${pid}" 2>/dev/null && kill "${pid}" 2>/dev/null || true
  done
  wait || true
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

########################################
# 1. Start teacher sglang server
########################################
echo "Starting teacher sglang at ${TEACHER_NODE} (${TEACHER_IP}:${TEACHER_PORT})..."
srun --nodes=1 --ntasks=1 -w "${TEACHER_NODE}" \
  --container-image "${IMAGE}" --container-mounts "${TEACHER_MOUNTS}" --container-remap-root \
  bash -lc "
    set -euxo pipefail
    python3 -m sglang.launch_server \
      --model-path ${TEACHER_HF_ROOT} \
      --host 0.0.0.0 \
      --port ${TEACHER_PORT} \
      --tp ${TEACHER_TP} \
      --mem-fraction-static ${TEACHER_MEM_FRAC} \
      --chunked-prefill-size 4096
  " &
RAY_LAUNCHER_PIDS+=($!)

TEACHER_PID=${RAY_LAUNCHER_PIDS[-1]}
TEACHER_MAX_PROBES=${TEACHER_MAX_PROBES:-2400}  # 240 × 10s = 40 min (H200: ~21 min, H100: ~34 min)
echo "Waiting for teacher server at http://${TEACHER_IP}:${TEACHER_PORT}/health_generate (pid=${TEACHER_PID}, max=${TEACHER_MAX_PROBES} probes)..."
for attempt in $(seq 1 "${TEACHER_MAX_PROBES}"); do
  if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
    echo "ERROR: teacher srun process (pid=${TEACHER_PID}) exited early — check container/mount errors above." >&2; exit 1
  fi
  if curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" > /dev/null 2>&1; then
    echo "== Teacher server healthy after ${attempt} probe(s) =="; break
  fi
  [[ "${attempt}" == "${TEACHER_MAX_PROBES}" ]] && { echo "ERROR: teacher server never became healthy after ${TEACHER_MAX_PROBES} probes." >&2; exit 1; }
  sleep 10
done

########################################
# 2. Start Ray head on first miles node
########################################
echo "Starting Ray HEAD at ${HEAD_NODE} (${MASTER_ADDR})..."
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  --container-image "${IMAGE}" --container-mounts "${MILES_MOUNTS}" --container-remap-root \
  --container-env=WANDB_API_KEY \
  bash -lc "
    set -euxo pipefail
    cd ${MILES_ROOT}
    export RAY_TMPDIR=${RAY_TMP_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1 | tail -3
    ray stop --force || true
    ray start --head --node-ip-address=\"${MASTER_ADDR}\" --port=${MASTER_PORT} \
      --num-gpus ${GPUS_PER_NODE} --disable-usage-stats \
      --temp-dir=${RAY_TMP_ROOT} --dashboard-host=0.0.0.0 --dashboard-port=8265 --block
  " &
RAY_LAUNCHER_PIDS+=($!)
sleep 10

########################################
# 3. Start Ray workers on remaining miles nodes
########################################
worker_num=$(( MILES_NODES - 1 ))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${MILES_NODE_ARRAY[$i]}
  echo "Starting Ray WORKER $i at ${node_i}..."
  srun --nodes=1 --ntasks=1 -w "${node_i}" \
    --container-image "${IMAGE}" --container-mounts "${MILES_MOUNTS}" --container-remap-root \
    --container-env=WANDB_API_KEY \
    bash -lc "
      set -euxo pipefail
      cd ${MILES_ROOT}
      export RAY_TMPDIR=${RAY_TMP_ROOT}
      pip install -e . --no-deps --break-system-packages --quiet 2>&1 | tail -3
      ray stop --force || true
      ray start --address ${MASTER_ADDR}:${MASTER_PORT} --num-gpus ${GPUS_PER_NODE} \
        --temp-dir=${RAY_TMP_ROOT} --disable-usage-stats --block
    " &
  RAY_LAUNCHER_PIDS+=($!)
  sleep 5
done

echo "Waiting for Ray cluster warm-up..."
sleep 30
for pid in "${RAY_LAUNCHER_PIDS[@]}"; do
  kill -0 "${pid}" 2>/dev/null || { echo "ERROR: launcher pid ${pid} died during warm-up." >&2; exit 1; }
done

echo "Probing Ray GCS at ${MASTER_ADDR}:${MASTER_PORT}..."
for attempt in $(seq 1 36); do
  if timeout 2 bash -c "</dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then
    echo "== GCS reachable after ${attempt} attempt(s) =="; break
  fi
  [[ "${attempt}" == "36" ]] && { echo "ERROR: Ray GCS never came up." >&2; exit 1; }
  sleep 5
done

########################################
# 4. Submit OPD training job
########################################
set +e
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  --container-image "${IMAGE}" --container-mounts "${MILES_MOUNTS}" --container-remap-root \
  --container-env=WANDB_API_KEY,NCCL_SOCKET_IFNAME,NCCL_SOCKET_FAMILY,NCCL_DEBUG,no_proxy \
  bash -lc "
    set -euxo pipefail
    cd ${MILES_ROOT}
    export RAY_TMPDIR=${RAY_TMP_ROOT}
    pip install -e . --no-deps --break-system-packages --quiet 2>&1 | tail -3

    source ${MODEL_SCRIPT}

    if [[ ! -f ${DATASETS_ROOT}/dapo-math-17k/dapo-math-17k.jsonl ]]; then
      hf download zhuzilin/dapo-math-17k --repo-type dataset --local-dir ${DATASETS_ROOT}/dapo-math-17k || true
    fi

    CKPT_ARGS=(
      --hf-checkpoint ${STUDENT_HF_ROOT}
      --ref-load ${STUDENT_REF_ROOT}
      --load ${OUTPUTS_ROOT}/${RUN_NAME}
      --save ${OUTPUTS_ROOT}/${RUN_NAME}
      --save-interval ${SAVE_INTERVAL}
    )
    ROLLOUT_ARGS=(
      --prompt-data ${DATASETS_ROOT}/dapo-math-17k/dapo-math-17k.jsonl
      --input-key prompt
      --apply-chat-template
      --rollout-shuffle
      --num-rollout ${NUM_ROLLOUT}
      --rollout-batch-size ${ROLLOUT_BATCH}
      --n-samples-per-prompt ${N_SAMPLES}
      --rollout-max-response-len ${MAX_RESPONSE_LEN}
      --rollout-temperature 1
      --global-batch-size ${GLOBAL_BATCH}
      --balance-data
    )
    GRPO_ARGS=(
      --advantage-estimator grpo
      --use-opd
      --opd-type sglang
      --opd-kl-coef ${OPD_KL_COEF}
      --use-kl-loss
      --kl-loss-coef 0.00
      --kl-loss-type low_var_kl
      --entropy-coef 0.00
    )
    RM_ARGS=(
      --custom-rm-path examples.on_policy_distillation.on_policy_distillation.reward_func
      --custom-reward-post-process-path examples.on_policy_distillation.on_policy_distillation.post_process_rewards
      --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
      ${CROSS_TOKENIZER_ARGS}
    )
    PERF_ARGS=(
      --tensor-model-parallel-size ${TP}
      --sequence-parallel
      --pipeline-model-parallel-size ${PP}
      --context-parallel-size 1
      --expert-model-parallel-size 1
      --expert-tensor-parallel-size 1
      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1
      --use-dynamic-batch-size
      --max-tokens-per-gpu 16384
    )
    OPTIMIZER_ARGS=(
      --optimizer adam
      --lr 1e-6
      --lr-decay-style constant
      --weight-decay 0.1
      --adam-beta1 0.9
      --adam-beta2 0.98
    )
    SGLANG_ARGS=(
      --rollout-num-gpus-per-engine 1
      --sglang-mem-fraction-static 0.4
    )
    MISC_ARGS=(
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash
    )
    WANDB_ARGS=(
      #--use-wandb
      # --wandb-project miles-dev
      # --wandb-group qwen3-8B-opd
      # --wandb-key \${WANDB_API_KEY}
    )

    ray job submit --address=http://${MASTER_ADDR}:8265 \
      --runtime-env-json='{\"env_vars\":{\"PYTHONPATH\":\"${MILES_ROOT}:/root/Megatron-LM/\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\",\"RAY_DEDUP_LOGS\":\"0\",\"NCCL_NVLS_ENABLE\":\"${HAS_NVLINK}\",\"NCCL_TIMEOUT\":\"3600\",\"no_proxy\":\"'\"\${no_proxy}\"'\"}}' \
      -- python3 train.py \
      --actor-num-nodes ${ACTOR_NODES} \
      --actor-num-gpus-per-node ${GPUS_PER_NODE} \
      --rollout-num-gpus ${ROLLOUT_GPUS} \
      \"\${MODEL_ARGS[@]}\" \"\${CKPT_ARGS[@]}\" \"\${ROLLOUT_ARGS[@]}\" \
      \"\${OPTIMIZER_ARGS[@]}\" \"\${GRPO_ARGS[@]}\" \"\${PERF_ARGS[@]}\" \
      \"\${SGLANG_ARGS[@]}\" \"\${MISC_ARGS[@]}\" \"\${WANDB_ARGS[@]}\" \"\${RM_ARGS[@]}\"
  " 2>&1 | tee ./logs/miles-opd-qwen3-${SLURM_JOB_ID}.log
train_exit_code=${PIPESTATUS[0]}
set -e

[[ "${train_exit_code}" -ne 0 ]] && { echo "Training exited with code ${train_exit_code}."; exit "${train_exit_code}"; }
echo "OPD training completed successfully."
