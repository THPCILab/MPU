#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RESULT_PREFIX="v27_4090_7gpu_single_mesh_stage_refine_seed43"
SHARD_DIR="results/${RESULT_PREFIX}_shards"
LOG_DIR="results/${RESULT_PREFIX}_logs"
CKPT_DIR="results/${RESULT_PREFIX}_ckpt"
WARMUP_CKPT="${CKPT_DIR}/shared_warmup.pt"
WARMUP_JSON="results/${RESULT_PREFIX}_warmup.json"
MERGED_JSON="results/${RESULT_PREFIX}_merged.json"

mkdir -p "${SHARD_DIR}" "${LOG_DIR}" "${CKPT_DIR}"
rm -f "${SHARD_DIR}"/*.json "${LOG_DIR}"/*.log "${MERGED_JSON}" "${WARMUP_JSON}" "${WARMUP_CKPT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY

python tools/download_emnist.py

COMMON_ARGS=(
  --preset main
  --task-set three
  --seed 43
  --input-preprocess pca
  --train-samples 40000
  --test-samples 12000
  --hidden-dim 64
  --num-layers 1
  --input-rank 64
  --epochs 80
  --pretrain-epochs 80
  --baseline-epochs 1
  --batch-size 512
  --base-lr 0.0035
  --electrical-lr-scale 1.0
  --private-delta-lr-scale 1.0
  --private-delta-weight-decay 0.0001
  --score-batches 20
  --score-source warmup
  --score-method fisher_damage
  --guided-node-prior stage_refine
  --guided-node-refine-fraction 0.35
  --guided-node-balance global
  --clements-layer-mode stage
  --warmup-shared-ratio 0
  --use-best-warmup-state
  --require-cuda
  --head-mode task_specific
  --photonic-readout per_mesh_intensity
  --norm-type layer
  --freeze-electrical-after-warmup
  --skip-electronic-baselines
  --skip-optical-unfrozen
  --efficiency off
  --save-mask-matrices
  --save-confusion-matrices
  --save-condition-checkpoints
  --track-train-accuracy
  --checkpoint-dir "${CKPT_DIR}"
  --warmup-checkpoint "${WARMUP_CKPT}"
)

echo "[warmup] seed=43 single 64x64 mesh -> ${WARMUP_CKPT}"
CUDA_VISIBLE_DEVICES=0 python -u train_mpu_pnn.py \
  "${COMMON_ARGS[@]}" \
  --output "${WARMUP_JSON}" \
  --ratios 0 \
  --modes node_pruning \
  --warmup-only \
  --no-resume \
  > "${LOG_DIR}/warmup.log" 2>&1

pids=()
labels=()
logs=()

run_shard() {
  local gpu="$1"
  local tag="$2"
  local mode_csv="$3"
  shift 3
  local ratios=("$@")
  IFS=',' read -r -a modes <<< "${mode_csv}"
  local ratio_tag
  ratio_tag="$(IFS=-; echo "${ratios[*]}")"
  local out="${SHARD_DIR}/${tag}_r${ratio_tag}.json"
  local log="${LOG_DIR}/${tag}_r${ratio_tag}.log"

  echo "[GPU ${gpu}] ${tag}: modes=${modes[*]} ratios=${ratios[*]} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u train_mpu_pnn.py \
    "${COMMON_ARGS[@]}" \
    --output "${out}" \
    --ratios "${ratios[@]}" \
    --modes "${modes[@]}" \
    --no-resume \
    > "${log}" 2>&1 &
  local pid=$!
  pids+=("${pid}")
  labels+=("gpu${gpu}_${tag}")
  logs+=("${log}")
  echo "  pid=${pid} log=${log}"
}

run_shard 0 boundary node_pruning,clements_layer 0 100
run_shard 1 node20   node_pruning 20
run_shard 2 node40   node_pruning 40
run_shard 3 node60   node_pruning 60
run_shard 4 node80   node_pruning 80
run_shard 5 layer_lo clements_layer 20 40
run_shard 6 layer_hi clements_layer 60 80

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  label="${labels[$i]}"
  log="${logs[$i]}"
  if wait "${pid}"; then
    echo "[OK] ${label}"
  else
    status=$?
    echo "[FAILED] ${label} exit=${status} log=${log}" >&2
    tail -n 120 "${log}" >&2 || true
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

shopt -s nullglob
shards=("${SHARD_DIR}"/*.json)

python tools/merge_shards.py \
  --inputs "${shards[@]}" \
  --output "${MERGED_JSON}"

python tools/plot_nature_figures.py \
  --input "${MERGED_JSON}" \
  --out-dir results/v27_nature_figures

python tools/export_figure_data.py \
  --input "${MERGED_JSON}" \
  --fig-dir results/v27_nature_figures \
  --out-dir paper_data

echo "Done. v27 4090 single-mesh JSON: ${MERGED_JSON}"
