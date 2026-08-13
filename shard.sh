#!/usr/bin/env bash
# One sweep shard. Code and MNIST come from /vol/stage (already staged), so a
# shard uploads nothing and the container starts computing immediately.
#
# Env: SHARD, N_SHARDS, N_SEEDS, WORKERS, TAG
set -euo pipefail

: "${SHARD:?}" "${N_SHARDS:?}" "${N_SEEDS:?}" "${WORKERS:?}"
TAG="${TAG:-param-explore-v3}"

export WC_CODE_COMMIT=6ebe0b5
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p /work/code out
tar xzf /vol/stage/wc_code.tar.gz -C /work/code
tar xzf /vol/stage/mnist.tar.gz  -C /work/code
cd /work/code

# Bring in the two driver fixes made after the tarball was packed.
cp /work/run_modal_sweep.py . 2>/dev/null || true

echo "shard $SHARD/$N_SHARDS  seeds=$N_SEEDS  workers=$WORKERS  tag=$TAG"
python -c "import torch,numpy;print('torch',torch.__version__,'numpy',numpy.__version__)"

time python run_modal_sweep.py \
    --out-root /work/out \
    --bundle-root /vol/bundles \
    --n-seeds "$N_SEEDS" --shard "$SHARD" --n-shards "$N_SHARDS" \
    --workers "$WORKERS" --code-tag "$TAG"

# Only the small deliverables come home; the 10.6 MB/run bundles stay on /vol
# (relocated there by the driver's heartbeat, so /work/out holds just logs).
# Name them per-shard so 10 concurrent harvests cannot collide.
cd /work
cp out/param_explore_log.csv "out/shard_${SHARD}_log.csv"
[ -s out/failures.jsonl ] && cp out/failures.jsonl "out/shard_${SHARD}_failures.jsonl"
# Keep a shard log on the Volume too: harvest is best-effort, /vol is durable.
mkdir -p /vol/logs && cp "out/shard_${SHARD}_log.csv" /vol/logs/
rm -rf out/_w* out/worker_logs/../_w* 2>/dev/null || true

echo "=== /vol/bundles totals ==="
find /vol/bundles -name manifest.json | wc -l
du -sh /vol/bundles
