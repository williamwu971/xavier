#!/bin/sh
set -eu
cd "$(dirname "$0")"
make memory_cost
# Each size runs CPU-private, CPU-shared, GPU-private, GPU-shared.
# Set REPEATS=3 to alternate allocation order and assess run-to-run noise.
for size in 0.0625 64; do
    ./memory_cost "$size" "${SECONDS_PER_CASE:-10}" "${CPU_THREADS:-4}" \
        "${PATTERN:-sequential}" "${REPEATS:-1}"
done
