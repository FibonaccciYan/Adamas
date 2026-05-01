#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-run}"
MODEL="${MODEL:-Llama-3.1-8B-Instruct}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${ROOT_DIR}/test_results/${MODEL}"
ARCHIVE_ROOT="${ROOT_DIR}/test_results/e2e_runs"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${STAMP}_${LABEL}"

mkdir -p "${RESULT_DIR}" "${ARCHIVE_DIR}"

if [ -f "${RESULT_DIR}/e2e_Adamas.txt" ]; then
    cp "${RESULT_DIR}/e2e_Adamas.txt" "${ARCHIVE_DIR}/e2e_Adamas.before.txt"
fi

: > "${RESULT_DIR}/e2e_Adamas.txt"

(
    cd "${ROOT_DIR}"
    bash scripts/bench_efficiency_e2e.sh
)

cp "${RESULT_DIR}/e2e_Adamas.txt" "${ARCHIVE_DIR}/e2e_Adamas.txt"
cp "${RESULT_DIR}"/log_*.log "${ARCHIVE_DIR}/" 2>/dev/null || true
git -C "${ROOT_DIR}" status --short > "${ARCHIVE_DIR}/git_status.txt"
git -C "${ROOT_DIR}" diff > "${ARCHIVE_DIR}/git_diff.patch"

echo "${ARCHIVE_DIR}"
