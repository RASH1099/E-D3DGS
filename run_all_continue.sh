#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="/home/jiangzhenghan/project/code/E-D3DGS_original"
SCRIPT_DIR="${BASE_DIR}"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "$LOG_DIR"

SCRIPTS=(
  "MeetingRoom_LightTheCandles.sh"
)

stamp="$(date '+%Y%m%d_%H%M%S')"
SUMMARY_CSV="${LOG_DIR}/summary_${stamp}.csv"
MASTER_LOG="${LOG_DIR}/all_${stamp}.log"

echo "script,start_time,end_time,exit_code,log_file" > "$SUMMARY_CSV"

{
  echo "========== RUN START: $(date '+%F %T') =========="
  echo "BASE_DIR  : $BASE_DIR"
  echo "SCRIPT_DIR: $SCRIPT_DIR"
  echo "LOG_DIR   : $LOG_DIR"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  echo "Scripts:"
  printf '  - %s\n' "${SCRIPTS[@]}"
  echo "================================================="
} | tee -a "$MASTER_LOG"

cd "$SCRIPT_DIR"

for s in "${SCRIPTS[@]}"; do
  start_time="$(date '+%F %T')"
  script_path="${SCRIPT_DIR}/${s}"

  if [[ ! -f "$script_path" ]]; then
    echo "[MISSING] $script_path" | tee -a "$MASTER_LOG"
    echo "${s},${start_time},$(date '+%F %T'),127,NA" >> "$SUMMARY_CSV"
    continue
  fi

  chmod +x "$script_path" || true
  log_file="${LOG_DIR}/${s%.sh}_${stamp}.log"

  echo | tee -a "$MASTER_LOG"
  echo "===== START ${s} @ ${start_time} =====" | tee -a "$MASTER_LOG"
  echo "Script: $script_path" | tee -a "$MASTER_LOG"
  echo "Log   : $log_file" | tee -a "$MASTER_LOG"

  bash "$script_path" 2>&1 | tee -a "$log_file"
  exit_code="${PIPESTATUS[0]}"

  end_time="$(date '+%F %T')"
  echo "===== END   ${s} @ ${end_time} | exit=${exit_code} =====" | tee -a "$MASTER_LOG"

  echo "${s},${start_time},${end_time},${exit_code},${log_file}" >> "$SUMMARY_CSV"
done

echo | tee -a "$MASTER_LOG"
echo "========== RUN END: $(date '+%F %T') ==========" | tee -a "$MASTER_LOG"
echo "Summary CSV : $SUMMARY_CSV" | tee -a "$MASTER_LOG"
echo "Master log  : $MASTER_LOG" | tee -a "$MASTER_LOG"
