#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux ]]; then echo 'Run this script in the Ubuntu VM.'; exit 1; fi
export RADNOTE_UID="${SUDO_UID:-$(id -u)}"
export RADNOTE_GID="${SUDO_GID:-$(id -g)}"
export RADNOTE_MODEL_PRECISION=int8
if docker info >/dev/null 2>&1; then
  docker_command=(docker)
else
  docker_command=(sudo env "RADNOTE_UID=$RADNOTE_UID" "RADNOTE_GID=$RADNOTE_GID" "RADNOTE_MODEL_PRECISION=int8" docker)
fi
"${docker_command[@]}" info >/dev/null
mkdir -p results/vm
if [[ "$(id -u)" == 0 ]]; then chown "$RADNOTE_UID:$RADNOTE_GID" results results/vm; fi
python3 scripts/preflight.py
"${docker_command[@]}" compose config >results/vm/compose_resolved.yml
{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  uname -a
  cat /etc/os-release
  lscpu
  free -h
  df -h .
  systemd-detect-virt || true
  "${docker_command[@]}" version
  "${docker_command[@]}" compose version
} >results/vm/hardware_and_versions.txt
echo 'Build the backend and lightweight client images.'
"${docker_command[@]}" compose build backend client 2>&1 | tee results/vm/docker_build.log
"${docker_command[@]}" image inspect radnote-ai-backend:3.0 radnote-ai-client:3.0 >results/vm/docker_images.json
echo 'Run offline precision and direct-pipeline measurements with the regular service stopped.'
"${docker_command[@]}" compose stop client backend
"${docker_command[@]}" compose --profile tools run --rm --no-deps benchmark python scripts/benchmark_precision.py --output results/vm/precision_comparison.json
"${docker_command[@]}" compose --profile tools run --rm --no-deps benchmark python scripts/benchmark_execution.py --mode direct --output results/vm/local_execution.json
echo 'Start the Dockerized service and client; wait for readiness.'
"${docker_command[@]}" compose up -d --wait --wait-timeout 240 backend client
"${docker_command[@]}" compose ps >results/vm/compose_status.txt
curl --fail --silent http://127.0.0.1:8000/health >results/vm/backend_health.json
curl --fail --silent http://127.0.0.1:8765/ready >results/vm/client_ready.json
curl --fail --silent http://127.0.0.1:8000/metrics >results/vm/backend_metrics_before.json
stats_pid=''
stop_stats() { if [[ -n "$stats_pid" ]]; then kill "$stats_pid" 2>/dev/null || true; wait "$stats_pid" 2>/dev/null || true; fi; }
trap stop_stats EXIT
(
  while true; do
    printf '{"timestamp_utc":"%s"}\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    "${docker_command[@]}" stats --no-stream --format '{{json .}}' $("${docker_command[@]}" compose ps -q backend client)
    sleep 2
  done
) >results/vm/docker_stats.jsonl &
stats_pid=$!
echo 'Verify direct API and proxy uploads, then measure HTTP client-server execution.'
"${docker_command[@]}" compose --profile tools run --rm --no-deps benchmark python scripts/verify_project.py --server http://backend:8000 --proxy http://client:8765 --output results/vm/integration.json
"${docker_command[@]}" compose --profile tools run --rm --no-deps benchmark python scripts/benchmark_execution.py --mode http --server http://client:8765 --output results/vm/client_server_execution.json
echo 'Benchmark all 60 synthetic pages and two sustained load windows.'
"${docker_command[@]}" compose --profile tools run --rm --no-deps benchmark python scripts/benchmark_api.py --server http://client:8765 --output results/vm
"${docker_command[@]}" compose --profile tools run --rm --no-deps benchmark python scripts/benchmark_load.py --server http://client:8765 --seconds 60 --output results/vm/load_sustained.json
curl --fail --silent http://127.0.0.1:8000/metrics >results/vm/backend_metrics_after.json
"${docker_command[@]}" compose logs --no-color backend client >results/vm/service_logs.txt
stop_stats
stats_pid=''
python3 scripts/summarize_vm_results.py --directory results/vm
echo 'VM measurements are saved in results/vm. Open http://127.0.0.1:8765 in the Ubuntu browser for the final live demonstration.'
