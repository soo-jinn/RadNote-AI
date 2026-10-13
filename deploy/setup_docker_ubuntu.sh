#!/usr/bin/env bash
set -euo pipefail
root_command() { if [[ "$(id -u)" == 0 ]]; then "$@"; else sudo "$@"; fi; }
if [[ ! -f /etc/os-release ]]; then echo 'Run this setup inside your Ubuntu VM.'; exit 1; fi
source /etc/os-release
if [[ "$ID" != ubuntu ]]; then echo 'This helper supports Ubuntu only.'; exit 1; fi
if [[ "$(uname -m)" != x86_64 ]]; then echo 'The supplied CPU dependency lock targets the amd64 course VM.'; exit 1; fi
if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then
  root_command systemctl start docker
  echo 'Existing Docker Engine and Compose plugin detected.'
  exit 0
fi
if dpkg-query -W -f='${Status}' docker.io 2>/dev/null | grep -q 'install ok installed'; then
  echo 'Ubuntu docker.io is already installed. Ask the lab administrator to provide its compatible Compose plugin or migrate it using the official Docker Ubuntu instructions. This helper will not replace existing lab packages.'
  exit 1
fi
root_command apt-get update
root_command apt-get install -y ca-certificates curl python3 unzip
root_command install -m 0755 -d /etc/apt/keyrings
root_command curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
root_command chmod a+r /etc/apt/keyrings/docker.asc
codename="${UBUNTU_CODENAME:-$VERSION_CODENAME}"
architecture="$(dpkg --print-architecture)"
printf 'Types: deb\nURIs: https://download.docker.com/linux/ubuntu\nSuites: %s\nComponents: stable\nArchitectures: %s\nSigned-By: /etc/apt/keyrings/docker.asc\n' "$codename" "$architecture" | root_command tee /etc/apt/sources.list.d/docker.sources >/dev/null
root_command apt-get update
root_command apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
root_command systemctl enable --now docker
root_command docker run --rm hello-world
docker compose version
