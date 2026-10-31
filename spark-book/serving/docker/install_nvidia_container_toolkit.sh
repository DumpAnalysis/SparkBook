#!/usr/bin/env bash
# NVIDIA runtime/CDI only. Keep the existing driver, including Windows-mapped WSL driver.
set -euo pipefail
version=1.20.1-1
refresh_k3s=false
import_image=
while (($#)); do
 case "$1" in
  --refresh-k3s) refresh_k3s=true; shift ;;
  --version|--import-image)
   (($#>=2)) || exit 2
   case "$1" in --version) version=$2 ;; --import-image) import_image=$(realpath "$2") ;; esac
   shift 2 ;;
  --help) echo 'Usage: bash install_nvidia_container_toolkit.sh [--version 1.20.1-1] [--refresh-k3s] [--import-image EXISTING_TAR]'; exit 0 ;;
  *) echo "Unknown option $1"; exit 2 ;;
 esac
done
[[ $(id -u) != 0 ]] || { echo 'Run as your ordinary sudo-capable user'; exit 2; }
[[ $(ps -p 1 -o comm=) == systemd ]] || { echo 'Systemd required'; exit 2; }
if [[ -n ${WSL_DISTRO_NAME:-} ]]; then
 [[ -r /usr/lib/wsl/lib/libcuda.so.1 && -r /usr/lib/wsl/lib/libnvidia-ml.so.1 ]] || { echo 'Existing Windows-mapped WSL driver libraries required'; exit 2; }
fi
out="$PWD/runs/nvidia-container-install-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$(dirname "$out")"
mkdir "$out"
cp "${BASH_SOURCE[0]}" "$out/source.sh"
exec > >(tee "$out/stdout.txt") 2> >(tee "$out/stderr.txt" >&2)
started=$(date +%s)
trap 'rc=$?; printf "installer_exit=%s UTC=%s elapsed_seconds=%s\n" "$rc" "$(date -u +%FT%TZ)" "$(( $(date +%s) - started ))"; printf "%s\n" "$rc" > "$out/exit-code.txt"; exit "$rc"' EXIT
run() {
 local began rc
 began=$(date +%s)
 printf '\nUTC=%s CWD=%q COMMAND=' "$(date -u +%FT%TZ)" "$PWD"; printf '%q ' "$@"; printf '\n'
 if "$@"; then rc=0; else rc=$?; fi
 printf 'COMMAND_EXIT=%s UTC=%s ELAPSED_SECONDS=%s\n' "$rc" "$(date -u +%FT%TZ)" "$(( $(date +%s) - began ))"
 return "$rc"
}
run sha256sum "$out/source.sh"
run uname -a
run nvidia-smi -L
run sudo -v
# A Docker daemon restart must not interrupt running workloads.
running=$(sudo docker ps -q)
[[ -z "$running" ]] || { run sudo docker ps; echo 'Stop/schedule existing containers before this runtime-maintenance step.'; exit 2; }
run sudo apt-get update
run sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg
run curl --fail --location --proto '=https' --tlsv1.2 https://nvidia.github.io/libnvidia-container/gpgkey -o "$out/nvidia.asc"
run curl --fail --location --proto '=https' --tlsv1.2 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list -o "$out/repository-original.list"
mkdir -m 0700 "$out/public-key-check"
run gpg --batch --homedir "$out/public-key-check" --dearmor --output "$out/nvidia.gpg" "$out/nvidia.asc"
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' "$out/repository-original.list" > "$out/nvidia-container-toolkit.list"
run sha256sum "$out/nvidia.asc" "$out/nvidia.gpg" "$out/repository-original.list" "$out/nvidia-container-toolkit.list"
for pair in "nvidia.gpg:/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg" "nvidia-container-toolkit.list:/etc/apt/sources.list.d/nvidia-container-toolkit.list"; do
 file=${pair%%:*}; destination=${pair#*:}
 if [[ -e "$destination" ]]; then
  cmp -s "$out/$file" "$destination" || { echo "Existing $destination differs; preserved for review"; exit 2; }
 else
  run sudo install -d -m 0755 "$(dirname "$destination")"
  run sudo install -m 0644 "$out/$file" "$destination"
 fi
done
run sudo apt-get update
packages=("nvidia-container-toolkit=$version" "nvidia-container-toolkit-base=$version" "libnvidia-container-tools=$version" "libnvidia-container1=$version")
run apt-cache policy nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1
run sudo apt-get --simulate install --no-install-recommends "${packages[@]}" | tee "$out/package-plan.txt"
while read -r action package remainder; do
 if [[ "$action" == Inst ]]; then
  case "$package" in nvidia-driver*|cuda-drivers*|nvidia-dkms*|linux-modules-nvidia*) echo "Refusing display-driver dependency $package"; exit 2 ;; esac
 fi
done < "$out/package-plan.txt"
run sudo apt-get install -y --no-install-recommends "${packages[@]}"
run nvidia-ctk --version
run nvidia-container-runtime --version
run dpkg-query -W '-f=${Package}=${Version}\n' nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1
if [[ -e /etc/docker/daemon.json ]]; then
 backup="/etc/docker/daemon.json.spark-book-$(date -u +%Y%m%dT%H%M%SZ)-$$"
 run sudo cp -a /etc/docker/daemon.json "$backup"
 run sudo sha256sum "$backup"
fi
run sudo nvidia-ctk runtime configure --runtime=docker
run sudo dockerd --validate --config-file=/etc/docker/daemon.json
run sudo systemctl enable --now nvidia-cdi-refresh.path
if ! run sudo systemctl restart nvidia-cdi-refresh.service; then
 run sudo journalctl -u nvidia-cdi-refresh.service -n 80 --no-pager
 exit 2
fi
run nvidia-ctk --debug cdi list
run nvidia-ctk cdi list | tee "$out/cdi-devices.txt"
[[ $(cat "$out/cdi-devices.txt") == *nvidia.com/gpu=all* ]] || { echo "BLOCKED: no all-GPU CDI device generated"; exit 2; }
run sudo sha256sum /var/run/cdi/nvidia.yaml
running=$(sudo docker ps -q)
[[ -z "$running" ]] || { echo 'A container started during maintenance; Docker restart deferred'; exit 2; }
run sudo systemctl restart docker
run sudo docker info --format '{{json .Runtimes}}'
if "$refresh_k3s"; then
 run sudo k3s kubectl get pods -A
 run sudo systemctl restart k3s
 # K3s detects nvidia-container-runtime itself; do not edit generated containerd config.
 api_ready=false
 for attempt in {1..60}; do
  if sudo k3s kubectl get nodes --request-timeout=5s -o name > "$out/nodes-after-restart.txt" && [[ -s "$out/nodes-after-restart.txt" ]]; then api_ready=true; break; fi
  sleep 2
 done
 "$api_ready" || { echo "BLOCKED: K3s API/nodes unavailable after restart"; exit 2; }
 run sudo k3s kubectl wait --for=condition=Ready node --all --timeout=180s
 run sudo k3s kubectl get runtimeclass
fi
if [[ -n "$import_image" ]]; then
 run sha256sum "$import_image"
 run sudo k3s ctr images import "$import_image"
fi
printf '\nRuntime/CDI configuration completed. Actual container CUDA computation and Kubernetes GPU discovery still require qualification.\n'
