#!/usr/bin/env bash
# Native Docker Engine + one-node K3s for an Ubuntu systemd VM/WSL2.
# No display driver installation, WSL settings change, or credential export.
set -euo pipefail
k3s_version=v1.37.0+k3s1
docker_version=
enable_access=false
check_only=false
log_dir=
usage() {
 cat <<'EOF'
Usage: bash install_docker_k3s.sh [options]
  --enable-user-access       Explicitly grant this user Docker-group access and
                            a private administrator kubeconfig for this lab.
  --docker-version VERSION  Exact apt version for docker-ce and docker-ce-cli.
                            Default: resolve the repository candidate and record it.
  --k3s-version VERSION     Default: v1.37.0+k3s1; existing different version stops.
  --log-dir NEW_DIRECTORY   Default: ./runs/container-install-UTC-PID
  --check-only              Verify existing installation; no sudo/setup changes.
  --help                    Show this help.
Run as an ordinary Ubuntu user with sudo access. Systemd must already be running.
Docker-group access is root-equivalent. Kubeconfig gives local cluster admin access.
EOF
}
while (($#)); do
 case "$1" in
  --enable-user-access) enable_access=true; shift ;;
  --check-only) check_only=true; shift ;;
  --docker-version|--k3s-version|--log-dir)
   (($# >= 2)) || { usage; exit 2; }
   case "$1" in --docker-version) docker_version=$2 ;; --k3s-version) k3s_version=$2 ;; --log-dir) log_dir=$2 ;; esac
   shift 2 ;;
  --help|-h) usage; exit 0 ;;
  *) printf 'Unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
 esac
done
[[ $(id -u) != 0 ]] || { echo 'Run as your ordinary sudo-capable Ubuntu user.' >&2; exit 2; }
[[ "$k3s_version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+\+k3s[0-9]+$ ]] || { echo 'Invalid K3s version'; exit 2; }
install_user=$(id -un)
install_uid=$(id -u)
install_gid=$(id -g)
install_home=$(getent passwd "$install_user" | cut -d: -f6)
[[ -n "$install_home" && -d "$install_home" ]] || { echo 'Cannot resolve user home'; exit 2; }
log_dir=${log_dir:-"$PWD/runs/container-install-$(date -u +%Y%m%dT%H%M%SZ)-$$"}
mkdir -p -- "$(dirname -- "$log_dir")"
mkdir -- "$log_dir"  # refuse to overwrite an earlier attempt
log_dir=$(realpath -- "$log_dir")
cp -- "${BASH_SOURCE[0]}" "$log_dir/source.sh"
exec > >(tee "$log_dir/stdout.txt") 2> >(tee "$log_dir/stderr.txt" >&2)
start_epoch=$(date +%s)
finish() {
 rc=$?
 trap - EXIT
 printf 'installer_exit=%s utc=%s elapsed_seconds=%s\n' "$rc" "$(date -u +%FT%TZ)" "$(( $(date +%s) - start_epoch ))"
 printf '%s\n' "$rc" > "$log_dir/exit-code.txt"
 exit "$rc"
}
trap finish EXIT
run() {
 local began rc
 began=$(date +%s)
 printf '\nUTC=%s CWD=%q COMMAND=' "$(date -u +%FT%TZ)" "$PWD"
 printf '%q ' "$@"; printf '\n'
 if "$@"; then rc=0; else rc=$?; fi
 printf 'COMMAND_EXIT=%s UTC=%s ELAPSED_SECONDS=%s\n' "$rc" "$(date -u +%FT%TZ)" "$(( $(date +%s) - began ))"
 return "$rc"
}
run sha256sum "$log_dir/source.sh"
run pwd
run uname -a
printf 'WSL_DISTRO_NAME=%s\n' "${WSL_DISTRO_NAME:-not-a-WSL-session}"
run cat /etc/os-release
. /etc/os-release
[[ "$ID" == ubuntu ]] || { echo 'This installer targets Ubuntu only.'; exit 2; }
case "$VERSION_ID" in 22.04|24.04|26.04|26.10) ;; *) echo 'Ubuntu release requires separate Docker repository qualification.'; exit 2 ;; esac
architecture=$(dpkg --print-architecture)
case "$architecture" in arm64|amd64) ;; *) echo 'This lab installer qualifies arm64/amd64 only.'; exit 2 ;; esac
[[ $(ps -p 1 -o comm=) == systemd ]] || { echo 'Systemd must already be PID1; this script does not change WSL settings.'; exit 2; }
run systemctl is-system-running || run systemctl --failed --no-pager
private_config="$install_home/.kube/spark-book-k3s.yaml"
if ! "$check_only"; then
 run sudo -v
 # Refuse conflicting distribution packages rather than remove existing work.
 if ! command -v docker >/dev/null; then
  for package in docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc; do
   status=$(dpkg-query -W -f='${Status}' "$package" 2>/dev/null || true)
   [[ "$status" != 'install ok installed' ]] || { echo "Conflicting installed package $package: review manually before replacing it."; exit 2; }
  done
 fi
 run sudo apt-get update
 run sudo apt-get install -y ca-certificates curl gnupg
 run curl --fail --location --proto '=https' --tlsv1.2 https://download.docker.com/linux/ubuntu/gpg -o "$log_dir/docker.asc"
 mkdir -m 0700 "$log_dir/public-key-check"
 # Public key verification only; never use or export the user's own keyring.
 gpg --batch --homedir "$log_dir/public-key-check" --show-keys --with-colons "$log_dir/docker.asc" > "$log_dir/docker-key.txt"
 fingerprint=$(awk -F: '$1=="fpr" {print $10; exit}' "$log_dir/docker-key.txt")
 [[ "$fingerprint" == 9DC858229FC7DD38854AE2D88D81803C0EBFCD88 ]] || { echo 'Docker signing-key fingerprint mismatch; stop for review.'; exit 2; }
 run sha256sum "$log_dir/docker.asc"
 suite=${UBUNTU_CODENAME:-$VERSION_CODENAME}
 printf 'Types: deb\nURIs: https://download.docker.com/linux/ubuntu\nSuites: %s\nComponents: stable\nArchitectures: %s\nSigned-By: /etc/apt/keyrings/docker.asc\n' "$suite" "$architecture" > "$log_dir/docker.sources"
 # Do not silently replace another administrator's repository/key configuration.
 if [[ -e /etc/apt/sources.list.d/docker.list ]]; then echo 'Existing docker.list: review/reconcile repository configuration first.'; exit 2; fi
 for pair in 'docker.sources:/etc/apt/sources.list.d/docker.sources' 'docker.asc:/etc/apt/keyrings/docker.asc'; do
  generated=${pair%%:*}; installed=${pair#*:}
  if [[ -e "$installed" ]]; then
   cmp -s "$log_dir/$generated" "$installed" || { echo "Existing $installed differs; preserved for manual review."; exit 2; }
  else
   run sudo install -d -m 0755 "$(dirname -- "$installed")"
   run sudo install -m 0644 "$log_dir/$generated" "$installed"
  fi
 done
 run sudo apt-get update
 run apt-cache policy docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
 if command -v docker >/dev/null; then
  installed_version=$(dpkg-query -W -f='${Version}' docker-ce 2>/dev/null || true)
  [[ -n "$installed_version" ]] || { echo 'Docker exists but is not the expected native docker-ce package; inspect active context first.'; exit 2; }
  [[ -z "$docker_version" || "$docker_version" == "$installed_version" ]] || { echo 'Existing Docker version differs; no automatic upgrade/downgrade.'; exit 2; }
  echo "REUSE Docker $installed_version"
 else
  if [[ -z "$docker_version" ]]; then docker_version=$(apt-cache policy docker-ce | awk '/Candidate:/ {print $2; exit}'); fi
  [[ -n "$docker_version" && "$docker_version" != '(none)' ]] || { echo 'No Docker candidate for this release/architecture'; exit 2; }
  run sudo apt-get install -y "docker-ce=$docker_version" "docker-ce-cli=$docker_version" containerd.io docker-buildx-plugin docker-compose-plugin
 fi
 run sudo systemctl enable --now docker
 run sudo docker version
 run sudo docker compose version
 run sudo docker run --rm hello-world
 run sudo docker image inspect hello-world --format '{{json .RepoDigests}}'
 run dpkg-query -W '-f=${Package}=${Version}\n' docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
 run sudo install -d -m 0755 /usr/local/bin
 if command -v k3s >/dev/null; then
  actual_k3s=$(k3s --version | head -1 | awk '{print $3}')
  [[ "$actual_k3s" == "$k3s_version" ]] || { echo "Existing K3s $actual_k3s differs; no automatic cluster upgrade."; exit 2; }
  echo "REUSE K3s $actual_k3s; preserve service/configuration"
  run sudo systemctl start k3s
 else
  [[ ! -e /etc/rancher/k3s/config.yaml && ! -e /etc/systemd/system/k3s.service ]] || { echo 'Existing K3s configuration found: inspect before installation.'; exit 2; }
  # Versioned official installer, retained before execution; binary checksum checked by installer.
  run curl --fail --location --proto '=https' --tlsv1.2 "https://raw.githubusercontent.com/k3s-io/k3s/$k3s_version/install.sh" -o "$log_dir/install-k3s.sh"
  run sha256sum "$log_dir/install-k3s.sh"
  run sudo env "INSTALL_K3S_VERSION=$k3s_version" sh "$log_dir/install-k3s.sh" server --disable traefik --disable servicelb --write-kubeconfig-mode 600
 fi
 if "$enable_access"; then
  run sudo usermod -aG docker "$install_user"
  run mkdir -p "$install_home/.kube"
  run chmod 700 "$install_home/.kube"
  if [[ -e "$private_config" ]]; then
   # A renewed local kubeconfig is preserved privately before refresh, never in logs.
   if ! sudo cmp -s /etc/rancher/k3s/k3s.yaml "$private_config"; then
    run cp -p "$private_config" "$private_config.backup-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    run sudo install -m 0600 -o "$install_uid" -g "$install_gid" /etc/rancher/k3s/k3s.yaml "$private_config"
   fi
  else
   run sudo install -m 0600 -o "$install_uid" -g "$install_gid" /etc/rancher/k3s/k3s.yaml "$private_config"
  fi
  # sudo starts a process with current group membership; no sg/newgrp dependency.
  run sudo -u "$install_user" -- docker info --format '{{.ServerVersion}} {{.Architecture}}'
  kube=(env "KUBECONFIG=$private_config" kubectl)
 else
  kube=(sudo k3s kubectl)
 fi
else
 run docker version
 run docker compose version
 run docker info --format '{{.ServerVersion}} {{.Architecture}}'
 [[ -r "$private_config" ]] || { echo 'Check-only needs the readable private lab kubeconfig; use setup with --enable-user-access or verify through sudo manually.'; exit 2; }
 kube=(env "KUBECONFIG=$private_config" kubectl)
fi
run k3s --version
# First wait for registration; kubectl wait returns immediately if no node exists yet.
registered=false
for attempt in {1..60}; do
 if "${kube[@]}" get nodes --request-timeout=5s -o name > "$log_dir/nodes-current.txt"; then
  if [[ -s "$log_dir/nodes-current.txt" ]]; then registered=true; break; fi
 fi
 printf 'UTC=%s registration_attempt=%s no-node-yet\n' "$(date -u +%FT%TZ)" "$attempt"
 sleep 2
done
if ! "$registered"; then
 run "${kube[@]}" get events -A --sort-by=.metadata.creationTimestamp || true
 echo 'BLOCKED: no node registered within bounded polling window'; exit 2
fi
run "${kube[@]}" wait --for=condition=Ready node --all --timeout=180s
run "${kube[@]}" get nodes -o wide
run "${kube[@]}" get pods -A
printf '\nPASS Docker and K3s installation checks. Logs: %s\n' "$log_dir"
printf 'GPU Container Toolkit, device-plugin support and actual GPU inference require separate qualification.\n'
