# the recorded run installation and serving qualification

Version 38 editorial copy of the supplied INSTALL_LOCAL.md; the original
bytes remain in the supporting execution records
Two historical-draft sentences below are clarified to avoid treating an early
BLOCKED handoff as the final result. No captured raw record is changed.

Scope: Docker 29.9.2, K3s v1.37.0+k3s1 and Toolkit 1.20.1-1 were actually
installed by the owner. The Toolkit installer ran successfully. The combined
Docker/K3s installer was syntax/check-only validated against that installation;
its fresh-VM install path is UNEXECUTED. Review repository trust, administrator
access and daemon-restart effects before running maintenance on your own system.
The exported private kubeconfig is intentionally absent.

The generic companion wrappers are retained for teaching. For this exact
vLLM qualification, inspect the final recipe, YAML and source differences in
the supporting execution records/, including artifacts/vllm-unblock-recipe.json.
Do not assume the earlier wrapper already contains those adaptations.

# Reproduce native Docker and K3s installation

Run `install_docker_k3s.sh` from this directory in an Ubuntu systemd VM or WSL2
session as an ordinary sudo-capable user. The script supports Ubuntu22.04,24.04,
26.04 and26.10 on arm64/amd64. WSL must already have systemd enabled. It does not
change WSL settings or install a Linux NVIDIA display driver.

```bash
cd ~/work/spark-book
bash serving/docker/install_docker_k3s.sh --enable-user-access
```

`--enable-user-access` is an explicit choice: Docker-group membership grants
root-equivalent daemon access; the private `~/.kube/spark-book-k3s.yaml` grants
administrator access to this local lab cluster. Omit the flag for sudo-only use.
The script uses `sudo -u` to verify refreshed groups and needs neither `rg`, `sg`
nor `newgrp`. Your existing shell may still need a new login for Docker access.

K3s is pinned to `v1.37.0+k3s1`. Docker's repository candidate is resolved and
installed as an exact apt version, then all installed package versions and image
digests are recorded. To reproduce this recorded installation’s Docker CE/CLI on Ubuntu26.04:

```bash
bash serving/docker/install_docker_k3s.sh --enable-user-access \
  --docker-version '5:29.9.2-1~ubuntu.26.04~resolute'
```

That apt version is distribution-specific. Containerd, Buildx and Compose versions
are recorded from the repository; this is not a complete historical apt mirror.
An existing matching installation is reused. Different K3s versions, conflicting
packages or different repository configuration stop for review. Existing model
files, containers, cluster configuration and unrelated kubeconfigs are preserved.
K3s uses its own containerd image store, separate from Docker's.

Every attempt gets a new `runs/container-install-*` directory, with source hash,
exact command arguments, cwd, UTC times, elapsed seconds, stdout/stderr and exit
status. Public signing-key and installer bytes/hashes are retained. Passwords are
entered locally through sudo; kubeconfig and tokens are never printed or copied
into these logs. The script waits for node registration before readiness. Initial
bootstrap can still fail due to networking, storage or host capabilities; retain
the actual failure instead of treating installation as qualification.

Verify an existing setup without system changes, from a session with Docker access:

```bash
bash serving/docker/install_docker_k3s.sh --check-only
KUBECONFIG="$HOME/.kube/spark-book-k3s.yaml" kubectl get nodes
```

The recorded investigation records the original installation, failed helper/readiness attempts and
fixes. Check-only validation of this consolidated script is separate from a fresh
VM installation; a clean-VM rerun has not yet been performed. GPU runtime/toolkit,
Kubernetes device-plugin discovery, ARM64 serving images and actual GPU inference
remain separate exercise qualifications. No Docker/K3s installer alone proves them.

Sources: [Docker Ubuntu installation](https://docs.docker.com/engine/install/ubuntu/),
[K3s quick start](https://docs.k3s.io/quick-start),
[K3s requirements](https://docs.k3s.io/installation/requirements).

## Optional NVIDIA runtime/CDI setup

After Docker works, install the pinned Container Toolkit (not a display driver):

```bash
bash serving/docker/install_nvidia_container_toolkit.sh --refresh-k3s
```

It records the apt plan, refuses display-driver dependencies, preserves existing
Docker configuration, enables the CDI refresh service and restarts Docker and
optionally K3s. It refuses to restart Docker while containers are running. Use a
maintenance window. An optional `--import-image /absolute/path/image.tar` imports
an already built CPU image into K3s's separate containerd store and records its
hash. Toolkit1.20.1-1 is the initial pin; package availability and WSL device
support must be measured. Runtime configuration is not a passing GPU computation.
Kubernetes GPU advertising additionally requires a qualified NVIDIA device plugin.

Sources: [Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[CDI support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).
The generic native-Linux driver installation step in the vendor guide is omitted
for this existing Windows-mapped WSL driver.

The recorded investigation observed results: Docker29.9.2/ARM64, K3sv1.37.0+k3s1 and Toolkit1.20.1-1
were actually installed. CDI injected the existing Windows-mapped driver; a CUDA
addition over1048576 elements inside Docker passed its numerical check. The
Toolkit installer completed successfully with `--refresh-k3s --import-image`.
These results qualify this recorded VM, not every installation of those versions.

For Exercise146, project-local Helm4.3.1 installed NVIDIA chart0.20.1. The chart's
default PCI/GFD affinity initially selected zero WSL nodes. the recorded investigation retained that
attempt, removed that affinity gate and restricted the plugin to the hostname
already verified by actual GPU computation. The plugin then registered and the
node advertised one `nvidia.com/gpu`; the first immediate read preceded resource
propagation. `failOnInitError=true` remained enabled. The exact values and chart
bytes are in the supporting execution records. On another VM, qualify the selected node rather
than copying this machine's hostname or declaring fake GPU capacity.

The recorded investigation also found two vLLM0.26.0 startup constraints with the teaching container
settings. UID10001 has no passwd entry in that image. Set `USER=sparkbook` and
`TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor` alongside the existing writable caches;
this retains the same numeric UID and read-only filesystem. The corrected import
was actually checked as UID10001. Separately, the image's default GPU memory
utilization0.92 requested42.29GiB while the process reported7.49GiB free. The
recorded diagnostic retry uses `--gpu-memory-utilization 0.12` with the original
1GiB KV-cache cap. This is a measured local budget adjustment, not a universal VM
setting or an increase in physical memory. Final model qualification is recorded
in Exercise148; do not infer it merely from these setup fixes. Original
errors, exact command changes and the isolated deployment-source patches are in
the evidence export; baseline execution sources were preserved in the evidence workspace; Version 38 now integrates the final results into the manuscript.

The next retained failure was the V2 runner's default WSL pinned-memory policy:
`UVA is not available`. The installed release exposes
`VLLM_WSL2_ENABLE_PIN_MEMORY=1`. Before a model retry, the recorded investigation actually checked
4096 values using the release's pinned CPU/UVA buffer and GPU arithmetic, with
maximum error0 and checksum25163776. The isolated deployment and container
commands record this opt-in. This qualifies that small operation on this system;
UVA (Unified Virtual Addressing) does not create additional physical RAM.

vLLM's native JIT helpers also need an executable cache filesystem. Docker's
`--tmpfs /tmp:rw,size=1g,mode=1777` produced a measured `noexec` mount and failed
shared-object loading. A tiny compiled function reproduced that error and passed
with `--tmpfs /tmp:rw,exec,nosuid,nodev,size=1g,mode=1777`. the recorded investigation retains this
control and uses the corrected mount for the subsequent model attempt. The
numeric UID, read-only root filesystem and dropped capabilities remain unchanged.

The initial recorded handoff draft recorded vLLM as **BLOCKED**: after the tested setup
corrections and model loading, Triton's helper invoked `/sbin/ldconfig -p`, which
terminated with SIGSEGV11 before readiness. A separate GPU-enabled probe passed
that query, while a CPU-only probe also crashed; the root cause was not established.
At that point the explicit Triton driver-library-path option had not yet been
qualified. The requested continuation and its successful Docker qualification
are recorded below. That earlier draft had no vLLM generation/concurrency or Kubernetes success claim; the final continuation below supersedes it. These failures
do not invalidate the separate GPU arithmetic, llama.cpp Docker/Kubernetes and
CPU Triton backend checks. Preserve the exact image digest and all failed captures
when investigating further.

The user requested an additional unblock attempt before completing the recorded investigation. The
image's Triton implementation accepts `TRITON_LIBCUDA_PATH`. The included
`vllm_wsl_entrypoint.py` discovers the actually injected WSL driver, verifies its
API, records its SHA-256, and creates temporary `libcuda.so`/`libcuda.so.1` aliases
under `/tmp/spark-cuda-driver`. It then executes the requested command with that
explicit path. It neither replaces driver bytes nor edits the host linker cache.
A real Triton3.6.0 JIT kernel using this path passed4096 values, maximum error0,
checksum8398848. Model and Kubernetes qualification are reported by the latest
Exercise148/149 results, superseding the earlier draft status when applicable.

The continuation successfully brought the Docker vLLM owner to readiness and
completed a streaming response plus32 requests at each of concurrency1 and2;
all64 non-streaming requests returned HTTP200 with consistent token usage.
`artifacts/vllm-unblock-recipe.json` in the recorded investigation records the exact successful argv,
image digest, settings and bootstrap hash. Reuse the existing verified Qwen1.5B
model; these retries did not download another model. The final Kubernetes result
is independent and is recorded under Exercise149.

For a fresh lab namespace, create it before server-side dry-running namespaced
objects. A dry-run Namespace document is not persisted for the other documents
in the same file. the recorded investigation preserves the initial namespace-not-found dry-run and
the subsequent successful server validation after actual namespace creation.

Latest continuation result: **Exercises148 and149 PASSED**. The configured vLLM
Docker owner passed streaming and64 total load requests; server counters matched
64 completions and2112 generated tokens. The Kubernetes owner also became ready
and completed a streaming response. Its original900-second rollout took891.916
seconds including image acquisition and startup; the prepared extended retry was
never executed. Keep the earlier failures for diagnosis and use the latest recipe
and manifests when reproducing this particular ARM64 WSL configuration.
