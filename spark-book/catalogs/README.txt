Recorded source paths in this note resolve under the separately restored evidence root.
See reader/EVIDENCE.txt for path lookup.

CURRENT READER EDITION — VERSION 55
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

NGC AND HUGGING FACE CATALOG LABORATORIES
Version 19, Chapters 89-90, Exercises 204-211

Read TARGET_LABS.txt for the full chapter procedures, interpretation and
complete transcripts. All commands run from the spark-book directory.
Use new output directories for each attempt. The supplied controls/ tree
is retained evidence; do not overwrite it during a repeat.

What was actually executed on 28 October 2026
  Public NGC resource metadata and verified ARM64 CLI archive acquisition.
  ELF architecture inspection without executing the ARM64 binary.
  Public NVCR index, ARM64 manifest and image configuration acquisition.
  Three in-memory manifest fault controls.
  Pinned public Hugging Face model and dataset metadata and downloads.
  Offline CPU inference with tiny-random-bert, including file verification,
  an empty loader cache, rejecting Python socket.connect and repeat output.
  CSV schema/split audit and one intentionally injected overlap.
  GPU probe reporting UNAVAILABLE with exit 2 in the CPU environment.
  Fifteen offline regression checks, all passed.

Eleven complete command captures are retained in controls/01-* through
controls/11-*. command.json records argv, start/end timestamps and exit.
transcript.txt retains merged stdout/stderr and the process return code.
controls/environment.json identifies the preparation runtime and scope.
No Docker, WSL2, native ARM64 CLI execution or CUDA success was executed.
Image layers were not downloaded. Earlier laboratory suites were not rerun.

Quick inspection without network, tokens or model downloads
  python3 catalogs/validate.py
  cat catalogs/controls/ngc-image/image.json
  cat catalogs/controls/hf-offline/result.json
  cat catalogs/controls/hf-audit/audit.json

New public metadata acquisition (standard library only)
  mkdir -p runs/catalogs downloads/catalogs
  python3 catalogs/capture.py --out runs/catalogs/204-log -- python3 catalogs/ngc_lab.py info --out runs/catalogs/204
  python3 catalogs/capture.py --out runs/catalogs/206-log -- python3 catalogs/ngc_lab.py image --out runs/catalogs/206
  python3 catalogs/capture.py --out runs/catalogs/208-log -- python3 catalogs/hf_lab.py info --out runs/catalogs/208

The NGC CLI resource is frozen at 4.34.10. The container discovery tag is
26.09-py3; resolve it into a fresh record and use its native child digest.
Tags may move. Retained image.json is evidence of the preparation response,
not a promise that a future tag resolution will return the same bytes.
The registry's anonymous bearer token stays in memory. No user API key is
read, stored or printed by the public helpers.

Hub downloads need huggingface-hub==0.36.0. Offline CPU inference also
needs the packages in requirements-cpu.txt and torch==2.10.0 from the
official CPU index. TARGET_LABS.txt gives the isolated environment setup.
Never replace a qualified CUDA installation with that CPU environment.
The public snapshot helper sets token=False and requests explicit files at
full commits. Downloads go into new destinations outside controls/.

Artifact scope and permitted conclusions
  tiny-random-bert is a random loading fixture, not a useful sentence model.
  demo1 has five train rows and five test rows; it is a split-audit fixture.
  The pinned metadata supplies no license value for either selected repo.
  The dataset's training readiness is explicitly NOT ESTABLISHED.
  Publisher archives, weights and CSV files are not included in this ZIP.
  Receipts and derived numerical results are included. The offline fixture
  records unused task-head weights; no base-encoder weights were missing.
  A Python socket guard is not a firewall for arbitrary native extensions.
  A matching hash proves byte identity against the expected value, not
  model quality or a complete assessment of third-party software.

Target GPU continuation
  Read Chapters 2, 73 and 81 and qualify the existing WSL2 GPU path first.
  Use the mapped Windows NVIDIA driver. Install no Linux display driver.
  python3 catalogs/capture.py --out runs/catalogs/206-gpu -- bash catalogs/run_ngc_probe.sh runs/catalogs/206/image.json
  This pulls about 11.43 GiB of compressed image layers for the retained
  candidate, with additional disk room needed for extraction and caches.
  The Docker --memory 12g limit is a host-memory limit, not a GPU quota.
  Run the Hub CUDA continuation in the already-qualified CUDA environment:
  python catalogs/capture.py --out runs/catalogs/210-cuda-log -- python catalogs/hf_lab.py offline --snapshot downloads/catalogs/hf-model --out runs/catalogs/210-cuda --device cuda

Full implementation
  lock.json               selected asset versions, commits and files
  common.py               bounded downloads, relative paths, file receipts
  ngc_lab.py              resource, ELF, registry and fault controls
  hf_lab.py               metadata, selected snapshots, local model, CSV audit
  gpu_probe.py            explicit CUDA availability and arithmetic oracle
  run_ngc_probe.sh        pinned ARM64 image and bounded Docker invocation
  capture.py              complete console transcript and exit record
  validate.py             fifteen offline regression tests
  requirements-cpu.txt    isolated CPU model/client package pins
  sources.json            primary references 170-185
  TARGET_LABS.txt          complete book chapter text and command transcripts

Credential entry belongs outside capture.py and shell tracing. Keep account
tokens and credential configuration out of experiment archives. The public
controls do not need authentication. For gated assets, account approval,
token scope and the artifact's own terms must be checked separately.
