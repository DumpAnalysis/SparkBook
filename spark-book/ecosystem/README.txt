Recorded source paths in this note resolve under the separately restored evidence root.
See reader/EVIDENCE.txt for path lookup.

CURRENT READER EDITION — VERSION 55
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

ECOSYSTEM LABORATORY - VERSION 17

Chapters 76-80, Exercises 152-171. Run commands from spark-book.
Target: ARM64 WSL2, Windows NVIDIA driver already mapped into Linux.
Example: 64 GB unified system memory, about 30 GB reported GPU memory.
The owner's CUDA 13.4 report is not a certificate for every runtime.

EXECUTED HERE
  Python 3.12.14, NumPy 2.3.5, ONNX 1.20.1, ONNX Runtime 1.23.2,
  PyTorch 2.10.0+cpu. x86-64 Linux; no GPU, Docker or target frameworks.
  controls/ contains eight complete command transcripts, raw ONNX graphs,
  inputs and outputs, runtime JSON trace, native rejection errors,
  quantization metrics, split/mask evidence and a tiny CPU adapter.
  These synthetic controls are not trained semantic encoders or laptop
  benchmarks. Twenty checks passed; see controls/validation.json.

REPRODUCE CPU CONTROLS IN A SEPARATE ENVIRONMENT
  python3 -m venv .venv-ecosystem-cpu
  source .venv-ecosystem-cpu/bin/activate
  python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -r ecosystem/requirements-cpu.txt
  python ecosystem/reproduce.py --out runs/ecosystem-cpu
  python ecosystem/validate.py --evidence runs/ecosystem-cpu

Use a new output directory for every run. Missing CPU wheels on a target
architecture are a packaging result; do not substitute x86 binaries.
Do not apply CPU requirements to an existing CUDA environment.

TARGET PROCEDURES - NOT EXECUTED IN PREPARATION
Read TARGET_LABS.txt, which reproduces all five chapters.
  rapids_target.py: cuDF groups, cuML PCA and RMM allocation accounting.
  lab.py ort --provider CUDAExecutionProvider: strict provider observation.
  export_encoder_target.py: local encoder with explicit masked mean + L2.
  tensorrt_target.py: TensorRT 11 engine build, dynamic profile and values.
  make_triton_plan.py: packages an engine only after its report passes.
  nemo_prepare_target.py: pinned public model/data snapshots and a small
    recipe derived from the qualified image's own YAML.
  nemo_export_target.py: accepts one consolidated HF-compatible adapter,
    compares merged logits, and writes a serving handoff with file hashes.
  capture.py --out NEW -- COMMAND: full merged stdout/stderr and exit.

Candidate tags are discovery inputs; resolve immutable ARM64 digests and
qualify actual device execution before continuing. Follow the book's
existing serving Docker qualification helper. No Linux driver install.
Do not run several GPU owners while establishing the baseline.

TensorRT 11 uses strong typing and explicit QDQ; old --fp16/--int8 flags
are not used. ONNX Runtime GPU/provider versions need their own matrix;
the CPU control version does not imply TensorRT 11 compatibility.

NeMo training is a target experiment. The tiny CPU LoRA control is not
NeMo. Four training updates and a one-prompt merge comparison are
execution/conversion checks; use held-out quality tests before serving.

Evidence is machine specific. Paths/timestamps in retained runtime traces
identify the preparation run. Reproduce into a new directory to get your
own trace and compare outcomes rather than byte-identical timing.

Version 15 clarifications
  RAPIDS runs as the WSL UID/GID, with writable HOME and CuPy cache in /tmp.
  PCA uses row-form C [2,16], so its projector C.T @ C is [16,16].
  The 1.23.2 CPU package is not the ARM64 GPU candidate. PyPI 1.29.0
  provides aarch64 GPU wheels for Python 3.11-3.14 and glibc 2.34+,
  built for CUDA 13.0+ and cuDNN 9; qualify these separately.
  NeMo preparation checks mappings/RNG schema before downloads and retains
  StatefulRNG. CPU export checks for Hugging Face peft explicitly.
  Exact target commands and references are in TARGET_LABS.txt.

Version 17 RAPIDS correction
  --group-add conda retains access to root:conda mode-770 /opt/conda while
  --user keeps bind-mounted outputs host-owned. Do not assume the host GID
  matches the image conda GID. Distinguish conda-path denial, /out ownership
  and CUDA failures. The new command check uses different host IDs, but
  neither it nor the earlier Docker double executes a RAPIDS image.
  Qualify actual image permissions and GPU execution on the WSL2 laptop.
