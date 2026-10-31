Recorded source paths in this note resolve under the separately restored evidence root.
See reader/EVIDENCE.txt for path lookup.

CURRENT READER EDITION — VERSION 55
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

Version 4 advanced companion
Chapters 40-48 and Appendix F of Running and Diagnosing Local Models on RTX Spark Laptops

Run commands from ~/spark-book inside ARM64 WSL2. The Windows NVIDIA driver
is already mapped into WSL2. These files install no Linux display driver.
Keep the 64 GB installed-memory observation separate from the approximately
30 GB reported GPU budget and from current free/total CUDA bytes.

Evidence status
reference.py, casebook.py, diagnostic_agent.py replay modes, measure.py template,
and validate_advanced.py were run on a preparation CPU without CUDA or PyTorch.
Their seven complete sessions are in transcripts.txt. validation.txt records
21 portable tests. They do not execute CUDA, a transformer model, or live model
planning. CUDA and PyTorch source was reviewed, not compiled/executed here.
Target success examples in the book are ILLUSTRATIVE. No laptop benchmarks
are included. CUTLASS must be separately qualified for the actual device.

Portable exercises (Python 3.10+, except CUTLASS helper requires 3.11+)
  python3 advanced/validate_advanced.py
  python3 advanced/reference.py transformer --out runs/advanced/reference.json
  python3 advanced/reference.py int4 --out runs/advanced/int4.json
  python3 advanced/casebook.py
  python3 advanced/diagnostic_agent.py --replay good --out runs/advanced/agent-good.json
  python3 advanced/diagnostic_agent.py --replay missing --out runs/advanced/agent-missing.json
  python3 advanced/diagnostic_agent.py --replay denied --out runs/advanced/agent-denied.json
  python3 advanced/measure.py --template --out runs/advanced/measurement-template.json
Use new output names on repeats. The denied replay intentionally returns 2.

Standalone CUDA build and execution (qualified ARM64 toolkit and cuBLAS)
  bash advanced/build.sh
  ./advanced/bin/collectives warp
  ./advanced/bin/collectives tensor
  ./advanced/bin/collectives int4
  ./advanced/bin/async-lab buffers
  ./advanced/bin/async-lab bad
  ./advanced/bin/async-lab fixed
  ./advanced/bin/async-lab graphs
  ./advanced/bin/instruction-debug
  ./advanced/bin/instruction-opt
Compiler logs are under runs/advanced. Save them before rebuilding.
The bad dependency case deliberately completes consumer before producer;
it exposes a deterministic stale result without undefined concurrent access.
It exits zero when that intended negative control is observed.

Contracts
common.cuh owns checked allocation/stream/event helpers; destructors make
best-effort cleanup. collectives.cu uses a single 256-thread block for reduction,
a single 32-thread warp for a 16x16x16 FP16 WMMA tile, and bounded INT4 fixtures.
The INT4 format has one scale, signed [-8,7] codes, low nibble first, half-way
rounding away from zero, saturation, and a zero unused high nibble. It is NOT
GGUF, NF4, or a production model format. Empty INT4 input launches no kernel.
async_lab.cu keeps fixed allocations alive through graph replay. Event intervals
and host-submit intervals describe different scopes. Capture/instantiation are
excluded. Actual overlap and speed must be measured, not inferred.

PyTorch route (the qualified .venv-ft baseline from Chapter 21)
  source .venv-ft/bin/activate
  MAX_JOBS=2 python advanced/torch_operator.py
  MAX_JOBS=2 python advanced/torch_operator.py --bad-backward
  python advanced/transformer_torch.py --device cpu --out runs/advanced/transformer-cpu.json
  python advanced/transformer_torch.py --device cuda --out runs/advanced/transformer-cuda.json
  python advanced/training_diagnostics.py audit --device cuda
  python advanced/training_diagnostics.py resume --device cuda --out runs/advanced/resume-a
The --bad-backward negative control returns 2 only after gradcheck exposes its
wrong formula. Run the correct and incorrect registrations in separate processes.
The extension needs Ninja, compiler headers, NVCC, and compatible CUDA/PyTorch
builds. It is not an ABI-stable wheel. Its contract is contiguous equal-shape
CUDA float32/float64 inputs on one device, no broadcasting, and a new output.
Fake/autograd registration, forward oracle, opcheck, gradcheck, noncontiguous
rejection, and current-stream checks remain separate checks.
The transformer has deterministic teaching matrices, one head of width 8,
5 tokens, an MLP width of 16, and adjacent-pair RoPE. It has no pretrained
weights or language-generation capability. Float64 is diagnostic precision.
The training demo is a tiny float32 network, not LoRA. It checkpoints model,
optimizer, Python/CPU/CUDA RNG, step and data cursor. It has no scheduler or
scaler; real training that uses those must save their state as well.

CUTLASS
Use cutlass_compare.py --help. Supply a qualified profiler path, an exact
compatible kernel name from its operation list, and its 40-character source
commit. The helper retains commands, binary hash, console, and CSV and requires
passed cuBLAS verification. No default operation is assumed for RTX Spark.
The helper uses fixed 512x512x512 dimensions and FP16 row-major inputs with
FP32 accumulation/output. Do not compare its time with the 16x16 WMMA fixture.

Debugger and tracing
  cuda-gdb --nx --batch -x advanced/inspect.gdb advanced/bin/instruction-debug
Use Chapter 29's WSL2 prerequisites and launch-debugging restrictions.
Read Chapter 48 for PTX, SASS, register, and exact-binary identity capture.
Use Chapter 43's Nsight Systems commands only after that tool is qualified.
Debugger/sanitizer/profiler modes are separate experiments.

Agent and actual measurements
prepare_evidence.py takes four explicit operator-selected files and writes
bounded snapshots with source metadata. diagnostic_agent.py --evidence DIR
uses the existing local inference service; --replay uses no planner.
Only inventory, sanitizer, timeline, and memory can be read. No arbitrary file
or shell tool exists. Exact quote validation does not prove the interpretation.
measure.py --torch --out NEW.json collects WSL observations, not benchmarks.
Each command has its own status. The template deliberately has no measurements.

All source files are complete. Build outputs, model weights, environments, and
local hardware captures are excluded from the archive. Preserve actual logs
and source revisions in your own experiment directories.
