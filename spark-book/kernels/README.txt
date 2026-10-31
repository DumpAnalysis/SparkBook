Historical commands retain their original working paths. Use reader/guide.py for a fresh run and reader/records.py for the corresponding book pages.

CURRENT READER EDITION — VERSION 56
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

CUDA kernel writing laboratory - Version 3

Run from ~/spark-book in ARM64 WSL2. Reuse the mapped Windows driver.
Prerequisites: qualified NVCC, host C++ compiler, successful Chapter 27 lab.
No third-party libraries or model weights are needed by this laboratory.

Files
  kernels.cuh             complete reference kernels
  kernels_starter.cuh     TODO bodies for the writing exercises
  kernel_lab.cu           allocation, CPU oracles, launch, validation, timing
  build.sh                reference or starter compiler invocation
  run_cases.py            115 bounded GPU cases with complete JSON transcripts
  reference.py            portable teaching oracles and index simulations
  timing_summary.py       CSV arithmetic verification and optional CPU fixture
  validate_reference.py   eight portable checks
  validation.txt          actual portable check report

Build and first run
  bash kernels/build.sh reference
  ./kernels/bin/kernel-lab add 1003 256
Edit the TODO body for add_once in kernels/kernels_starter.cuh, then:
  bash kernels/build.sh starter
  ./kernels/bin/kernel-student add 1003 256
An unimplemented output kernel should fail verification. Start with the
relevant command or case group; the full suite requires all TODOs finished.
Compilation logs are retained under runs/kernels. Builds replace their
named executable and build log. Copy a revision before your next change.

CLI modes (positive integer arguments)
  add N BLOCK
  stride N BLOCK GRID
  transpose naive|tile ROWS COLS
  reduce N BLOCK
  hist N
  matmul naive|tile M K N
  fused N BLOCK
  rms ROWS WIDTH
  softmax ROWS WIDTH
  bench N BLOCK REPEATS NEW_CSV_PATH

Bounds
N <= 2097152; BLOCK is 128, 256, or 512; explicit GRID <= 128.
Transpose dimensions <= 2048. Matmul dimensions <= 128.
Row kernels: ROWS <= 16, WIDTH <= 4096. REPEATS <= 1000.
Zero sizes are rejected by this harness rather than launched.
Transpose tile uses block=(32,8). Matmul tile uses block=(16,16).
Reduction and row kernels require power-of-two block sizes and enough
dynamic shared memory. Histogram uses 256 threads and 16 valid bins.
Arrays are distinct, contiguous, row-major, float32 unless documented.
Histogram inputs/outputs are unsigned 32-bit integers. Inputs are finite.

Complete GPU cases
  python3 kernels/run_cases.py kernels/bin/kernel-student --group vector --out runs/kernels/vector.json
  python3 kernels/run_cases.py kernels/bin/kernel-student --out runs/kernels/all.json
Groups: vector 54; transpose 12; reduce 27; hist 3; matmul 8; rows 8;
fused 3; all 115. Files must be new. A failure is preserved and stops the
runner. These cases were NOT run on a GPU during book preparation.

Timing
  ./kernels/bin/kernel-lab bench 1048576 256 100 runs/kernels/timing.csv
  python3 kernels/timing_summary.py runs/kernels/timing.csv
Both variants must verify. CSV must be new. Five warm-up iterations precede
100 measured iterations in this example. Events bracket a single-stream
launch interval; copies and allocation are outside it. Host submission gaps
may occur inside that interval. The split variant is always measured first;
repeat experiments in reversed order when checking order/thermal effects.
Logical bytes are 20*N for split and 12*N for fused. These are requested
algorithmic bytes, not measured DRAM traffic. Reports contain no invented
GPU speeds. Do not profile the timing process while treating it as a baseline.

Portable checks
  python3 kernels/validate_reference.py
  python3 kernels/reference.py
  python3 kernels/timing_summary.py runs/kernels/synthetic.csv --fixture
The fixture timings are explicitly SYNTHETIC and demonstrate arithmetic only.
Eight portable tests passed in preparation. CUDA C++ was source-reviewed,
but not compiled or executed: no NVCC or target GPU was available.
Python syntax and Bash syntax were checked. No GPU performance is claimed.

Writing progression
45 add_once; 46 add_stride; 47 transpose_naive; 48 transpose_tile;
49 reduce_blocks; 50 histogram16; 51 matmul_naive; 52 matmul_tile;
53 fused_relu and events; 54 timing interpretation and profiler correlation;
55 rms_rows; 56 softmax_rows and complete regression cases.

These are teaching implementations, not optimized replacements for cuBLAS,
framework operators, or model-runtime kernels. No autograd or framework
registration is supplied. See the book for contracts, reasoning, variations,
sanitizer checks, numerical limits, and full command transcripts.
