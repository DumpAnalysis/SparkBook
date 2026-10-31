Historical commands retain their original working paths. Use reader/guide.py for a fresh run and reader/records.py for the corresponding book pages.

CURRENT READER EDITION — VERSION 56
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

Sharing One GPU Among Workers
Chapter 71, Exercises 130-135

Recommendation
Start with CPU preparation/request workers and one process owning the CUDA
model. For training one model, keep its optimizer in that same training
process. Sharing compute does not create extra memory or extra CUDA devices.
The approximately 30 GB GPU report is a common budget within the example
laptop's 64 GB unified system, not a per-worker allocation.

Scope and prerequisites
Use the qualified WSL2 framework environment from Chapter 22. The preparation
environment was x86-64 Linux, Python 3.12.14 and PyTorch 2.10.0+cpu; see
evidence/environment.json for exact versions. These scripts need torch,
numpy and matplotlib. They download nothing and install no driver.
CUDA execution, WSL2 performance, MPS and CUDA IPC remain unqualified here.

Run from spark-book, with a NEW output directory for every invocation:
  python sharing/reproduce.py --out runs/sharing-repeat
  python sharing/validate.py --evidence runs/sharing-repeat

Validate the supplied evidence without rerunning the experiments:
  python sharing/validate.py

The reproduction driver preserves commands, combined stdout/stderr and exit
codes. It bounds each command to 120 seconds and cleans up only its own
process group. A CUDA absence is UNAVAILABLE (exit 2), not a passing GPU test.

Program contracts
core.py builds a deterministic 20,544-parameter embedding fixture. Its
82,176 weight bytes are a logical tensor payload, not process or GPU memory.
The 16 requests each contain 16 synthetic integer tokens and return a
normalized 64-element vector. This is not a trained sentence encoder.
The NumPy oracle checks numeric outputs and identity; it measures no quality.

service.py launches CPU client processes and separate model-owning processes
with spawn. In shared mode, one owner handles all clients. In replicas mode,
each client has a separate owner and model copy. CUDA owners all use cuda:0.
There are no DDP/NCCL ranks, CUDA IPC tensors, network listeners or MPS setup.
Only Python values pass through local pipes; tensors stay inside the owner.
Exactly one outstanding request per client bounds accepted work. The parent
dispatches at most one batch per owner and rotates the client starting point.
It uses a fixed max batch size and a collection window after admission. The
window is not a total response deadline. Round-robin request selection does
not guarantee equal GPU time for requests of different lengths.

  python sharing/service.py --workers 4 --batch 4 --out runs/share-cpu
  python sharing/service.py --device cuda --workers 2 --batch 2 --out runs/share-gpu
  python sharing/service.py --device cuda --mode replicas --workers 2 --batch 1 --out runs/replicas-gpu

The GPU version uses a tiny fixture; measure a representative model before
using these recommendations for capacity planning. Every owner warms up one
request before the timed phase. Timing starts after all processes are ready
and ends when all clients finish, including dispatch and result delivery.
Client latency is send-to-receive. Queue delay is admission-to-owner-start,
including dispatch and collection. Host service includes input conversion,
model execution and output conversion; GPU completion is observed by the
synchronous CPU result copy. Sixteen requests are a correctness run, not a
credible latency-tail benchmark. Repeat longer trials with equal profiling,
warmup, shapes, requests and other GPU activity to compare performance.

--profile writes owner-N-profiler.json with CPU activity and, for CUDA,
requested CUDA activity. Confirm actual device events in the file before
claiming a GPU trace; capture failures remain errors. The supplied CPU
profile emitted a PyTorch cycle warning and a process-name lookup warning;
both are preserved in the complete transcript. It contains operator events.
Profiled timing must not be directly compared with an unprofiled baseline.

Each CUDA owner records allocated/reserved bytes and their peaks after the
warmup. Allocated bytes are inside reserved bytes; never add them together.
The counters omit other processes and allocations outside that allocator.
Independent peaks need not occur together. Do not infer total GPU memory
from model weight bytes or treat allocator limits as resource reservations.

train.py runs two CPU-only preparation processes and a single model/optimizer
owner. At most two batches are in flight. Four ordered AdamW updates are
checked against a serial CPU reference, with parameter tolerance 1e-5 and
loss tolerance 1e-6. The model is an embedding fixture trained on synthetic
vector targets, not a BERT or sentence-quality experiment. Its checkpoint
records the next step and optimizer state; this script does not resume it.
  python sharing/train.py --device cuda --out runs/one-trainer

faults.py is a SEPARATE CPU-only control-plane fixture, not an advertised
cancellation API in service.py. It checks capacity two, pending cancellation,
expiry at explicit logical times, and an owner exiting 17 before computation.
A fresh owner executes a read-only request. An interrupted reply has UNKNOWN
outcome; retries of optimizer updates or external agent actions require an
idempotency/commit protocol. It does not kill an active GPU kernel, test
device OOM recovery or demonstrate persistent model recovery.
  python sharing/faults.py --out runs/sharing-faults

plot.py renders all client response waits and owner host-service spans. The
Chrome trace uses a common host monotonic clock. These spans are not kernels.
  python sharing/plot.py --run runs/share-cpu --out runs/share-timeline

streams.py is a target-only CUDA correctness exercise with two streams,
pinned host input, asynchronous copies, explicit dependencies and retained
buffer lifetimes. It exports a profiler trace and checks two matrix products
against CPU. It does not infer overlap from two stream objects or sum their
event times into wall time. GPU traces and profiler support require local
qualification. No CUDA success transcript is supplied.
  python sharing/streams.py --out runs/two-streams

For real sentence embeddings, replace the fixture with a pinned encoder and
preserve tokenization, pooling, normalization, model revision and request
ordering. Batch only compatible model/adapter/configuration requests. For
agents, use bounded requests to the existing model server; the clients here
are workload generators, not autonomous planners. Training and interactive
inference usually deserve separate scheduled windows unless measured memory
peaks and response latency justify concurrent operation.

MPS and IPC
The main exercises require neither. NVIDIA's current MPS documentation
describes Linux support and MPS v3, but does not establish compatibility with
the exact ARM64 WSL2 RTX Spark setup. Do not enable MPS for these exercises
without separately qualifying support, monitoring attribution, shutdown and
failure handling. The owner cleanup above assumes ordinary independent CUDA
processes. CUDA tensor IPC additionally requires producer lifetime control
and target support; unified memory alone proves neither.

Validation
Twelve new checks passed against the retained CPU evidence. Earlier book
checks were not rerun. validation.txt lists the checks. The validator accepts
--evidence so a newly reproduced directory can be checked independently.
All code is included; the source is syntax-checked. GPU branches were reviewed
but not executed in preparation. No CUDA/WSL2 benchmark is claimed.
