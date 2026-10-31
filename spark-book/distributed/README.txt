Historical commands retain their original working paths. Use reader/guide.py for a fresh run and reader/records.py for the corresponding book pages.

CURRENT READER EDITION — VERSION 56
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

Exercise 129 now defaults to TORCH_DISTRIBUTED_DEBUG=INFO for DDP and logical
nodes. Use --debug-level OFF for a matched retry in a fresh directory. The
intentional collective mismatch uses DETAIL by default. The independent CPU
review passed under INFO and OFF. The original WSL2 CPU DDP run failed under
DETAIL. The Version 54 follow-up review reports that DDP and logical nodes
both passed the numerical reference check under INFO on x86-64 Linux.
The new WSL2 qualification is pending.

Distributed training on one laptop
Version 11 companion, Chapter 70, Exercises 123-129

Run from spark-book with an already qualified PyTorch environment active.
The executed preparation environment used Python 3.12.14, torch 2.10.0+cpu,
NumPy 2.3.5 and Matplotlib 3.10.8 on x86-64 Linux. These are observed versions,
not an instruction to install an x86 wheel on the ARM64 WSL2 laptop.
No script here installs a driver, downloads a model or changes WSL settings.

START WITH NEW OUTPUT DIRECTORIES
  python distributed/lab.py probe --out runs/dist-probe
  python distributed/pipe_lab.py train --out runs/dist-train
  python distributed/pipe_lab.py train --workers 4 --out runs/dist-four
  python distributed/pipe_lab.py slow --out runs/dist-slow
  python distributed/lab.py plot --source runs/dist-slow --out runs/dist-plot
  python distributed/pipe_lab.py crash --out runs/dist-crash
  python distributed/pipe_lab.py mismatch --out runs/dist-mismatch
  python distributed/pipe_lab.py timeout --out runs/dist-timeout
  python distributed/pipe_lab.py checkpoint --out runs/dist-recovery
  python distributed/pipe_lab.py shard --out runs/dist-shard

These controls use spawned CPU processes and local pipes. The parent validates
collective sequence/operation pairs and computes sums, means or gathers.
This is real interprocess communication, but NOT PyTorch DDP, Gloo, NCCL,
FSDP, a ring all-reduce or a multi-GPU performance emulator.

The 22,880-parameter BERT-style model uses synthetic integer token IDs, two
encoder layers, no dropout and float64. Eight examples and sixteen supervised
tokens form each global batch. Four AdamW updates are compared with an
independent full-batch reference, tolerance 1e-9. Data partitions and rank
parameters are retained. This is separate from Chapter 53's WordPiece model.

Recovery starts NEW processes from each rank's saved model, AdamW state,
next global step and CPU RNG. The data generator depends only on global step.
It requires the same world size. There is no dropout, external sampler,
scheduler or mixed-precision scaler. Only load checkpoints you trust.

Fault exercise success means an EXPECTED_FAILURE was correctly identified.
The driver returns zero for a recognized intended fault, while preserving
nonzero worker exits and the failure classification. A generic failed launch
does not count as the intended fault. Each pipe worker retains stdout, stderr,
events, spans and error text. Original timestamps and paths identify preparation.

PYTORCH DDP AND LOGICAL NODES REQUIRE SEPARATE QUALIFICATION
  python distributed/lab.py ddp --out runs/dist-ddp
  python distributed/lab.py nodes --out runs/dist-nodes
  python distributed/lab.py slow --out runs/dist-ddp-slow
  python distributed/lab.py crash --out runs/dist-ddp-crash
  python distributed/lab.py mismatch --out runs/dist-ddp-mismatch
  python distributed/lab.py timeout --out runs/dist-ddp-timeout
  python distributed/lab.py checkpoint --out runs/dist-ddp-recovery
  python distributed/lab.py shard --out runs/dist-gloo-shard

Preparation could import Gloo, but process-group initialization failed with
Operation not permitted in its network-interface setup. No DDP training step
ran. The logical two-node attempt failed at that boundary too. Full stdout,
stderr, launch arguments and node exit codes are retained under evidence/.
Do not turn those qualification failures into a successful DDP transcript.

The single-node launcher uses static torchrun rendezvous with an explicit
127.0.0.1 address and GLOO_SOCKET_IFNAME=lo. Two logical nodes use distinct
node ranks on the same host. Selecting then releasing a free port has a
bind-to-launch race: a conflict is retained as a failed run. These local-only
launchers do not configure real multi-machine networking.

MULTIPLE CUDA DEVICES
  python distributed/lab.py gpu-ddp --workers 1 --out runs/dist-gpu-one
  python distributed/lab.py gpu-ddp --workers 2 --out runs/dist-gpu-two
  python distributed/lab.py gpu-shard --workers 2 --out runs/dist-gpu-shard

The first command needs one device; the others need two distinct visible
CUDA devices plus a compatible NCCL build and WSL2 execution path. A failed
precondition returns 2 and writes qualification.json with UNAVAILABLE.
LOCAL_RANK selects a different device per local worker. Never repeat device
zero to pretend that a single GPU is two. No successful CUDA execution was
performed during preparation. CPU probe and unavailable GPU evidence do not
describe the owner's laptop. These commands are correctness exercises, not
throughput benchmarks. CUDA phases synchronize and perturb natural overlap.

SHARDING AND TIMELINES
Sharding checks a 4x6 weight matrix split into two 4x3 slices: forward output,
summed input gradient, and local weight gradients. The full reference is also
constructed on each worker, so logical ownership is not a process-memory saving.
The plot shows host intervals, including transfer and waiting, on one host's
monotonic clock. It is not a GPU trace. timeline.json uses Chrome trace events.
Do not infer network bandwidth or multi-host clock agreement from it.

EVIDENCE AND REPRODUCTION
  python distributed/reproduce.py --out runs/distributed-reproduction
  python distributed/validate.py

reproduce.py captures new driver outputs, child logs and actual return codes.
It permits DDP, logical-node and GPU qualification failures while requiring
the pipe controls to pass. Check each qualification separately.
validate.py checks the SUPPLIED preparation evidence plus independent
weighted-loss and bad-shard-order counterexamples. Fourteen checks passed.
It does not automatically validate a newly reproduced directory. Compare new
results against their contracts and reference errors before claiming success.

The fourteen checks are separate from the earlier 121 checks recorded in
Version 9; those earlier suites were not rerun for this addition. No WSL2,
ARM64, GPU or NCCL performance validation is implied by these CPU results.
