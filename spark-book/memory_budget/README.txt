Recorded source paths in this note resolve under the separately restored evidence root.
See reader/EVIDENCE.txt for path lookup.

CURRENT READER EDITION — VERSION 55
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

EXERCISE 251 — WSL2 AND CUDA MEMORY BOUNDARIES

Purpose
Explain why a 16 GB WSL2 limit, 30.50 GiB NVML total and 45.97 GiB CUDA
reported total can coexist. Distinguish a query from successful allocation,
GPU writes and complete retained-byte verification. Chapter 3 gives the
explanations and interpretation; existing Exercises 1–250 retain their IDs.

Evidence status
The recorded allocation investigation supplies a query (OBSERVED), a PASSED
1 GiB pilot and a PASSED 18 GiB allocation with simultaneous retention, writes,
every-byte verification, final exits and release observations. Verified
19,327,352,832 bytes exceeded guest MemTotal 16,854,429,696 bytes.
Use python reader/records.py 251 --files to locate the complete records.
The original terminal pastes remain historical REPORTED queries in
USER-OBSERVATIONS.txt. Missing timestamps, stderr and exits are not invented.
The archive retains earlier preparation controls and the exact six-file
source subset executed by the owner. After restoration, that source subset
is in exercises/251/resources/source/ under SPARK_BOOK_EVIDENCE.
The pilot includes two sampled guest swap-out intervals. The target includes
none in eight intervals, without proving no paging or physical residency.
Windows samples supply host observations and additional live guards.
Archived automation uses sealed absolute destinations. Use the reusable
procedures below with new output paths for your own investigation.

Prerequisites
Use ~/work/spark-book in WSL2 and its already qualified .venv-exercises.
PyTorch with working CUDA is required. No package installation, model
acquisition, driver change or network access is performed by these helpers.
Read-only mode still initializes a CUDA context, which may consume memory.
Save other work and stop your own model/training jobs before allocation.
Check Windows Available RAM and background activity in Task Manager; retain
that observation manually. Guest and CUDA checks cannot see all host pressure.

1. Collect the three views
  cd ~/work/spark-book
  source exercise-env.sh
  python -B memory_budget/run.py --out runs/memory-views-001

Read stdout.txt (JSON lines), stderr.txt and metadata.json in the new directory.
The wrapper records the exact interpreter/command, UTC times, hashes, PID,
raw streams, child exit and wrapper exit. Raw nvidia-smi/free command output
and return codes are nested losslessly as strings in stdout.txt. The source
hashes identify this helper. /proc/meminfo supplies bytes, avoiding free -h
rounding. Calls are sequential, not simultaneous. Missing tools remain marked
UNAVAILABLE; no substitute is silently used. CUDA absence exits 3, not a pass.
Zero PyTorch allocator counters do not mean no device/driver/context memory.

2. Run a 1 GiB pilot
Record the ACTUAL Windows observation in the variable before invoking this:
  HOST_NOTE='Replace with actual Windows Available RAM, time and workload state'
  python -B memory_budget/run.py --out runs/memory-pilot-001 \
    --allocate-gib 1 --host-note "$HOST_NOTE"

The quoted placeholder is a procedure, not evidence. Fill it in. Examine
host/guest pressure and the complete result before deciding on the next run.
Do not automatically retry a failed, interrupted, timed-out or guarded run.

3. Optional check above the observed guest MemTotal of 15.70 GiB
Use only after the pilot passes and sufficient host headroom remains. Record
a fresh actual HOST_NOTE. Keep Task Manager visible. In another WSL terminal,
  vmstat 1
The first vmstat row is a since-boot average; inspect later si/so activity,
not swap usage alone. Stop the probe with Ctrl+C if host responsiveness or
available memory deteriorates. Preserve the failure and monitor record.
Then, in the exercise terminal:
  python -B memory_budget/run.py --out runs/memory-above-guest-001 \
    --allocate-gib 18 --host-note "$HOST_NOTE"

Fixed bounds
The helper accepts integers 0..18 GiB; 0 queries only. Allocation uses 1 GiB
uint8 blocks and retains all of them simultaneously. It writes a different
constant byte pattern per block, synchronizes, then checks EVERY byte of ALL
retained blocks in 8 MiB tiles. It does not create an 18 GiB CPU copy. Each
tile comparison may allocate an additional small GPU buffer. It refuses the
next block below 2 GiB guest MemAvailable or below next-block-plus-4 GiB
reported CUDA free memory. These are teaching guardrails, not guarantees of
host stability. It refuses allocation outside a WSL kernel and never falls
back to CPU. No allocation search or automatic escalation occurs.
The default process timeout is 180 seconds, configurable only from 10..300.
Timeout/interrupt signals only the owned child process group. It gives that
child 10 seconds to release memory, then kills it if necessary. A forcibly
terminated child may have no final release record; process teardown is not
a measured clean release. The wrapper preserves the status.

Results and interpretation
OBSERVED/exit 0: queries completed; capacity remains untested.
PASSED/exit 0: all requested bytes were retained, written, and checked on
CUDA; examine process_result and wrapper metadata as well as the result line.
The report compares those bytes with the captured guest MemTotal. A successful
18 GiB run exceeds an approximately 16 GiB guest only if this comparison is true.
This proves that bounded allocation and access at that time, not residency,
maximum capacity, model fit, bandwidth, absence of paging or training success.
A single constant per block detects some corruption but is not exhaustive
memory testing, address-line testing or a hardware reliability certification.
FAILED/exit 1: preserve type/message and the last retained count; do not infer
that 16 GiB is a universal ceiling. Small validation buffers can also fail.
UNAVAILABLE/exit 3: CUDA/import prerequisite absent; no CPU replacement.
Wrapper exit 124: timed out; 130: interrupted. Other child exits remain visible.
A cleanup observation failure makes the process nonzero even if the earlier
allocation check said PASSED. Read the full record.
After release, empty_cache releases unused PyTorch cache; it does not destroy
the CUDA context or other processes' allocations. Nonzero counters alone do
not establish a leak. An empty process table does not establish an idle device.

Optional guest-limit comparison
Only if CPU-side loading/offload needs more memory, compare the original limit
with a separately recorded 32 GB ceiling. Create .wslconfig if absent, or save a copy first; changing
it belongs to the Windows host, not a Linux driver installation. Save every
WSL distribution's work before wsl --shutdown, which stops all distributions.
Reopen WSL, use a new output directory and repeat the same bounded workload.
Record Windows/WSL state and other jobs. Do not combine before/after snapshots
as one execution. Increased guest headroom need not increase CUDA totals.
Do not increase to 48 GB merely because CUDA reported about 46 GiB.
Swap is disk-backed guest virtual memory, not extra physical GPU capacity.

Source explanation
NVIDIA's Windows on Arm RTX Spark guide describes a dedicated carveout and
shared segment backed by the same physical RAM. NVML reports the dedicated
segment; cudaMemGetInfo covers dedicated plus shared budgets. GPU allocations
can spill into shared memory with different performance. This documented
host model is consistent with the WSL observations; it does not establish an
exact segment map for this WSL process. Do not subtract readings across APIs
or add them into a new physical capacity. CUDA explicitly does not guarantee
allocation of every byte reported as free. Physical UMA is distinct from
CUDA managed allocation and WSL's supported managed-memory features.
See SOURCES.json for primary documentation checked on 30 October 2026.
