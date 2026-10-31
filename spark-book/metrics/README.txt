Historical commands retain their original working paths. Use reader/guide.py for a fresh run and reader/records.py for the corresponding book pages.

CURRENT READER EDITION — VERSION 56
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

GPU metrics and activity visualization — Version 6, Chapter 57

Run from the spark-book directory in WSL2. Use an already qualified Python
and CUDA environment. These programs install no driver and change no clocks,
power limits or system settings. NVIDIA driver access remains mapped from Windows.

Programs
monitor.py       stdlib sampler and bounded foreground-command wrapper
make_plan.py     writes readable JSON arrays of command arguments
plot.py          NumPy/Matplotlib plots, CSV, availability and timing summaries
fixture.py       deterministic, explicitly ILLUSTRATIVE teaching observations
compare.py       compares successful sessions with identical plan definitions
validate_metrics.py  21 portable checks; no RTX Spark required

Requirements
Python 3.10+ for collection. NumPy and Matplotlib for plotting/tests.
Use the embedding environment's pinned numpy==2.3.5 and matplotlib==3.10.8,
or retain the versions in another qualified environment. The collector does
not need PyTorch. Embedding/training workloads need their earlier environments.
The installed nvidia-smi must be compatible with the mapped Windows driver.
If not in PATH, monitor.py tries /usr/lib/wsl/lib/nvidia-smi. An absent executable
is recorded as a query failure, not repaired by installing a Linux driver.

First verify portable behavior
  python metrics/validate_metrics.py
  python metrics/fixture.py --out runs/metrics/fixture-new
  python metrics/plot.py --run runs/metrics/fixture-new --out runs/metrics/fixture-new-plots
All GPU numbers from fixture.py are invented. Never put them in a hardware report.

CPU control
  python metrics/make_plan.py cpu --out runs/metrics/cpu-plan.json
  python metrics/monitor.py --plan runs/metrics/cpu-plan.json --out runs/metrics/cpu-01 --no-gpu
  python metrics/plot.py --run runs/metrics/cpu-01 --out runs/metrics/cpu-01-plots
The reference plans in evidence/ contain preparation-machine interpreter paths.
Regenerate plans using your active WSL2 interpreter before running them.

GPU idle capture
  python metrics/make_plan.py idle --out runs/metrics/idle-plan.json
  python metrics/monitor.py --plan runs/metrics/idle-plan.json --out runs/metrics/idle-01 --interval 1
  python metrics/plot.py --run runs/metrics/idle-01 --out runs/metrics/idle-01-plots
Use --gpu GPU-... with the recorded UUID when available; default selector is 0.
Review session success and per-field statuses. Successful commands do not imply
usable GPU counters. No single query result proves correct CUDA placement.

Other plan kinds
inference: start the existing CUDA-qualified server first; wraps five requests.
embedding: stops neither servers nor other programs; prepare the local MiniLM
           snapshot and stop the inference server yourself before running.
training:  requires the qualified embedding trainer and uses three epochs.
           Supply --batch 4 to retain the book's training setting.

Example
  python metrics/make_plan.py embedding --batch 8 --out runs/metrics/emb-plan.json
  python metrics/monitor.py --plan runs/metrics/emb-plan.json --out runs/metrics/emb-01

The generated plan executes a foreground argv array, without a shell.
Review it before use. A literal {run} in an argument becomes the absolute output
session directory. The default workload outputs are therefore retained under
each session even when the plan is reused. --result may select another path;
then the caller is responsible for preserving separate workload outputs.
Plans are limited to 32 phases; each has a timeout no greater than one hour.
The generator supplies finite timeouts and bounded teaching workloads.
On a failed command the remaining phases are skipped. A timed-out or interrupted
owned foreground process group receives termination. A separately running
inference server is not owned by the wrapper. Do not use daemonizing commands.
A successful command exit is not a device-level CUDA completion marker.

Files in each capture
metadata.json    platform, interpreter, field units, plan, probe results, origin
raw-queries.jsonl complete argv, status, stdout/stderr and times for each query
samples.jsonl   GPU values/statuses, query spans, guest memory and UTC labels
events.json     resolved workload argv, host phase intervals and exit statuses
session.json    completion count, collector errors and success
phase-NN.stdout.txt / stderr.txt  full workload console (including failures)
workload*       workload result or trained artifact when using the default plan

Each numeric field has a status. N/A becomes null/unavailable, never zero.
Zero remains a valid observation. Unexpected CSV shape or invalid numeric text
is rejected. Probe queries run individually; accepted fields are polled together.
N/A fields stay eligible for later readings. Rejected fields require a new
session to re-probe. A failed combined poll records query_error for its fields.
Polling deadlines use a monotonic clock; missed slots are skipped. Each query
has a five-second timeout. Requested interval accepts 0.1..60 seconds.
Device sensor windows and refresh rates may differ from this polling cadence.
Memory guest fields come from /proc/meminfo, converted kB to MiB with /1024.

Plot outputs
activity.png     sampled GPU/memory busy, memory capacity, guest memory, phases
environment.png  temperature, power, clocks, query duration, phases
samples.csv      wide numeric/status table; unavailable numeric cells are empty
summary.json     counts, statuses, sample min/median/max, observed intervals
A line gap is inserted when adjacent retained times differ by more than 2.5 times
the requested interval. Missing values also break lines; data are never imputed.
Legends display valid/retained row counts, not elapsed-time coverage. The drawing
is an offline report, not a live dashboard. All raw time spans remain available.
Summary phase counts require a query span fully inside the host interval;
sensor averaging windows may still cross its boundaries. No energy estimate is
computed, and sample medians are not time-weighted means.

Monitoring comparison
Run the identical plan with --without-sampling for a baseline; it still records
phase timing and workload output. Alternate order, retain at least five runs
per condition, and compare application-level throughput as well.
  python metrics/compare.py --baseline runs/metrics/base-01 --observed runs/metrics/observed-01
The command accepts more directories after each option. It rejects different
plans, failed sessions and illustrative fixtures. A single pair cannot establish
stable overhead. A faster observed run is not evidence that sampling helps.
The baseline has no samples.jsonl and is not a plotting input.

Limits
Device-wide counters may include Windows desktop and other WSL processes.
The 64 GB physical system, roughly 30 GB reported GPU pool and WSL guest memory
must not be added. These views may overlap. Utilization is not achieved FLOPS.
Memory busy is not allocated capacity. Power scope is not automatically the
entire laptop. Host phase bands are not GPU stream lanes or NVTX ranges.
Use Chapters 31–33 for actual CUDA timelines, framework traces and counters.
Nsight hardware-metric support and permissions must be checked separately.

Evidence
metrics/evidence/ includes complete executed CPU transcripts and output files.
The fixture GPU data are ILLUSTRATIVE. CPU-control memory/timing were measured
in a preparation Linux environment, not on the example laptop. GPU target
exercises remain unexecuted. Tests cover parser failure, gaps, phase retention,
timeouts, CPU-disabled semantics and per-run output paths.

Version 9 host-observation fallback
Use --windows-host on monitor.py to invoke Windows nvidia-smi.exe through
WSL interop, or add --smi /mnt/c/.../nvidia-smi.exe for a verified path.
The workload stays in WSL. Metadata identifies the GPU provider; guest_MiB
still describes WSL. Plot titles identify the Windows host source.
NVIDIA documents missing guest utilization and active-process queries.
Host availability is also measured per field and is not guaranteed.
Separate host and guest runs must not be spliced into a simultaneous trace.
