Recorded source paths in this note resolve under the separately restored evidence root.
See reader/EVIDENCE.txt for path lookup.

CURRENT READER EDITION — VERSION 55
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

Version 7 diagnostic studies
Chapters 58-69, Exercises 99-122. Run from the spark-book root under WSL2.
Use the existing Windows-mapped NVIDIA driver. Do not install a Linux display driver.
The example has 64GB system memory and approximately 30GB GPU-reported capacity;
these are overlapping accounting views, not additive memory pools.

Programs
lab.py: numerical, compiler, pipeline, roofline, memory, hang, request, quality,
        reduce and audit controls. Run python diagnostics/lab.py --help.
casebook.py: inspect retained boundaries and reproducers, patterns, blind cards
             and separate answers; generate a checked worked diagnosis.
validate.py: 24 portable regression checks.
patterns.json: 12 teaching patterns with alternatives and applicability limits.
blind/cases.json and blind/answers.json: separate evidence and worked answers.
evidence/: complete command transcripts, full redirected stderr, output data,
           compiler source logs, stack dumps and figures from preparation.

Dependencies and scope
Use the isolated qualified PyTorch environment from earlier chapters.
CPU preparation used torch 2.10.0+cpu, OpenTelemetry API/SDK 1.38.0.
Exact installed versions of plotting libraries are in evidence/versions.txt.
To add local tracing dependencies to that active environment:
  python -m pip install opentelemetry-api==1.38.0 opentelemetry-sdk==1.38.0
Roofline and trace plots also need numpy and matplotlib from earlier labs.
Do not install the CPU torch package over a qualified CUDA build on the laptop.
No publisher model downloads are needed for these new controls.

Representative commands (output directories must be new)
  python diagnostics/lab.py numerics --out runs/numeric-control
  python diagnostics/lab.py numerics --fault --out runs/numeric-fault
  python diagnostics/lab.py compile --mode static --out runs/capture-static
  python diagnostics/lab.py compile --mode dynamic --out runs/capture-dynamic
  python diagnostics/lab.py compile --mode break --out runs/capture-break
  TORCH_LOGS=output_code python diagnostics/lab.py compile --mode inductor --out runs/inductor 2>runs/inductor-code.txt
  python diagnostics/lab.py pipeline --mode ready --out runs/supply-ready
  python diagnostics/lab.py pipeline --mode delay --out runs/supply-delay
  python diagnostics/lab.py pipeline --mode prefetch --out runs/supply-prefetch
  python diagnostics/lab.py roofline --out runs/roofline
  python diagnostics/lab.py memory --mode retain --out runs/memory-retain
  python diagnostics/lab.py memory --mode release --out runs/memory-release
  python diagnostics/lab.py memory --mode vary --out runs/memory-vary
  python diagnostics/lab.py hang --mode blocked --out runs/hang-blocked
  python diagnostics/lab.py hang --mode repaired --out runs/hang-repaired
  python diagnostics/lab.py request --mode normal --out runs/request-normal
  python diagnostics/lab.py request --mode slow --out runs/request-slow
  python diagnostics/lab.py request --mode retry --out runs/request-retry
  python diagnostics/lab.py quality --out runs/quality
  python diagnostics/lab.py reduce --out runs/reproducer
  python diagnostics/lab.py audit --trace runs/request-retry --out runs/audit
  python diagnostics/validate.py
  python diagnostics/casebook.py patterns
  python diagnostics/casebook.py case --id C01
  python diagnostics/casebook.py case --id C01 --answer

Contracts
Numerics, compile, pipeline and memory accept --device cuda, with no silent CPU
fallback. These GPU paths have not been executed in preparation. Memory snapshots
require --device cuda --snapshot and use version-sensitive underscored APIs.
Memory controls allocate at most a few MiB; they do not attempt GPU exhaustion.
The capture-count compiler backend executes FX eagerly, not optimized GPU code.
Inductor was actually compiled on CPU. First-call time is not guaranteed cache-cold.
The pipeline control uses one producer thread and a queue of two batches, not
DataLoader multiprocessing. Work time includes controlled host delay. Device
completion is synchronized per batch; no copy/compute overlap is asserted.
Hang mode launches and stops only its own child. Stack and progress files survive.
OpenTelemetry uses ALWAYS_ON local sampling; no remote exporter is configured.
Its small in-memory exporter is a fixture, not a long-running production design.
Request, quality and blind fixtures execute no learned model. Roofline ceilings
are explicitly hypothetical. Case C05 is illustrative GPU reasoning.
The audit is intentionally narrow, requiring its teaching schema. It is not a
validator for arbitrary OpenTelemetry exports or Chapter 57 telemetry files.
A successful control exit can mean the expected fault was detected, not that the
faulty child/workload itself succeeded. Preserve the individual outcomes.

Evidence and reproduction
29 command transcripts are retained; redirected stderr files are complete.
Use new run directories and filenames; bundled evidence paths already exist.
Timings, paths, thread addresses and trace IDs vary across systems. Inspect the
contracts and numerical acceptance criteria, not exact timing equality.
Portable tests passed: 24 new, 86 retained from prior editions, 110 cumulative.
Source SHA256 accompanies the executed control evidence. Read the book for target
qualification, controlled variations and worked interpretations.
