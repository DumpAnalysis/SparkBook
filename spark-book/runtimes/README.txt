Recorded source paths in this note resolve under the separately restored evidence root.
See reader/EVIDENCE.txt for path lookup.

CURRENT READER EDITION — VERSION 55
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

LOCAL RUNTIME AND NEMOTRON LABORATORY — VERSION 27
Chapters 97–100; Exercises 235–250. All target commands run inside WSL2.

QUICK REPRODUCTION
From spark-book/ with Python 3.12 or newer:
  python3 runtimes/reproduce.py --out runs/runtimes-cpu
  python3 runtimes/checks.py
The output directory must be new and inside spark-book/. Paths outside
the tree, including symlink escapes, receive an argparse error (exit 2)
before output is created. An existing in-tree directory is also refused:
the current runner raises ContractError with a traceback (exit 1) and
leaves its contents unchanged. Choose a fresh path for each run; do not
reuse or clear an evidence directory to make the command succeed.
No third-party packages are required for these controls. Preparation used Python 3.12.14,
Linux x86_64. The loopback HTTP exercise needs local socket binding.
No learned runtime, GPU or WSL2 success follows from these control results.

WHAT WAS EXECUTED
Sixteen complete CPU sessions are in evidence/release/235 through evidence/release/250.
The root evidence/ records a refused rerun into occupied artifact paths;
see its ATTEMPT_NOTE.txt. The release/ subtree used the Version 25 sources and a fresh destination; final/
is an earlier successful control run.
Eight regressions passed; regressions.txt preserves their full output.
The controls include real loopback HTTP with scripted response bodies,
stream termination failures, shape and finite-vector checks, dataset overlap,
byte identity binding, index ordering, memory arithmetic and failure counts.
The synthetic responses/metrics are labelled. Public metadata was retrieved
from the official APIs and is retained under metadata/. No new model weights,
Ollama archive, llmster, Unsloth training or Nemotron runtime was executed.

TARGET WORK
TARGET_LABS.txt contains the complete chapter prose, commands and CPU
transcripts. No generated output stands in for a target GPU transcript.
Use catalogs/capture.py for each target command and preserve failures.
Use fresh result directories. Keep all active data in the Linux filesystem.
The Windows NVIDIA driver is already mapped into WSL2: do not install a
Linux display driver. Record the CUDA 13.4 toolkit separately from the
runtime libraries actually loaded by PyTorch or an inference backend.

PROGRAM CONTRACTS
common.py restricts HTTP inference to explicit 127.0.0.1 ports, ignores
inherited proxies, refuses redirects, limits a response to 4 MiB and uses a
120-second socket timeout. This is not a total wall-clock deadline for a
slow trickle response. It implements the data-only SSE/NDJSON subset used
by these probes and rejects incomplete streams; it is not a general SSE SDK.

probe.py retains requests, raw response bytes, inventory, result and failures.
Ollama requests require a caller-recorded digest checked before and after.
OpenAI-compatible requests record reported model names; the operator must
bind those names to the actual loaded model file/runtime. The probe buffers
the stream and reports total client duration, never time to first token.
A length-limited or unknown finish is not labelled a fully completed answer.
It has no bearer-token option and does not silently use a remote endpoint.

acquire.py reads retained full revision IDs and checks sizes plus LFS SHA256
or Git blob identities. Plan mode downloads nothing. --download writes only
to a new directory; partial or mismatched downloads remain for inspection.
Artifact HTTPS redirects are expected for Hugging Face storage. Catalog
identity verifies transferred bytes, not model quality or conversion fidelity.

ollama_archive.py selects the recorded Linux ARM64 or amd64 release asset,
then downloads only on explicit request and verifies its size and SHA256.
It neither extracts the archive nor installs a driver or service.

train_unsloth.py is a target-only native CUDA recipe. Its initial ten-step
scheduler stops at update five and saves optimizer, scheduler and RNG state;
--resume-from continues through update ten. The contract hashes the local
base files and fixes sequence, batch, seed and adapter rank. This is a smoke
training task, not a model-quality benchmark. Loss covers nonpadding prompt
and label tokens. A distinct padding token preserves EOS supervision.
Target execution, package resolution, recovery equivalence and GPU allocation
remain unmeasured. Install Unsloth 2026.10.10 only in a separate candidate
environment; preserve the actual complete dependency freeze and native wheel
identities. The pin is not a claim of compatibility with every ARM64 system.

merge_unsloth.py verifies the same base and adapter path, merges locally and
hashes the result. It does not download a conversion engine. Perform the
separate pinned llama.cpp conversion and quantization only after comparing
held-out quality and checking temporary memory/disk requirements.

PRESERVED IDENTITIES
Qwen base commit: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
NVIDIA Nemotron upstream metadata commit:
  bf77c3174f68ad409e1c2aa60daeb46e32d1c606
Selected ggml-org Nemotron GGUF conversion commit:
  fe801509a580ff84a5728d4a9104e3f6f7e9b0dd
Ollama candidate release: v0.40.11
Unsloth candidate package: 2026.10.10
LM Studio installer: capture bytes/hash/version on the target; no frozen
llmster or backend installation was prepared here.

MEMORY DISCIPLINE
The observed Q8 file is 33,585,500,064 bytes, greater than 30 GiB even before
runtime workspace. Q4 is 22,421,827,488 bytes. File size is not peak allocation.
The 40 GiB guest-available and 6 GiB reserve values in Exercise 248 are invented
arithmetic inputs, not WSL settings or laptop measurements. Measure host and
guest headroom, CUDA free bytes and paging before admitting a large model.
Do not add the reported GPU figure to 64 GB installed unified memory.

SMALL CHAT BASELINE
acquire.py chat-gguf selects the 1,117,320,736-byte Qwen 1.5B Q4_K_M file
at commit 91cad51170dc346986eccefdc2dd33a9da36ead9. Its exact SHA256 is
6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e.
Only its public metadata and no-download plan were retrieved in preparation.

VERSION 26 REVIEW CORRECTIONS
The reproduction runner now reports a clear argparse error for an output
path outside spark-book, before creating it. review-v26/baseline-runtimes/
contains the original reproduce.py for checking historical source hashes.
All other runtime Python sources and historical captures are unchanged.
New CPU captures and path-guard checks are in review-v26/evidence/.
TARGET_LABS.txt incorporates the corrected Nemotron build chronology and
the explicit reason for keeping the conversion revision pinned.

VERSION 27 REVIEW CLOSURE
The Version 26 review closes all accuracy findings. Its supplied independent
CPU captures are preserved under review-v27/reviewer-evidence/: eight
regressions, three output-path checks and all sixteen controls passed on
Python 3.13.17, Linux x86-64. All nine runtime source hashes match this edition.
These are reviewer results, not new preparation runs or WSL2 GPU evidence.
No executable source or historical capture changed. The chronology citation
in TARGET_LABS.txt now includes reference 11, the dated release page.
