Recorded source paths in this note resolve under the separately restored evidence root.
See reader/EVIDENCE.txt for path lookup.

CURRENT READER EDITION — VERSION 55
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

WSL2 orchestration and decision-model laboratory - Version 24
Chapters 92-96, Exercises 220-234

READ FIRST
TARGET_LABS.txt contains the complete chapter procedures and transcripts.
Run commands from spark-book in the WSL Linux filesystem. The Windows
NVIDIA driver is already mapped; these programs install no display driver.
Create an environment separate from CUDA training. Python 3.12 was used for
preparation, not imposed as a hard requirement. The independent reviewer ran
all fifteen pinned controls on Python 3.13.17, Linux x86-64 CPU. Keep pip check,
the environment record and your own freeze; qualify other interpreter and
native-platform combinations rather than assuming they were executed here:

  python3 -m venv .venv-orchestration
  source .venv-orchestration/bin/activate
  python -m pip install -r orchestration/requirements.txt
  python -m pip check
  python orchestration/environment.py --out runs/orch-environment.json

requirements.txt pins direct dependencies. The preparation freeze records
the Linux amd64 resolution. Native ARM64 installation is a target check;
do not force x86 wheels into an ARM64 interpreter. Retain its own freeze.
installation-preparation.txt is the complete preparation pip output.

REPRODUCE THE CPU CONTROLS
  python catalogs/capture.py --out runs/orchestration-repeat -- \
    python orchestration/reproduce.py --out runs/orchestration-repeat/details

The outer directory and details directory must be new. Outputs are not
silently overwritten. The driver stops at the first failed session, retains
its exit status and leaves all preceding evidence. Individual programs have
--help and can be repeated with fresh output paths.

WHAT ACTUALLY EXECUTES
LangChain: real create_agent, scripted chat test double, bounded tools,
validation errors and graph recursion cutoff.
LangGraph: real SqliteSaver checkpoints, pause, separate-process restart,
wrong-thread/duplicate-resume rejection and nested scripted agent.
LangSmith: actual local trace objects and evaluate(upload_results=False),
explicit offline client metadata and a socket-attempt assertion.
PydanticAI: actual Agent/FunctionModel, invalid output, retries, budget
exhaustion and captured failure messages.
HTTP: real ChatOpenAI and PydanticAI adapters against a loopback fixture,
poisoned proxy control and endpoint validation. No model inference.
Jev: fabricated Choice/Score/Noul response contracts, six rejected faults,
synthetic binary calibration arithmetic and an unsent request plan.

EVIDENCE
evidence/12-final-driver/details contains 15 passing CPU command captures.
evidence/13-http-final rechecks the final required-tool acceptance guard.
evidence/14-budget-final rechecks retention of budget-failure messages.
evidence/final-source-hashes.json identifies the shipped Python sources.
The earlier driver/source-hashes.json preserves the sources at that run;
langchain_lab.py and pydantic_lab.py received the two checked changes above.
The preparation platform was Python 3.12.14, Linux x86-64, CPU only.
No learned model, ARM64, WSL2 GPU, hosted LangSmith or hosted Jev ran here.

Preparation failures are retained:
05 and 06: implicit LangSmith /info discovery was blocked by the offline
connection guard. The final explicit client disables batching and supplies
local info; disabling trace uploads alone was insufficient.
07: a PydanticAI usage property was mistakenly called as an older API method.
08: RunTree default serialization omitted child_runs. The final trace writer
recursively retains children and uses model_dump instead of deprecated json.
01-04 and 09-11 are earlier successful development controls. The PydanticAI
banner in 03-04 is now disabled; it is not deleted from historical captures.
The final HTTP transport warning and evaluator beta warning remain verbatim.

LOCAL MODEL QUALIFICATION
Keep one already-qualified model server as the GPU owner. Run each target
command separately before trying parallel clients. All tools in these new
examples read SIMULATED evidence. Schema validity is not factual validity.

  python orchestration/langchain_lab.py local \
    --base-url http://127.0.0.1:8080/v1 --model local-model \
    --out runs/lc-local.json
  python orchestration/pydantic_lab.py local \
    --base-url http://127.0.0.1:8080/v1 --model local-model \
    --out runs/pydantic-local.json

Only explicit IPv4 loopback HTTP endpoints with /v1 are accepted. Clients
disable inherited proxies and redirects. Model/tool support still depends
on the server, template and model. Plain-text chat success is insufficient.
No fallback to a hosted model is configured. Preserve target failures.
Client timestamps and duration are not kernel timings. Correlate with the
Chapter 91 monitoring window and the CUDA tracing exercises independently.

OPTIONAL HOSTED SERVICES
LangSmith service mode requires --upload-synthetic and LANGSMITH_API_KEY.
Jev hosted mode requires --send-synthetic and TYPESAFE_API_KEY.
Neither mode is invoked by reproduce.py. They send fixed synthetic input
only to their named official endpoint. Never paste keys into a transcript.
Jev's documented API runs remotely; there is no local Jev GPU recipe here.
SDK trace JSON is not a self-hosted LangSmith deployment or UI validation.

Checkpoint files in the evidence are intentional, synthetic teaching state.
They are not GPU memory dumps. Sequential restart does not prove exactly-once
effects or concurrent service correctness. Do not use checkpoints from an
untrusted source as input to application recovery.

VERSION 24 INDEPENDENT REVIEW
The supplied Version 23 review reports no accuracy findings. Its raw Python
3.13.17 reproduction is retained at ../review-v24/reviewer-evidence/
v23-orchestration-reproduce/. All fifteen sessions passed; source-hashes.json
matches all ten current scripts. These are independent reviewer captures,
not replacements for preparation evidence above. No new runtime suite was
executed for this editorial revision.

The Jev response identifies the model used, which may differ from a requested
alias. Compare the resolved version with the pinned expectation; jev-latest
does not establish jev-1.14.0 for this control. The hosted example leaves
TypeSafe-Danger-Zone-Images and TypeSafe-Danger-Zone-Long-Context unset.
Official sources: https://api.typesafe.ai/openapi.json and
https://docs.typesafe.ai/models (consulted 28 October 2026).
