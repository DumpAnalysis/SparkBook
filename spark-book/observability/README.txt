Historical commands retain their original working paths. Use reader/guide.py for a fresh run and reader/records.py for the corresponding book pages.

CURRENT READER EDITION — VERSION 56
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

Prometheus and Grafana laboratory - Version 22
Chapter 91, Exercises 212-219. Target work stays inside WSL2.

Read TARGET_LABS.txt for full procedures, acceptance tests and transcripts.
Run from spark-book unless a command explicitly changes directory.
Acquisition requires Python 3.12+ and dpkg-deb; no new Python package is needed.

The recorded workflows are available in the current learner guides. Exercise 214 runs the strict integration, Exercise 215 renders all three dashboard states, and Exercise 218 measures the qualified Triton service with and without monitoring.

MANUAL STACK WALKTHROUGH
python3 observability/setup_tools.py --out "$HOME/spark-observability-tools"
python3 observability/test_exporter.py
python3 observability/fixture.py --out runs/obs-fixture.json --case baseline
python3 observability/stack.py --tools "$HOME/spark-observability-tools/tools.json" --run-dir runs/obs-01 --fixture runs/obs-fixture.json

Second WSL terminal:
python3 observability/client.py targets
python3 observability/client.py grafana --run-dir runs/obs-01

Open http://127.0.0.1:3000 and sign in as admin using the password in
runs/obs-01/grafana-password.txt. Do not publish that file. Ctrl-C stops the
foreground stack and retains data/logs. Each new stack needs a new directory.
Ports: 9400 exporter, 8089 synthetic CPU work, 9090 Prometheus, 3000 Grafana.
All listeners bind 127.0.0.1. Stop conflicts; do not run two stacks at once.

LIVE MODE
Stop the fixture stack, then:
python3 observability/stack.py --tools "$HOME/spark-observability-tools/tools.json" --run-dir runs/obs-live-01 --live --gpu 0

Live mode requires a WSL kernel and a qualified mapped-driver GPU UUID.
It never replaces a failed live reading with a fixture. Select source=wsl
in Grafana. Unsupported fields stay absent; zero is a distinct valid value.
No Linux display driver is installed. The Windows NVIDIA driver is mapped.
The ~30 GB device-reported total is not an extra pool added to 64 GB host memory.

AUTOMATED QUALIFICATION (after stopping the interactive stack)
python3 observability/integration.py --tools "$HOME/spark-observability-tools/tools.json" --run-dir runs/obs-check-stack --evidence runs/obs-check-evidence

The default command requires Grafana's query backend and a proxied query to
work. --allow-unavailable-grafana-backend deliberately permits a partial CPU
control, records UNAVAILABLE, and never qualifies a working dashboard.

FILES
setup_tools.py: pinned native releases, hashes, extraction and version receipt.
exporter.py / fixture.py: WSL fields and explicit SIMULATED fault cases.
configure.py / dashboard.json: authoritative generator and importable dashboard.
stack.py: foreground process owner, cleanup and private generated password.
rules.yml / rules-test.yml: one recording rule, five alerts, six test scenarios.
workload.py / client.py: synthetic CPU HTTP work, histogram and API commands.
attach.py / queries.txt: already-qualified inference endpoints and expressions.
docker_config.py: digest-only Compose files for Docker in the same WSL distro.
test_exporter.py / integration.py: contracts and native process/API controls.
TARGET_LABS.txt / sources.json: full chapter procedures and official sources.
evidence/: exact captures, failed attempts, raw API results and process logs.

VALIDATION BOUNDARY
Sixteen regression tests and six rule scenarios passed. Native amd64
Prometheus 3.15.1 collected fixtures and evaluated queries/alerts. Grafana OSS
13.3.0 database health, data source and 13-panel dashboard provisioning passed.
Its query backend was UNAVAILABLE: the preparation sandbox denied plugin
Unix sockets. No working chart, ARM64, live WSL2/GPU, Docker or Kubernetes
monitoring run is claimed. Preserve failures; do not disable signature checks.
GPU fixtures and synthetic CPU responses are not GPU/model benchmarks.
DCGM requires separate support qualification. Alerts do not send notifications.

RETENTION
GPU raw logs rotate at 1 MiB with three backups. Prometheus retains 24h /
500 MiB of blocks; head/WAL and other files need additional disk space.
Native service logs and databases need reviewed retention during longer runs.
Downloaded tools, runtime databases and passwords are excluded from the ZIP.

VERSION 24 REVIEW NOTE
The raw strict Grafana integration logs, supplied with the Version 23 review,
are retained in ../review-v24/reviewer-evidence/v22-grafana-strict-run/.
They record successful datasource health and one proxied series on x86-64
Linux where Unix sockets were allowed. This is independent reviewer execution,
not a preparation run, rendered-dashboard check or WSL2 GPU qualification.
The reused driver's embedded "CPU preparation" label is preserved verbatim;
the evidence provenance is independent review. Historical UNAVAILABLE captures
and the earlier review response are unchanged. See review-v24/RESPONSE.txt
and Appendix E.
