Historical commands retain their original working paths. Use reader/guide.py for a fresh run and reader/records.py for the corresponding book pages.

CURRENT READER EDITION — VERSION 56
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

SERVING AND DEPLOYMENT — VERSION 13

Chapters 72–75, Exercises 136–151. All target deployment work is inside WSL2.
Start from the spark-book root. Use a virtual environment, install
serving/requirements.txt, and use a fresh output directory for every run.

python -m pip install -r serving/requirements.txt
python serving/reproduce.py --out runs/serving-repeat
python serving/validate.py --evidence runs/serving-repeat

The shipped evidence is EXECUTED on x86-64 Linux CPU, Python 3.12.14,
NumPy 2.3.5, ONNX 1.20.1. It contains real loopback HTTP responses and
complete server logs. Python backend tests use an API double, not Triton.
ONNX reference evaluation is CPU only, not ONNX Runtime or CUDA.
Twenty checks passed; older suites were not rerun.

TARGET_LABS.txt contains all new chapter procedures and acceptance criteria.
docker/BOOTSTRAP.txt contains the native Docker Engine and Container Toolkit
preparation path. The Docker daemon and K3s containerd are separate runtimes.
No Linux display-driver installation is included or required by these labs.

Docker, kubectl, K3s, Helm, nvidia-smi, vLLM and Triton server were absent in
preparation. Their inventory result is retained. Resolve candidate images to
actual ARM64 digests on the laptop. No fake successful deployment output or
invented image digest is supplied. deploy.py generates YAML; it never applies it.
The printed target commands intentionally modify only the named local lab
containers/namespace; installation is an explicit separate preparation step.

server.py is a bounded teaching service, not a public production server. It uses
one CPU NumPy owner, a two-request inference admission bound, sixteen transport
threads, a 4096-byte JSON cap and strict small input limits. No authentication,
TLS, request deduplication or active-kernel cancellation is implemented.
The fixture is not a trained sentence encoder or a semantic quality benchmark.

client.py modes: embed, load, stream, chat-load, triton, triton-load.
Use --model local-model for llama.cpp/vLLM; lab_embedding for Triton CPU;
lab_embedding_gpu for ONNX GPU; lab-embedding for the teaching HTTP service.
Single request and stream failures return nonzero; load modes retain every
outcome for analysis. Never report success-latency quantiles without errors.

The Python/ONNX repositories in evidence are tiny reusable fixture artifacts.
No publisher model weights are bundled. Download Qwen only using the pinned
revision given in the target runbook and retain model hashes and image identity.
Generated manifests use a one-node hostPath for model files, not portable storage.
Read the runbook's permissions, readiness, memory and rollout qualifications.

Kubernetes CPU learning remains useful if the WSL GPU plugin is unsupported.
One GPU owner with bounded requests is the default; no MPS, MIG or time-slicing
configuration is enabled. Tensor parallel size 1 does not emulate multiple GPUs.
