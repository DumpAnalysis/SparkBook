Recorded source paths in this note resolve under the separately restored evidence root.
See reader/EVIDENCE.txt for path lookup.

CURRENT READER EDITION — VERSION 55
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

Version 17 media laboratories (introduced in Version 16)
Chapters 81-88; Exercises 172-203

Read TARGET_LABS.txt for the complete chapter text, commands, acceptance
criteria, and printed transcripts. Run from the spark-book directory.
All laptop procedures use ARM64 Linux inside WSL2. No Linux display driver
is installed; the Windows NVIDIA driver remains mapped into the guest.

Environment
Use a fresh Python 3.12 environment and first qualify the CUDA PyTorch and
torchvision pair in Exercise 172. Install requirements.txt in that environment.
The core Python versions match the CPU speech controls. Diffusers, Accelerate,
torchvision, and their GPU execution were not qualified in preparation.
The selected versions explain these APIs; they are not latest-version advice.
Record pip check and pip freeze before model work.

Model keys in models.lock.json
  qwen-vl   Qwen/Qwen2.5-VL-3B-Instruct
  sdxl      stabilityai/stable-diffusion-xl-base-1.0
  musicgen  facebook/musicgen-small
  mms       facebook/mms-tts-eng
  whisper   openai/whisper-tiny
Each entry contains an immutable publisher revision and download patterns.
Run: python media/assets.py KEY
Then: python media/assets.py KEY --verify
The normal root is models/media. To reproduce the printed CPU speech commands,
use --root ../models for mms and whisper, then use --device cpu.
Download only the next required checkpoint. Publisher weights are not in this
archive. The tiny original 2D teaching denoiser is included.

Retained execution
controls/ contains actual CPU fixtures, trained diffusion samples, MMS speech,
Whisper hypotheses, scoring controls, the one-owner simulation and audio plots.
The final 25 regression checks are in 10-tests-final/transcript.txt.
05-tests preserves the earlier 24-check run; do not add the counts together.
Ten final command transcripts are printed in the book. All eleven captured
preparation commands, including that interim test run, remain available.
The prepared speech output is synthetic speech, not a recording of the author.
Whisper returned a word over digital silence; that result is intentionally
retained. Output fluency does not prove supporting acoustic evidence.

Reproduction writes new directories. Do not overwrite controls/.
  python media/controls.py fixtures --out runs/media/fixtures-new
  python media/diffusion_control.py --out runs/media/diffusion-new
  python media/validate.py
  python media/targets.py tts --model models/media/mms --device cpu \
    --out runs/media/tts-cpu-new

Evidence limits
No Qwen-VL, SDXL, MusicGen, adaptation, GPU, Docker, Kubernetes, or complete
voice-pipeline success was executed during this preparation. The source paths,
command construction, bounded inputs and public APIs were inspected; local
qualification is still required. CPU preparation used x86-64 Linux, not the
example ARM64 laptop. Existing book evidence is retained unchanged.

Observability
targets.py records model/input identity, processor properties where applicable,
phase wall times and, for CUDA, allocator peaks and free/total memory. --profile
adds a Chrome trace; --snapshot uses version-sensitive PyTorch allocation-history
APIs. An allocator snapshot is not a raw GPU-memory dump. --trace-steps records
diffusion latent diagnostics and synchronizes the run; compare performance only
with a matching uninstrumented workload. CPU offload consumes shared host memory.

Training and serving
adapt_asr.py trains only the final Whisper decoder block and checks frozen
weights. prepare_adaptation.py creates small explicit teaching manifests.
The SDXL LoRA exercise uses a pinned upstream training script; image_adapter.py
compares the resulting adapter with its base. Neither two examples nor a few
optimizer steps establish useful generalization. The saved ASR optimizer is not
a complete resumable checkpoint contract.
voice.py joins ASR, a loopback text endpoint and TTS without executing commands
from model output. Its child models are serial, but an independent text server
remains resident. Digital silence is rejected; noisy non-speech still needs a
qualified voice-activity detector. Read the model licenses before deployment.

Version 17 corrections and evidence continuity
  voice.py uses an explicit proxy-free HTTP opener and refuses redirects.
  A live local HTTP regression with an unreachable proxy confirms the POST
  stays local; this is not an end-to-end ASR/LLM/TTS run.
  targets.py uses dtype="auto" for Qwen-VL and retains actual model and
  parameter dtypes plus processor min/max pixels. Its 256*28*28 floor is
  runner policy; the pinned checkpoint defaults to 56*56. API-double checks
  exercise the record contract, not Qwen inference or GPU compatibility.
  controls/preparation.json is the original Version 16 record. Its two
  changed source files are in review-v17/baseline-media/; all other source
  hashes still refer to the unchanged media files. Do not treat that
  historical record as a hash manifest for the current runner.
  See review-v17/RESPONSE.txt and evidence/ for the new checks and limits.
