Historical commands retain their original working paths. Use reader/guide.py for a fresh run and reader/records.py for the corresponding book pages.

CURRENT READER EDITION — VERSION 56
Start with reader/START_HERE.txt at the companion root. Use
reader/guide.py NUMBER for the portable entry procedure and the book for
current verified outcomes. Source-era environment/transcript statements below
remain historical context; they are not the current qualification status.
Use new run directories and preserve the qualified environment and assets.

VERSION 5 EMBEDDING LABORATORY — Chapters 49–56, Exercises 75–90
Run commands from the spark-book directory in WSL2 Linux.
The Windows NVIDIA driver is already mapped; do not install a guest display driver.

Start with embeddings/requirements.txt in a separate Python 3.12 environment.
Install and qualify torch separately as described in Chapters 13, 21 and 49.
These exact top-level analysis/training versions were used in CPU preparation;
ARM64 wheel availability and CUDA execution must be qualified on the laptop.

To reproduce the publisher snapshot:
  mkdir -p embeddings/models/minilm
  cp embeddings/evidence/minilm-source.json embeddings/models/minilm/book-source.json
  python embeddings/fetch_model.py
  python embeddings/encode.py --model embeddings/models/minilm --out embeddings/runs/local-base
  python embeddings/project.py --input embeddings/runs/local-base --out embeddings/runs/local-maps

Frozen reference outputs remain in embeddings/runs/base, maps, maps-alt,
custom-before, custom-after and tuned. They were generated on a CPU, not on the
example RTX Spark laptop. Their JSON reports retain hashes, versions and parameters.
Original full console transcripts are in embeddings/evidence/*.txt.
The exit marker is added by the capture harness. No publisher weights are bundled.

The tiny selected scratch BERT and custom sentence encoder ARE bundled under
embeddings/models/tiny-bert and custom-sentence so their tokenizer IDs and
recorded vector space can be inspected exactly. These are original teaching
artifacts trained on the included synthetic diagnostic prose. They are weak
models and must not be treated as general-purpose support systems.

Programs:
  data.py: original 18-family corpus and leakage checks
  common.py: NumPy pooling, normalization, metrics and digest helpers
  fetch_model.py: publisher commit resolution and local download
  encode.py: local model encoding and original-space retrieval evaluation
  project.py: PCA, t-SNE, UMAP, metrics, coordinate files, training-only transforms
  train_bert.py: WordPiece + scratch BERT MLM, or --init for continued MLM
  inspect_bert.py: tokenizer inspection and one-mask prediction
  finetune.py: contrastive optimization; --bert wraps a custom BERT
  diagnose.py: finite-vector, variance, cosine and principal-direction report
  search.py: exact index with model/document/vector integrity checks
  pooling_demo.py and heldout.py: small executable demonstrations
  gpu_session.sh: qualified WSL2 target workflow; not run during preparation
  validate_embeddings.py: twelve portable checks

Training examples (new destinations preserve the bundled reference models):
  python embeddings/train_bert.py --out embeddings/models/local-bert --steps 60
  python embeddings/finetune.py --model embeddings/models/local-bert --bert --out embeddings/models/local-sentence --lr 0.0001
  python embeddings/finetune.py --model embeddings/models/minilm --out embeddings/models/local-minilm-tuned
  python embeddings/finetune.py --model embeddings/models/minilm --out embeddings/models/local-wrong --wrong-pairs

Models are selected by validation loss or validation MRR, with the initialization
also eligible. The final test is evaluated after selection. Using test scores to
repeat model selection invalidates that test. There are only six queries in each
held-out retrieval partition; the wrong-pair control demonstrates how inadequate
that size can be. No quality improvement is guaranteed.

All training checkpoints here are inference artifacts, not restart snapshots.
Chapter 45 supplies optimizer/RNG-aware recovery material. Training uses float32
for a transparent baseline; mixed precision and cached contrastive loss are
separate experiments that require evaluation and memory measurement.

Use .npy and .npz with allow_pickle=False. Projection fits do not consume labels.
Maps are exploratory; retrieval always uses the original vector space.
No CUDA timings, peak capacity, or WSL2 GPU success is claimed by the CPU evidence.
