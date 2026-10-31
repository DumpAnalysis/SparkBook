# Local AI on RTX Spark Laptops

This is the Version 55 learner companion to *Local AI on RTX Spark Laptops*. It contains code, fixtures, and guides for 251 hands-on WSL2 exercises. The worked system is a 64 GB unified-memory RTX Spark laptop using the NVIDIA driver mapped into WSL2 from Windows.

## Start in WSL2

Clone the repository into a staging directory, then verify the companion against its manifest.

```bash
mkdir -p ~/work
git clone https://github.com/DumpAnalysis/SparkBook.git ~/work/SparkBook
cd ~/work/SparkBook/spark-book
python3 reader/verify_package.py
python3 reader/guide.py 1
```

For a first installation, create a separate working lab. The destination must be new. If you already have a qualified lab at `~/work/spark-book`, keep it and choose a fresh sibling directory for these commands.

```bash
python3 reader/new_lab.py ~/work/spark-book
cd ~/work/spark-book
python3 reader/preflight.py --require-wsl
```

Follow the setup guide to prepare the exercise environment. Start with Exercises 1–9, then use Exercise 23 to record the acceptance result for your first inference run. [Start here](spark-book/reader/START_HERE.txt) · [Fresh setup](spark-book/reader/FRESH_SETUP.txt)

## Find an exercise or a recorded run

The guides cover inference, fine-tuning, embeddings, multimodal generation, CUDA programming, GPU diagnostics, observability, and deployment. Each exercise number matches its entry in the book. To see the learning routes, browse the index, or inspect the recorded scope of distributed training, use the following commands from the companion or working lab.

```bash
cat reader/ROUTES.txt
python3 reader/guide.py --list
python3 reader/guide.py 129
python3 reader/records.py 129
```

The record index identifies what ran on the laptop and which checks passed, failed, or remain blocked. For original command metadata, stdout, and stderr, restore the separate Evidence Version 48 pack and use `reader/records.py NUMBER --files`. [Evidence guide](spark-book/reader/EVIDENCE.txt) · [Exercise guides](spark-book/reader/exercises)

## Public companion contents

All 251 guides and the exercise programs match the reviewed Version 55 companion. The public copy omits the historical NGC image-configuration file, `catalogs/controls/ngc-image/config.json`, which contains upstream container build history. The retained `image.json`, OCI index, and manifest provide the image identity and ARM64 platform used by the offline catalog exercises. Three saved PyTorch checkpoint files from the distributed and GPU-sharing controls are also omitted. The checkpoint-specific assertions in `distributed/validate.py` and `sharing/validate.py` require those original fixtures. The [distributed](spark-book/distributed/README.txt) and [sharing](spark-book/sharing/README.txt) guides describe how to run fresh recovery and training experiments. The two package inventories reflect the public file set.

## License

The original SparkBook materials use the [PolyForm Strict License 1.0.0](LICENSE.md). Third-party components and models retain their own licenses.
