# Local AI on RTX Spark Laptops

Version 56 of the learner companion supplies the code, fixtures and guides for all 251 exercises in *Local AI on RTX Spark Laptops*. The worked system is a 64 GB unified-memory RTX Spark laptop with the Windows NVIDIA driver mapped into WSL2.

## Start in WSL2

Clone the repository into a staging directory. Run the integrity check before creating a working lab.

```bash
mkdir -p ~/work
git clone https://github.com/DumpAnalysis/SparkBook.git ~/work/SparkBook
cd ~/work/SparkBook/spark-book
python3 reader/verify_package.py
python3 reader/guide.py 1
```

For a first installation, create your working project under a **new** directory. If you already use `~/work/spark-book`, choose a fresh sibling path in both commands.

```bash
python3 reader/new_lab.py ~/work/spark-book
cd ~/work/spark-book
python3 reader/preflight.py --require-wsl
```

Follow [Start here](spark-book/reader/START_HERE.txt) and [Fresh setup](spark-book/reader/FRESH_SETUP.txt) to prepare the environment. Exercises 1–9 lead to the first native model run. Exercise 23 records its acceptance result. Each later exercise identifies the packages, model or image downloads, hardware, and service credentials it needs.

## Find a procedure and read its transcript

From the working lab, use the same exercise number as the book. The lookup shows the printed page, execution scope, outcome and files you can open locally.

```bash
python3 reader/guide.py --list
python3 reader/guide.py 129
python3 reader/records.py 129 --files
```

Readers with the Word edition can also export the 483 published transcripts to `runs/book-transcripts/`. Supply the path to your DOCX. The exporter checks each transcript before writing it. Readers using the printed book or PDF can use the page numbers directly. [Transcript guide](spark-book/reader/EVIDENCE.txt)

```bash
python3 reader/import_transcripts.py /path/to/Local_AI_on_RTX_Spark_Laptops_Version56.docx
python3 reader/records.py 129 --files
```

## Checkpoint and tool exercises

The distributed-training fixtures include the rank-zero checkpoints after steps 2 and 4. The GPU-sharing training fixture includes the single-owner checkpoint after step 4. In the exercise Python environment with PyTorch and NumPy available, check the saved states from the working lab.

```bash
python distributed/validate.py
python sharing/validate.py
```

The [CUDA timestamp probe](spark-book/cuda/timestamp_probe) and [NeMo recovery hooks](spark-book/ecosystem/nemo_hooks) accompany Exercises 151 and 170. The [learning routes](spark-book/reader/ROUTES.txt) describe where to start for other topics.

## License

The original SparkBook materials use the [PolyForm Strict License 1.0.0](LICENSE.md). Third-party components and models retain their own licenses.
