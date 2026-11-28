# Local AI on RTX Spark Laptops

Version 59 of the learner companion supplies the code, fixtures and guides for all 251 exercises in *Local AI on RTX Spark Laptops*. The worked system is a 64 GB unified-memory RTX Spark laptop with the Windows NVIDIA driver mapped into WSL2.

## Start in WSL2

Extract the matching Version 59 release in WSL2, or clone the repository when its README identifies Version 59. From the release’s spark-book directory, run the integrity check before creating a working lab. Add `--book` for each edition file you have. The manifest checks both the DOCX and print PDF by SHA-256.

```bash
mkdir -p ~/work
git clone https://github.com/DumpAnalysis/SparkBook.git ~/work/SparkBook
cd ~/work/SparkBook/spark-book
python3 reader/verify_package.py
# Optional paired check, using your local book files
# python3 reader/verify_package.py --book /path/to/book.docx --book /path/to/book.pdf
python3 reader/guide.py 1
```

For a first installation, create your working project under a **new** directory. If you already use `~/work/spark-book`, choose a fresh sibling path in both commands.

```bash
python3 reader/new_lab.py ~/work/spark-book
cd ~/work/spark-book
python3 reader/preflight.py --require-wsl
```

Follow [Start here](spark-book/reader/START_HERE.txt) and [Fresh setup](spark-book/reader/FRESH_SETUP.txt) to prepare the environment. Exercises 1–9 lead to the first native model run. Exercise 23 records its acceptance result. Each later exercise identifies the packages, model or image downloads, hardware, and service credentials it needs.

## New to a technology

Every numbered exercise guide includes its background before the procedure. For example, Exercise 1 introduces Bash and the WSL2 guest. Exercise 151 explains how to distinguish a served model from the backend that executed it. You can also browse [Technology overviews](spark-book/reader/TECHNOLOGY_OVERVIEWS.md) or look up a topic in your working lab.

```bash
python3 reader/technologies.py 1
python3 reader/technologies.py --find "Triton"
python3 reader/technologies.py --list
```

## Find a procedure and read its transcript

From the working lab, use the same exercise number as the book. The lookup shows the printed page, execution scope, outcome and files you can open locally.

```bash
python3 reader/guide.py --list
python3 reader/guide.py 129
python3 reader/records.py 129 --files
```

The companion includes the 483 full published transcripts. Export them as individual UTF-8 files from the working lab. To compare directly with your Word edition, supply its filename to the exporter and choose a different output directory. The exporter verifies the same 483 checksums for either source. [Transcript guide](spark-book/reader/EVIDENCE.txt)

```bash
python3 reader/import_transcripts.py
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

### Version 59 illustrated revision 4

Exercises 167 and 171 now run a direct prompt and a three-question, textbook-assisted worksheet on the same cases and model weights. The worksheet records the selected reading, exact prompts, raw replies and finish reasons. Score the final answers using the original five items in `spark-book/reader/ANSWER_RUBRICS.txt`. The paired DOCX and PDF hashes and page index are in the Version 59 manifest.

### Earlier illustrated revision

Exercise 11 now stops the Q4 owner before preparing Q8 and restores Q4 after the matched requests. For Nsight Systems 2026.5.1, start a lab run and follow Exercises 39–40. The digest-pinned ARM64 setup, capture and same-version export are in `spark-book/reader/FRESH_SETUP.txt`. The revised scoring examples are in `spark-book/reader/ANSWER_RUBRICS.txt`.

### Version 59 illustrated revision 5

Developed with AI assistance (GPT-6.2 Astra and Claude Fable 5.2)

The [chapter self-check key](spark-book/reader/SELF_CHECKS.txt) covers 200 questions. [Exercise 186](spark-book/reader/exercises/186.txt) compares three VAE settings on matching latents and records each decoder call. Exercises 39–40 identify the historical and reader Nsight Systems results, Exercise 104 separates the two batch configurations, and Exercise 200 identifies the Whisper parameters being fine-tuned.

### Version 59 illustrated revision 6

The [chapter self-check key](spark-book/reader/SELF_CHECKS.txt) gives the corrected exercise sources. The dated reader records for [Exercise 39](spark-book/reader/reader-runs/039.txt), [Exercise 40](spark-book/reader/reader-runs/040.txt), [Exercise 129](spark-book/reader/reader-runs/129.txt), and [Exercise 186](spark-book/reader/reader-runs/186.txt) include the commands, outcomes and artifact identities behind the book’s reader results.

### Version 59 illustrated revision 7

The technology guide routes [custom-operator streams](spark-book/reader/TECHNOLOGY_OVERVIEWS.md#topic-streams) to Exercise 58, [CUDA events and double buffering](spark-book/reader/TECHNOLOGY_OVERVIEWS.md#topic-buffers) to Exercise 63, and [CUDA Graphs](spark-book/reader/TECHNOLOGY_OVERVIEWS.md#topic-graphs) to Exercise 65. The reader records identify the dated runs behind the printed results.

### Version 59 illustrated revision 8

Exercises 143, 169 and 206 clarify how to diagnose container memory pressure and interpret the 24 GiB NeMo and 12 GiB NGC limits alongside CUDA device measurements.
