Local AI on RTX Spark Laptops
Learner companion Version 55 — 31 October 2026

START HERE
Read reader/START_HERE.txt. Extract this learner ZIP into a staging directory
in WSL2, then create a new working project from the extracted spark-book/:

python3 reader/new_lab.py ~/work/spark-book

Choose a new sibling path if that destination already exists. Preserve an
existing qualified environment, downloaded models and personal run records.

cd ~/work/spark-book
python3 reader/preflight.py --require-wsl
python3 -m venv .venv-exercises
source reader/activate.sh
source reader/new_run.sh
python reader/guide.py 1

The first-inference route is Exercises 1–9, then 23. It uses the native
runtime and does not require PyTorch. Exercise 14 qualifies the framework
before its routes; Exercise 251 remains an advanced memory investigation.

PACKAGES
This learner package contains teaching programs, required small fixtures,
dependency records, 251 standalone guides and the current evidence map.
Model weights, containers, environments and user runs are not included.

The separate Evidence Version 48 ZIP contains the supplied execution records.
Extract it into staging and run its restore.py into a NEW evidence directory.
Set SPARK_BOOK_EVIDENCE to that restored directory when inspecting captures.
The optional Publisher Version 55 ZIP holds old inventories, reviews and
publication history. It is not needed for the first-inference route.

Read reader/EVIDENCE.txt for restoration and path examples. Use python reader/records.py NUMBER
to inspect an exercise’s records, adding --files to locate restored files.
The helper prints information and executes no experiment.
The book retains complete selected transcripts; all supplied execution files are
preserved in the evidence pack. Historical absolute paths remain historical.

VALIDATION
python3 reader/verify_package.py
VERSION55_MANIFEST.json defines this learner distribution. The source
inventory used by new_lab.py is reader/workspace-files.json. Run package
verification on a fresh extraction before creating personal files.
TRANSCRIPT_POLICY.txt is the current policy; old amendments are archived.
No new laptop execution is claimed by an editorial publication version.

Version 55 pairs this learner guide with the Version 55 manuscript. The recorded inputs are in Evidence Version 48. Each timing is for its named captured command; preparation, reading and later steps take additional time.

Verify a pristine extraction and its matching unedited Word manuscript together:
  python3 reader/verify_package.py --book /path/to/Local_AI_on_RTX_Spark_Laptops_Version55.docx
Use a manuscript stored outside the extracted package. Personal edits change its checksum.
