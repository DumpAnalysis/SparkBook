Local AI on RTX Spark Laptops
Learner companion Version 56 — 31 October 2026

START HERE
Read reader/START_HERE.txt. Clone https://github.com/DumpAnalysis/SparkBook or unzip the companion
in a staging directory in WSL2. Open spark-book/ and create the working lab:

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

MATERIALS
The working programs, 251 exercise guides, test data, training checkpoints and
module-specific dependency lists are in the spark-book/ directory. The public
repository is https://github.com/DumpAnalysis/SparkBook. Use the exercise's
instructions to acquire its model weights, runtime or container image.

Run python3 reader/records.py NUMBER --files to find the guide and printed run
pages. To read the printed commands and results in local text files, run
python3 reader/import_transcripts.py /path/to/book.docx from your working lab.
See reader/EVIDENCE.txt for the output location and scope.

VALIDATION
Run python3 reader/verify_package.py in a pristine checkout or extraction.
Add --book /path/to/book.docx to verify the matching manuscript.
The portable tests and the recorded laptop trials have separate scopes.
