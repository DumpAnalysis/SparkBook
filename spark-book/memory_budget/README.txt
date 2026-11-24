Exercises 29, 30 and 251 • WSL2 memory experiments

Start in your working spark-book directory. Activate the same Python environment
that passed the synchronized CUDA check in Exercise 14. A fresh installation may
call it .venv-framework-candidate. The recorded laptop used .venv-exercises.
Check the interpreter before doing more work.

  python -c "import sys,torch; print(sys.executable,torch.cuda.is_available())"
  source reader/new_run.sh

The commands below use a fresh LAB_RUN. Every stage directory must be new.
Stop other workloads that you own and save your work before a large allocation.
Windows PowerShell interoperability, wslpath, vmstat and a mapped CUDA device
are required for supervised stages. The supervisor uses read-only Windows CIM
counters. Preserve a BLOCKED result if the sample or a prerequisite is unavailable.

Exercises 29–30 • A model larger than the dedicated segment

  python large_model.py

The offline plan lists the revision and total bytes before any acquisition.
After checking disk space, obtain the nine verified shards (about 32.43 GiB).

  python large_model.py --download

Use the native runtime acquired in Exercise 4. The following stage rehashes the
shards and pinned native executable and libraries, checks current headroom, and
serves seven local requests with the original 51-of-65-layer placement profile.

  python memory_budget/guarded.py model --out "$LAB_RUN/model32"
  python memory_budget/guarded.py report --run "$LAB_RUN/model32"

Model admission requires the profile's CUDA and guest allowances. Host physical
headroom must cover the planned excess over dedicated free memory plus 3 GiB.
Initial Windows commit headroom must be at least 4 GiB. During the stage the
Windows available-physical and commit-headroom floors are each 2 GiB. Admission
may stop on a 16 GiB guest because current available RAM also matters. Never
adjust the pagefile or retry automatically to reproduce a historical outcome.

For Exercise 30, read the pressure request's UTC interval in the report and
pressure.json. Inspect host.jsonl and UTC vmstat.txt over the same interval.
Ignore vmstat's first, since-boot rate row. Retain admission.json,
native-stderr.txt, requests.json, native-exit.json and result.json. Host
physical headroom, host commit, guest swap activity and CUDA free bytes have
different scopes. Seven responses establish completion for this one load.

Exercise 251 • A bounded allocation beyond guest MemTotal

  python memory_budget/run.py --out "$LAB_RUN/memory-views"

This query initializes the CUDA context and records guest, management, CUDA
and framework views. It writes no allocation-test tensors. A budget reported
by an interface does not establish which bytes a program can write and verify.

  python memory_budget/guarded.py pilot --out "$LAB_RUN/memory-pilot"

Inspect result.json, the verified byte count in stdout.txt, the after_release
event, host.jsonl, vmstat.txt and the worker exit. Continue only with a passed
1 GiB pilot from the same WSL boot and Python environment. The next stage
retains up to eighteen 1 GiB blocks at once and may exert significant pressure.

  python memory_budget/guarded.py target --pilot "$LAB_RUN/memory-pilot" --out "$LAB_RUN/memory-above-guest"

Each allocation stage requires fresh host counters before launch and while it
runs. The host physical and commit floors are each 4 GiB. The guest floor is
2 GiB available, and each new CUDA block must leave a 4 GiB reserve. The
supervisor takes at least three valid baseline readings and stops on the first
bad reading, a query over five seconds, missing data for eight seconds, or a
180-second workload deadline. Stop with Ctrl+C if necessary. Directly calling
the inner allocation probe skips the Windows supervisor.

Each block is filled, retained and checked across every byte, in small tiles.
Require the final result, verified byte count, after_release event and clean
worker exit. BLOCKED and FAILED are valid observations. A completed run gives
a lower bound for that allocation path. It cannot establish maximum RAM,
residency, process-wide GPU usage or whether an arbitrary model will fit.

Read the historical outcomes with python reader/records.py 29, 30 or 251.
The former model supervisor tolerated two consecutive low host readings.
The portable procedure stops on the first invalid or low reading. These
procedures have CPU control checks. Their new outputs belong to your own
run. observations/windows-allocation.json contains the original time-stamped
pilot and target memory samples, with field names normalized for replay.

  python memory_budget/test_guarded.py
  python memory_budget/test_controls.py
