# Missing CUDA activity timestamps on RTX Spark N1X / ARM64 WSL2

Prepared from actual local execution on2026-10-30 UTC. This report has not been submitted to any vendor.

## Environment

- NVIDIA RTX Spark N1X6144-core Blackwell GPU, compute target sm_121.
- Windows11 Pro ARM64, build28000; Windows GPU driver32.0.16.1742; nvidia-smi reports KMD617.42 and CUDA UMD compatibility13.4.
- Ubuntu26.04.1 ARM64, Linux6.18.54.1-microsoft-standard-WSL2. Loaded CUDA driver is /usr/lib/wsl/lib/libcuda.so.1.
- Installed NVCC13.4.92; CUPTI reported13.4.92 in version.json, API130401, library libcupti.so.2026.3.1, SHA25694f706383b9f72807bbac02f12a3e33b0c793849ebade118cfca3ce539187bb5.
- Nsight Systems2026.3.2.476. Python3.14.4, PyTorch2.14.1+cu130, framework CUDA13.0.
- Host installed64GiB, WSL guest16,854,429,696bytes, NVML31,232MiB, CUDA budget49,357,000,704bytes are overlapping scopes, not additive physical pools. Tiny probes passed all allocations and numerical assertions.

## Reproduction and observations

Run reproduce_timestamps.py as described in README.md. It launches two kernels, synchronizes, copies back verified values, waits and flushes completed CUPTI records. Expected: nonzero kernel start and end>start. Actual: named/correlated kernel records have start=end=0, despite successful CUPTI activity calls and zero dropped records. The arithmetic checksum is16,777,216; CUDA-event durations are positive; clock64 records at least1million ticks.

The unchanged installed activity_trace_async sample reproduces zero kernel **and memcpy** timestamps with subscriber-scoped V2 APIs and state callbacks. Runtime/driver API timestamps are populated. Vendor exit0 therefore does not establish valid timing.

Across18 direct matrix probes, all GPU numerical and independent event-clock checks passed and no usable kernel timeline was produced. Two additional runs of the standalone resource reproduce the failure. The latency-timestamp mode delivered no kernel records; the other direct modes delivered zero timestamps. Three Torch probes have correct GPU results and zero kernel events. A fresh Nsight software trace has runtime/synchronization events but no kernel table. Full traces, SQLite, raw buffers rendered by the sample, stdout/stderr and exact commands are in the export.

## Tested hypotheses

| Hypothesis / experiment | Actual result |
|---|---|
| Flush before/after free, forced vs normal flush,500ms wait | No valid kernel timestamps |
| Warm context, serial vs concurrent collection, latency timestamps | No valid kernel timestamps |
| CLOCK_MONOTONIC_RAW, CLOCK_MONOTONIC, CLOCK_REALTIME timestamp callbacks | No valid kernel timestamps |
| Hardware event collection | API succeeds; kernel timestamps remain zero |
| Isolated publisher CUPTI13.4.58 | Verified native linkage; no improvement; older comparison, not upgrade |
| Fresh same-user WSL login, Seccomp0/NoNewPrivs0 | Native/Torch/Nsight timing still unavailable |
| WSL root, including NVIDIA sample | Same zero timestamps; not equivalent to Windows administrator context |
| CUPTI activity support query | SUCCESS and supported=1 |
| CUPTI counter profiler initialization/support query | CUPTI_ERROR_NOT_INITIALIZED, including root; returned support fields remain unknown |

Nsight describes the device as Tegra iGPU in a unified-memory tracing diagnostic. This prompted the root experiment; that message alone does not establish that kernel tracing is unsupported. The host HwSchMode registry value was not returned, so effective HAGS state is unknown. No host setting, driver or persistent environment change was made.

## Questions for platform/toolchain support

1. Is valid CUPTI GPU activity timing supported on this exact N1X/WindowsARM64/WSL2/driver combination, and which validated driver/toolkit pair is required?
2. Why do activity-support queries and kernel metadata succeed while both kernel and memcpy timestamps remain zero, even with completed-record flush and WSL root?
3. Is the separate profiler-initialization failure related, or a distinct unsupported counter capability?

No exact driver defect or permanent platform limitation has been proven. Native Windows tracing, a Windows administrator launch, host restart, alternate driver and native Linux comparison remain untested. Request a specific supported combination before changing this working compute environment.

## Documentation used to choose tests

- [CUPTI release notes](https://docs.nvidia.com/cupti/release-notes/release-notes.html): Tegra tracing privilege requirements and a CUDA13.4Update1 HES timestamp fix motivated tests; neither proves the cause here.
- [CUPTI activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html): normal completed-record flush and timestamp callback contracts.
- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/): cuda-sw selects software tracing.
- [Publisher redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.1.json): package URL, size and SHA-256 for the isolated comparison; an exact copy is included in artifacts/.

Local measurements are authoritative for this report. All previous exercise results remain attributed to their original batches. The five larger tracing exercises151,179,186,191,195 remain BLOCKED in016 pending a successful small timing probe.
