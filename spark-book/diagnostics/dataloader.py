"""Compare worker counts on a fixed, ordered CPU dataset."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def measure(dataset, workers):
    options = {"multiprocessing_context": "spawn"} if workers else {}
    started = time.monotonic()
    loader = DataLoader(dataset, batch_size=32, num_workers=workers, shuffle=False, **options)
    batches = []
    for (values,) in loader:
        if not batches:
            first = time.monotonic()
        batches.append(values)
    finished = time.monotonic()
    values = torch.cat(batches)
    if not torch.equal(values, dataset.tensors[0]):
        raise ValueError(f"Output differs from the ordered input with {workers} workers")
    return {
        "workers": workers,
        "items": values.numel(),
        "batches": len(batches),
        "input_digest": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
        "startup_seconds": first - started,
        "rest_seconds": finished - first,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new JSON evidence file")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output filename")
    dataset = TensorDataset(torch.arange(1024))
    rows = [measure(dataset, workers) for workers in (0, 2)]
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
        stream.write("\n")
    print(f"Both worker counts returned {rows[0]['items']} values in {rows[0]['batches']} batches.")


if __name__ == "__main__":
    main()
