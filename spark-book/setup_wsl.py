"""Acquire or verify the pinned native runtime and GGUF files for the WSL2 laboratory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request

from runtimes.pinned_download import fetch_pinned, locked_directory

ROOT = Path(__file__).resolve().parent


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def verify_file(path, pin=None):
    if not path.is_file():
        raise SystemExit(f"Missing asset: {path}")
    if pin and "bytes" in pin and path.stat().st_size != pin["bytes"]:
        raise SystemExit(f"Size mismatch: {path}")
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if pin and actual != pin["sha256"]:
        raise SystemExit(f"SHA-256 mismatch: {path}")
    return actual


def fetch(url, path, pin=None):
    if pin and "bytes" in pin:
        with locked_directory(path.parent):
            return fetch_pinned(url, path, pin["bytes"], sha256=pin["sha256"])
    if path.exists():
        return verify_file(path, pin)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".partial", delete=False) as out:
        temp = Path(out.name)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temp.open("wb") as out:
            shutil.copyfileobj(response, out)
        digest = verify_file(temp, pin)
        os.replace(temp, path)
        return digest
    finally:
        temp.unlink(missing_ok=True)


def install_runtime(archives, destination, server_state):
    """Stage verified archives, then replace only missing or different members."""
    destination.mkdir(exist_ok=True)
    if destination.is_symlink():
        raise ValueError(f'Expected a runtime directory: {destination}')
    with locked_directory(destination), tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp)
        for path in archives:
            with tarfile.open(path) as archive:
                archive.extractall(stage, filter="data")
        directories, changes = [], []
        for source in sorted(stage.rglob('*')):
            target = destination / source.relative_to(stage)
            if source.is_symlink():
                if target.is_symlink() and target.readlink() == source.readlink():
                    continue
            elif source.is_dir():
                if target.is_symlink() or (target.exists() and not target.is_dir()):
                    raise ValueError(f'Unexpected runtime directory entry: {target}')
                directories.append(target)
                continue
            elif not target.is_symlink() and target.is_file():
                mode, installed = source.stat(), target.stat()
                if mode.st_size == installed.st_size and (mode.st_mode & 0o111) == (installed.st_mode & 0o111):
                    with source.open('rb') as expected, target.open('rb') as actual:
                        if hashlib.file_digest(expected, 'sha256').digest() == hashlib.file_digest(actual, 'sha256').digest():
                            continue
            if target.is_dir() and not target.is_symlink():
                raise ValueError(f'Unexpected directory in place of a runtime file: {target}')
            changes.append((source, target))
        if changes and server_state.exists():
            raise SystemExit('Runtime files need repair. Stop the recorded server with bash serve.sh stop and retry.')
        for directory in directories:
            directory.mkdir(exist_ok=True)
        for source, target in changes:
            os.replace(source, target)
        print(f'runtime_files_replaced={len(changes)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=["q4_k_m", "q8_0"], default="q4_k_m")
    parser.add_argument("--verify-only", action="store_true", help="read and hash local assets without downloads, extraction or writes")
    args = parser.parse_args()
    pins = json.loads((ROOT / "reader/native-asset-pins.json").read_text())
    tag, repo, revision = (pins[k] for k in ("release", "model", "revision"))
    archives, shards = pins["archives"], pins["shards"][args.quant]
    if not args.verify_only:
        if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
            raise SystemExit("This setup requires Linux ARM64")
        for name in ("downloads", "linux-bin", "models/" + args.quant, "runs"):
            (ROOT / name).mkdir(parents=True, exist_ok=True)
    records = []
    for items, directory, base in (
        (archives, ROOT / "downloads", f"https://github.com/ggml-org/llama.cpp/releases/download/{tag}"),
        (shards, ROOT / "models" / args.quant, f"https://huggingface.co/{repo}/resolve/{revision}"),
    ):
        for pin in items:
            name = pin["name"]
            path = directory / name
            if args.verify_only:
                digest = verify_file(path, pin)
            else:
                digest = fetch(f"{base}/{name}", path, pin)
            records.append(dict(pin, sha256=digest))
            print(f"verified {digest}  {name}", flush=True)
    if args.verify_only:
        print(f"matched_files={len(records)} quant={args.quant}")
        return
    # Verify the complete set before altering the installed runtime or receipt.
    install_runtime([ROOT / "downloads" / pin["name"] for pin in archives],
                    ROOT / "linux-bin", ROOT / "runs/linux-server-state.json")
    manifest = {"release": tag, "model": repo, "revision": revision, "records": records}
    receipt = ROOT / "runs" / f"linux-manifest-{args.quant}.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=receipt.parent, delete=False) as temp:
        tmp = Path(temp.name)
        try:
            json.dump(manifest, temp, indent=2)
            temp.flush()
            os.replace(tmp, receipt)
        finally:
            tmp.unlink(missing_ok=True)
    print(f"runtime={tag} backend=cuda architecture=aarch64")
    print(f"model={repo} quant={args.quant.upper()} shards={len(shards)}")
    print("manifest=" + str(receipt.relative_to(ROOT)))
    print("No Linux display driver was installed.")


if __name__ == "__main__":
    main()
