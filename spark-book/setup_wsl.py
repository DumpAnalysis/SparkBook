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
    for pin in archives:
        with tarfile.open(ROOT / "downloads" / pin["name"]) as archive:
            archive.extractall(ROOT / "linux-bin", filter="data")
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
