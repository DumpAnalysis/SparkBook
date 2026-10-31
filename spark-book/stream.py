"""Measure client time to first nonempty content event, not kernel time."""
import argparse
import json
import pathlib
import re
import time
import urllib.request
from spark_lab import save

p = argparse.ArgumentParser()
p.add_argument("prompt")
p.add_argument("--base", default="http://127.0.0.1:8080")
p.add_argument("--out", required=True)
a = p.parse_args()
if not re.fullmatch(r"http://127\.0\.0\.1:[0-9]+", a.base):
    raise SystemExit("Use an explicit IPv4 loopback URL")
prompt = pathlib.Path(a.prompt).read_text(encoding="utf-8")
payload = {"model": "local-model", "messages": [
    {"role": "user", "content": prompt}], "temperature": 0,
    "seed": 42, "max_tokens": 128, "stream": True,
    "stream_options": {"include_usage": True}}
req = urllib.request.Request(a.base + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
start = time.perf_counter()
events, pieces, buffer = [], [], []
first = None
usage = None
error = None
done = False
try:
    with opener.open(req, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line:
                if line.startswith("data:"):
                    buffer.append(line[5:].lstrip())
                continue
            if not buffer:
                continue
            data = "\n".join(buffer)
            buffer = []
            elapsed = time.perf_counter() - start
            events.append({"elapsed_s": elapsed, "data": data})
            if data == "[DONE]":
                done = True
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first is None:
                        first = elapsed
                    pieces.append(content)
except Exception as exc:
    error = type(exc).__name__ + ": " + str(exc)
total = time.perf_counter() - start
record = {"request": payload, "events": events,
          "first_content_s": first, "elapsed_s": total,
          "usage": usage, "done": done, "error": error,
          "content": "".join(pieces)}
save(a.out, record)
print("content=" + json.dumps(record["content"], ensure_ascii=False))
print("first_content_seconds=" + (f"{first:.3f}" if first is not None else "NA"))
print(f"total_seconds={total:.3f}")
print("done=" + str(done))
print("error=" + str(error))
print("saved=" + a.out)
raise SystemExit(0 if done and error is None else 2)
