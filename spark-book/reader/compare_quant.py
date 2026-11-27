"""Check the matched HTTP requests and display the finish reason for each quantization."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('q4', type=Path)
    p.add_argument('q8', type=Path)
    args = p.parse_args()
    receipts = [json.loads(path.read_text()) for path in (args.q4, args.q8)]
    request = receipts[0]['request']
    if request != receipts[1]['request']:
        p.exit(1, 'Request bodies differ. Compare messages, model alias, seed and token budget.\n')
    print(f"request_bodies=identical  max_tokens={request['max_tokens']}")
    for label, receipt in zip(('Q4_K_M', 'Q8_0'), receipts):
        if receipt['status'] != 200:
            p.exit(1, f"{label} HTTP {receipt['status']}; see the saved response.\n")
        choice = receipt['body']['choices'][0]
        if not choice['message']['content'].strip() or not choice['finish_reason']:
            p.exit(1, f'{label} missing answer content or finish reason.\n')
        tokens = receipt['body']['usage']['completion_tokens']
        print(f"{label:7} HTTP 200  generated_tokens={tokens}  finish={choice['finish_reason']}")


if __name__ == '__main__':
    main()
