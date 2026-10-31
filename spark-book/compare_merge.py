"""Compare a saved BF16 merge with its base-plus-adapter model on the held-out prompts."""
import argparse
import gc
import json
from pathlib import Path

from finetune import MODEL, REVISION, load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', required=True, type=Path)
    parser.add_argument('--merged', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows('training-data/test.jsonl')
    if not rows or not torch.cuda.is_available():
        raise SystemExit('This comparison requires held-out rows and CUDA.')
    args.out.mkdir(parents=True, exist_ok=False)
    options = dict(dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)

    def evaluate(which):
        torch.manual_seed(42)
        if which == 'adapter':
            model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, **options)
            model = PeftModel.from_pretrained(model, args.adapter)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.merged, **options)
        model.to('cuda').eval()
        vectors, responses = [], []
        with torch.no_grad():
            for row in rows:
                text = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=False, add_generation_prompt=True)
                batch = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
                vectors.append(model(**batch).logits[0, -1].float().cpu())
                ids = model.generate(**batch, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                generated = ids[0, batch['input_ids'].shape[1]:].tolist()
                responses.append({'id':row['id'], 'generated_token_ids':generated,
                                  'response':tokenizer.decode(generated, skip_special_tokens=True)})
        (args.out/(which+'.json')).write_text(json.dumps(responses, indent=2)+'\n')
        return torch.stack(vectors), responses

    reference, first = evaluate('adapter')
    gc.collect()
    torch.cuda.empty_cache()
    merged, second = evaluate('merged')
    delta = merged-reference
    errors = [a['id'] for a,b in zip(first,second) if a['generated_token_ids'] != b['generated_token_ids']]
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(merged).all())
    result = {'base':MODEL, 'revision':REVISION, 'adapter':str(args.adapter.resolve()),
              'merged':str(args.merged.resolve()), 'finite_logits':finite,
              'first_token_argmax_equal':bool(torch.equal(reference.argmax(-1),merged.argmax(-1))),
              'maximum_absolute_logit_difference':float(delta.abs().max()) if finite else None,
              'rms_logit_difference':float(delta.square().mean().sqrt()) if finite else None,
              'greedy_mismatch_ids':errors,'greedy_continuations_equal':not errors}
    (args.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if finite and not errors else 1


if __name__=='__main__':
    raise SystemExit(main())
