"""TARGET: qualify one consolidated HF-compatible adapter, then merge on CPU.
Sharded training checkpoints are deliberately rejected by the adapter gate.
"""
import argparse,json,hashlib,importlib.util
from pathlib import Path
from lab import dump

def require_peft():
    if importlib.util.find_spec('peft') is None:
        raise RuntimeError('Hugging Face peft is required for CPU adapter export; qualify it with this environment\'s transformers version before retrying')

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();require_peft();a.out.mkdir(parents=True,exist_ok=False)
    candidates=list(a.checkpoint.resolve().rglob('adapter_config.json'))
    if len(candidates)!=1:raise RuntimeError('Select ONE completed consolidated adapter checkpoint; expected one adapter_config.json')
    adapter=candidates[0].parent
    if not list(adapter.glob('*.safetensors')):raise RuntimeError('No consolidated adapter safetensors')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(a.base,local_files_only=True,dtype=torch.float32).eval()
    model=PeftModel.from_pretrained(base,adapter,is_trainable=False).eval()
    prompt=tokenizer.apply_chat_template([{'role':'user','content':'Name one reason to collect a GPU trace.'}],
        tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(prompt,return_tensors='pt')
    with torch.inference_mode():
        before=model(**inputs).logits.float()
        generation=model.generate(**inputs,max_new_tokens=16,do_sample=False)
    merged=model.merge_and_unload().eval()
    with torch.inference_mode():after=merged(**inputs).logits.float()
    error=float((before-after).abs().max())
    if not torch.allclose(before,after,atol=1e-3,rtol=1e-3):raise RuntimeError(f'adapter merge changed logits; max_error={error}')
    merged.save_pretrained(a.out/'merged',safe_serialization=True);tokenizer.save_pretrained(a.out/'merged')
    inventory={str(f.relative_to(a.out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((a.out/'merged').rglob('*')) if f.is_file()}
    dump(a.out/'handoff.json',dict(base=str(a.base.resolve()),adapter=str(adapter),merge_max_logit_error=error,
        prompt=prompt,response=tokenizer.decode(generation[0],skip_special_tokens=True),files=inventory,
        note='One-prompt merge equivalence, not held-out quality or serving qualification'))
    print(f'CPU merge max_logit_error={error:.6g}; merged checkpoint and hashes retained')

if __name__=='__main__':main()
