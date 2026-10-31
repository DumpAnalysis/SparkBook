"""Small BF16 LoRA laboratory. Requires a compatible WSL2 CUDA stack."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import random

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def load_rows(path):
    rows=[json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate IDs within a split')
    for r in rows:
        if [m['role'] for m in r['messages']] != ['system','user','assistant']:
            raise ValueError('Expected system, user, assistant')
    return rows


def encode(row, tokenizer, max_length=512):
    prompt=tokenizer.apply_chat_template(row['messages'][:-1],
        tokenize=True, add_generation_prompt=True)
    full=tokenizer.apply_chat_template(row['messages'],
        tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError('Chat template prefix mismatch')
    if len(full) > max_length:
        raise ValueError('Example exceeds token limit; revise explicitly')
    labels=[-100]*len(prompt)+full[len(prompt):]
    if not any(v != -100 for v in labels[1:]):
        raise ValueError('No supervised assistant tokens')
    return {'input_ids':full, 'labels':labels,
            'attention_mask':[1]*len(full)}


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    p=argparse.ArgumentParser()
    p.add_argument('mode', choices=['train','evaluate','merge'])
    p.add_argument('--output', required=True)
    p.add_argument('--adapter')
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--max-length', type=int, default=512)
    args=p.parse_args()
    out=pathlib.Path(args.output)
    if out.exists(): raise SystemExit('Use a new output path')
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
    if not torch.cuda.is_bf16_supported(): raise SystemExit('BF16 unavailable')
    random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    base=AutoModelForCausalLM.from_pretrained(MODEL,
        revision=REVISION, dtype=torch.bfloat16,
        attn_implementation='sdpa', low_cpu_mem_usage=True)
    base.to('cuda')
    if args.mode=='train':
        if args.adapter: raise SystemExit('This lab starts a fresh adapter')
        config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
            target_modules=['q_proj','v_proj'],bias='none',
            task_type='CAUSAL_LM')
        model=get_peft_model(base,config)
        model.config.use_cache=False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    elif args.adapter:
        model=PeftModel.from_pretrained(base,args.adapter)
    else:
        model=base
    out.mkdir(parents=True)
    packages={n:importlib.metadata.version(n)
        for n in ['torch','transformers','peft','accelerate']}
    manifest={'model':MODEL,'revision':REVISION,'packages':packages,
              'arguments':vars(args),'seed':42}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('base='+MODEL,flush=True)
    print('precision=BF16 device=CUDA',flush=True)
    if args.mode=='merge':
        if not args.adapter: raise SystemExit('Merge needs --adapter')
        merged=model.merge_and_unload()
        merged.save_pretrained(out/'merged',safe_serialization=True)
        tokenizer.save_pretrained(out/'merged')
        print('merged='+str(out/'merged'),flush=True)
        return
    train=load_rows('training-data/train.jsonl')
    val=load_rows('training-data/validation.jsonl')
    test=load_rows('training-data/test.jsonl')
    sets=[{r['id'] for r in rows} for rows in [train,val,test]]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Split IDs overlap')
    for split in ['train','validation','test']:
        data=pathlib.Path('training-data/'+split+'.jsonl').read_bytes()
        manifest[split+'_sha256']=hashlib.sha256(data).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def tensors(row):
        return {k:torch.tensor([v],device='cuda')
                for k,v in encode(row,tokenizer,args.max_length).items()}
    def validation_loss():
        model.eval();total=0.;count=0
        with torch.no_grad():
            for row in val:
                batch=tensors(row)
                n=int((batch['labels'][:,1:]!=-100).sum())
                total+=float(model(**batch).loss)*n;count+=n
        return total/count
    if args.mode=='evaluate':
        model.eval();records=[]
        for row in test:
            prompt=tokenizer.apply_chat_template(row['messages'][:-1],
                tokenize=False,add_generation_prompt=True)
            batch=tokenizer(prompt,return_tensors='pt',
                add_special_tokens=False).to('cuda')
            with torch.no_grad():
                ids=model.generate(**batch,max_new_tokens=128,
                    do_sample=False,pad_token_id=tokenizer.eos_token_id)
            reply=tokenizer.decode(ids[0,batch['input_ids'].shape[1]:],
                                    skip_special_tokens=True)
            records.append({'id':row['id'],'prompt':row['messages'][:-1],
                'reference':row['messages'][-1]['content'],'response':reply})
            print('case='+row['id']+' response='+json.dumps(reply),flush=True)
        (out/'answers.json').write_text(json.dumps(records,indent=2))
        print('saved='+str(out/'answers.json'),flush=True)
        return
    if args.steps<1: raise SystemExit('steps must be positive')
    # Validate every example before the first optimizer update.
    for row in train+val: encode(row,tokenizer,args.max_length)
    params=[x for x in model.parameters() if x.requires_grad]
    print('trainable_parameters='+str(sum(x.numel() for x in params)),flush=True)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0)
    before=validation_loss()
    print(f'validation_before={before:.4f}',flush=True)
    torch.cuda.reset_peak_memory_stats()
    records=[];accumulation=4
    for step in range(1,args.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for _ in range(accumulation):
            row=train[random.randrange(len(train))]
            loss=model(**tensors(row)).loss
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            (loss/accumulation).backward();loss_sum+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(params,1.0)
        if not torch.isfinite(norm): raise RuntimeError('Non-finite gradient')
        optimizer.step();torch.cuda.synchronize()
        record={'step':step,'mean_microbatch_loss':loss_sum/accumulation,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        records.append(record)
        print(f"step={step:02d} loss={record['mean_microbatch_loss']:.4f} "
              f"peak_mib={record['peak_allocated_mib']:.1f}",flush=True)
        if step%5==0 or step==args.steps:
            checkpoint=out/f'adapter-step-{step:04d}'
            model.save_pretrained(checkpoint,safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            print('adapter_checkpoint='+str(checkpoint),flush=True)
        (out/'training.json').write_text(json.dumps(records,indent=2))
    after=validation_loss()
    (out/'validation.json').write_text(json.dumps({'before':before,'after':after},indent=2))
    print(f'validation_after={after:.4f}',flush=True)
    print('saved='+str(out),flush=True)


if __name__=='__main__':
    main()
