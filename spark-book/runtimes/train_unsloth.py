"""Target-only Unsloth smoke training with an exact base binding and restart.
Never run this in the book's standard-library control environment.
"""
import argparse,json,math,os,platform,sys,subprocess
from pathlib import Path
from common import fresh,tree_identity,write_json,ContractError
from lab import training_rows,split_check

def main(a):
    out=fresh(a.out)
    # No network model fallback or telemetry in this local experiment.
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true',TOKENIZERS_PARALLELISM='false')
    result={'scope':'target Unsloth training','status':'failed'}
    try:
        # Unsloth must be imported before Transformers/PEFT.
        import unsloth
        from unsloth import FastLanguageModel
        import torch
        from transformers import Trainer,TrainingArguments,TrainerCallback,DataCollatorForLanguageModeling
        if not torch.cuda.is_available():raise ContractError('CUDA unavailable; CPU fallback disabled')
        torch.manual_seed(42)
        base=Path(a.base).resolve();digest,files=tree_identity(base)
        cfg=json.loads((base/'config.json').read_text())
        if cfg.get('model_type')!='qwen2':raise ContractError('this recipe is qualified only for the selected Qwen2 base')
        contract={'base_tree_sha256':digest,'base_files':files,'sequence':256,'rank':8,'microbatch':1,'accumulation':4,'seed':42,'planned_steps':10}
        write_json(out/'contract.json',contract)
        checkpoint=None
        if a.resume_from:
            previous=Path(a.resume_from);prior=json.loads((previous/'contract.json').read_text())
            if prior!=contract:raise ContractError('training/base contract changed before resume')
            checkpoint=previous/'trainer/checkpoint-5'
            for name in ['trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth']:
                if not (checkpoint/name).exists():raise ContractError('incomplete checkpoint '+name)
        env={'python':platform.python_version(),'machine':platform.machine(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'free_total_bytes':torch.cuda.mem_get_info(),'unsloth':getattr(unsloth,'__version__','unknown')}
        write_json(out/'environment.json',env)
        freeze=subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True);(out/'pip-freeze.txt').write_text(freeze.stdout+freeze.stderr)
        model,tokenizer=FastLanguageModel.from_pretrained(model_name=str(base),max_seq_length=256,dtype=None,load_in_4bit=True,device_map={'':0},local_files_only=True)
        model=FastLanguageModel.get_peft_model(model,r=8,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],lora_alpha=16,lora_dropout=0,bias='none',use_gradient_checkpointing='unsloth',random_state=42)
        # Preserve genuine EOS supervision; pad with a different existing token.
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id==tokenizer.eos_token_id:
            candidate=tokenizer.convert_tokens_to_ids('<|endoftext|>')
            if candidate is None or candidate==tokenizer.eos_token_id:raise ContractError('need a distinct existing padding token')
            tokenizer.pad_token_id=candidate
        tokenizer.padding_side='right'
        train,test=training_rows();split_check([x[0] for x in train],[x[0] for x in test])
        def encode(rows):
            output=[]
            for text,label in rows:
                s='Classify as memory, input, or unknown.\nEvidence: '+text+'\nLabel: '+label+tokenizer.eos_token
                item=tokenizer(s,add_special_tokens=False)
                if len(item['input_ids'])>256:raise ContractError('training sample exceeds sequence cap')
                output.append(item)
            return output
        tr,ev=encode(train),encode(test)
        # Labels cover all non-padding tokens, intentionally including prompt tokens.
        collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
        class StopAtFive(TrainerCallback):
            def on_step_end(self,args,state,control,**kwargs):
                if not a.resume_from and state.global_step==5:control.should_training_stop=True;control.should_save=True
                return control
        args=TrainingArguments(output_dir=str(out/'trainer'),per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,max_steps=10,learning_rate=2e-4,lr_scheduler_type='linear',warmup_steps=0,logging_steps=1,logging_nan_inf_filter=False,save_strategy='steps',save_steps=5,save_total_limit=2,report_to='none',seed=42,data_seed=42,optim='adamw_torch',bf16=torch.cuda.is_bf16_supported(),fp16=not torch.cuda.is_bf16_supported(),dataloader_num_workers=0)
        trainer=Trainer(model=model,args=args,train_dataset=tr,eval_dataset=ev,data_collator=collator,callbacks=[StopAtFive()])
        with model.disable_adapter():base_eval=trainer.evaluate()
        torch.cuda.reset_peak_memory_stats();trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        write_json(out/'training-log.json',trainer.state.log_history)
        expected_step=10 if a.resume_from else 5
        if trainer.state.global_step!=expected_step:raise ContractError('unexpected completed update count')
        for event in trainer.state.log_history:
            for key in ['loss','grad_norm']:
                if key in event and not math.isfinite(event[key]):raise ContractError('nonfinite '+key)
        adapted_eval=trainer.evaluate();torch.cuda.synchronize()
        if not math.isfinite(adapted_eval['eval_loss']):raise ContractError('nonfinite evaluation loss')
        model.save_pretrained(str(out/'adapter'));tokenizer.save_pretrained(str(out/'adapter'))
        write_json(out/'training-log.json',trainer.state.log_history)
        result.update({'status':'complete','global_step':trainer.state.global_step,'base_eval':base_eval,'adapter_eval':adapted_eval,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'supervised_train_tokens_per_pass':sum(len(x['input_ids']) for x in tr),'restart_from':str(checkpoint) if checkpoint else None})
        print(json.dumps(result,indent=2));return 0
    except Exception as e:
        result.update({'error_type':type(e).__name__,'error':str(e)});print(json.dumps(result,indent=2));return 1
    finally:write_json(out/'result.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--resume-from');sys.exit(main(p.parse_args()))
