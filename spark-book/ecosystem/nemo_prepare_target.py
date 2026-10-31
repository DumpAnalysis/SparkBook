"""TARGET ONLY: resolve public revisions and adapt the image's own NeMo recipe.

Downloads Qwen3-0.6B and HellaSwag. Run inside a qualified, pinned image.
The CPU evidence does not execute this preparation or any NeMo recipe.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import yaml

REQUIRED_MAPPINGS = ('model','dataset','validation_dataset','peft','checkpoint',
    'step_scheduler','distributed','dist_env','dataloader','validation_dataloader','rng')

def validate_recipe(config):
    if not isinstance(config,dict):raise RuntimeError('recipe schema changed: expected a mapping')
    for key in REQUIRED_MAPPINGS:
        if not isinstance(config.get(key),dict):
            raise RuntimeError(f'recipe schema changed: missing or non-mapping {key}')
    if config['rng'].get('_target_') != 'nemo_automodel.components.training.rng.StatefulRNG':
        raise RuntimeError('recipe schema changed: unexpected RNG target')
    if not isinstance(config['rng'].get('seed'),int):
        raise RuntimeError('recipe schema changed: RNG seed must be an integer')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--recipe',type=Path,default=Path('/opt/Automodel/examples/llm_finetune/qwen/qwen3_0p6b_hellaswag_peft.yaml'))
    a=p.parse_args()
    if not a.recipe.is_file():raise RuntimeError('Selected image lacks the expected recipe; inspect its version before adapting')
    config=yaml.safe_load(a.recipe.read_text());validate_recipe(config)
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    from huggingface_hub import HfApi,snapshot_download
    from datasets import load_dataset
    from transformers import AutoTokenizer
    api=HfApi();model_id='Qwen/Qwen3-0.6B';dataset_id='rowan/hellaswag'
    mr=api.model_info(model_id).sha;dr=api.dataset_info(dataset_id).sha
    model_path=a.out/'model';data_path=a.out/'data';data_path.mkdir()
    snapshot_download(model_id,revision=mr,local_dir=model_path)
    manifests={}
    for split,n in [('train',32),('validation',8)]:
        ds=load_dataset(dataset_id,revision=dr,split=f'{split}[:{n}]')
        path=data_path/f'{split}.parquet';ds.to_parquet(str(path))
        manifests[split]={'rows':len(ds),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    lengths=[]
    ds=load_dataset(str(data_path),split='train')
    for row in ds:
        text=row['ctx']+row['endings'][int(row['label'])]
        lengths.append(len(tokenizer(text,add_special_tokens=True)['input_ids']))
    # This is a preflight guard, not the recipe's final formatted token length.
    if max(lengths)>2048:raise RuntimeError('Preflight token length exceeds lab limit')
    shutil.copy2(a.recipe,a.out/'upstream-recipe.yaml')
    config['rng']['seed']=1403
    config['model']['pretrained_model_name_or_path']=str(model_path)
    config['model']['torch_dtype']='bf16'
    config['peft'].update(dim=8,alpha=32,use_triton=False)
    config['step_scheduler'].update(global_batch_size=4,local_batch_size=1,max_steps=4,
        ckpt_every_steps=2,val_every_steps=2,num_epochs=1)
    config['distributed'].update(dp_size=1,tp_size=1,cp_size=1)
    config['dist_env']['timeout_minutes']=10
    config['checkpoint'].update(enabled=True,checkpoint_dir=str(a.out/'checkpoints'),
        model_save_format='safetensors',save_consolidated=True)
    config['dataset'].update(path_or_dataset=str(data_path),split='train',num_samples_limit=32)
    config['validation_dataset'].update(path_or_dataset=str(data_path),split='validation',num_samples_limit=8)
    config['dataloader']['num_workers']=0;config['validation_dataloader']['num_workers']=0
    config.pop('wandb',None);config.pop('ci',None)
    (a.out/'recipe.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    manifest=dict(model=model_id,model_revision=mr,dataset=dataset_id,dataset_revision=dr,
        rows=manifests,unformatted_max_tokens=max(lengths),upstream_recipe_sha256=hashlib.sha256(a.recipe.read_bytes()).hexdigest(),
        note='Resolved public revisions; training not yet executed; final recipe tokens require inspection')
    (a.out/'preparation.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
