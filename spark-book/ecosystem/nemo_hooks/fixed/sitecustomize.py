"""Record training invariants in this batch's torchrun worker only.

Imports are guarded to keep the launcher unchanged. No tensor, optimizer,
configuration, RNG, or data-order mutation is performed by these observers.
"""
import os,sys
if os.path.basename(sys.argv[0]) in ("automodel", "train_ft.py") and os.environ.get('SPARK_AUDIT_LABEL'):
 import json,hashlib,functools
 from pathlib import Path
 import torch
 from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction as Recipe
 audit=Path('/work/lab')/('audit-'+os.environ['SPARK_AUDIT_LABEL']);audit.mkdir(exist_ok=False)
 def local(t):return t.to_local() if hasattr(t,'to_local') else t
 def digest(t):
  v=local(t.detach()).cpu().contiguous();return hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()
 def inventory(self):
  return {f'{i}:{n}':{'shape':list(p.shape),'local_shape':list(local(p).shape),'dtype':str(p.dtype),'requires_grad':p.requires_grad,'numel':p.numel(),'sha256':digest(p)} for i,m in enumerate(self.model_parts) for n,p in m.named_parameters()}
 original_loop=Recipe.run_train_validation_loop
 @functools.wraps(original_loop)
 def loop(self,*args,**kwargs):
  before=inventory(self);(audit/'parameters-before.json').write_text(json.dumps(before,indent=2))
  print('SPARK_AUDIT trainable parameters',sum(p['numel'] for p in before.values() if p['requires_grad']),flush=True)
  try:return original_loop(self,*args,**kwargs)
  finally:
   after=inventory(self);(audit/'parameters-after.json').write_text(json.dumps(after,indent=2))
   frozen=[n for n,p in before.items() if not p['requires_grad']];changed=[n for n in frozen if before[n]['sha256']!=after[n]['sha256']]
   result={'frozen_parameter_tensors':len(frozen),'frozen_parameter_count':sum(before[n]['numel'] for n in frozen),'changed_frozen':changed,'trainable_names':[n for n,p in before.items() if p['requires_grad']],'scope':'Actual live parameters hashed before/after training loop; logging adds transfer and synchronization overhead.'}
   (audit/'frozen-check.json').write_text(json.dumps(result,indent=2));print('SPARK_AUDIT frozen tensors changed',len(changed),flush=True)
   assert not changed,'Frozen base parameter bytes changed'
 Recipe.run_train_validation_loop=loop
 original_step=Recipe._run_train_optim_step
 @functools.wraps(original_step)
 def step(self,batches,*args,**kwargs):
  rows=[]
  for batch in batches:
   selected={k:local(v.detach()).cpu().tolist() for k,v in batch.items() if k in ['input_ids','labels','loss_mask'] and isinstance(v,torch.Tensor)}
   rows.append({'sha256':hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),'tensors':selected})
  with (audit/'batches.jsonl').open('a') as f:f.write(json.dumps({'step':self.step_scheduler.step,'microbatches':rows})+'\n')
  return original_step(self,batches,*args,**kwargs)
 Recipe._run_train_optim_step=step

 # Observe actual constructor boundaries in the pinned container, without RNG mutation.
 import traceback
 from torch.utils.data.dataloader import _BaseDataLoaderIter
 from nemo_automodel.components.training.rng import StatefulRNG
 def rng_hash(state):return hashlib.sha256(state.numpy().tobytes()).hexdigest()
 def event(value):
  with (audit/'rng-events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
 original_iter_init=_BaseDataLoaderIter.__init__
 def traced_iter_init(self,loader,*args,**kwargs):
  before=torch.get_rng_state().clone()
  restored=getattr(loader,'next_iter_state',None) is not None
  eligible=(restored and loader.num_workers==0 and loader._dataset_kind==0 and loader.generator is None and getattr(loader.sampler,'shuffle',None) is False)
  # Rebuilding an existing single-process iterator needs no new worker seed.
  # Give only the base constructor a clone, so its unused seed draw cannot
  # advance the checkpoint-restored global CPU RNG. No saved state is edited.
  if eligible:loader.generator=torch.Generator().set_state(before)
  try:original_iter_init(self,loader,*args,**kwargs)
  finally:
   if eligible:loader.generator=None
  if eligible:assert torch.equal(before,torch.get_rng_state()),'Unexpected RNG use during corrected constructor'

  after=torch.get_rng_state().clone()
  clone=torch.Generator().set_state(before);draw=int(torch.empty((),dtype=torch.int64).random_(generator=clone))
  event({'event':'iterator_constructor','isolated_unused_worker_seed':eligible,'restored_iterator':restored,'iterator_type':type(self).__name__,'num_workers':loader.num_workers,'dataset_kind':loader._dataset_kind,'sampler_type':type(loader.sampler).__name__,'sampler_shuffle':getattr(loader.sampler,'shuffle',None),'loader_generator_is_none':loader.generator is None,'before_cpu_rng':rng_hash(before),'after_cpu_rng':rng_hash(after),'one_int64_draw_matches':torch.equal(clone.get_state(),after),'base_seed_equals_predicted':self._base_seed==draw,'stack':traceback.format_stack()})
 _BaseDataLoaderIter.__init__=traced_iter_init
 original_rng_load=StatefulRNG.load_state_dict
 def traced_rng_load(self,state):
  original_rng_load(self,state)
  event({'event':'checkpoint_rng_restored','cpu_rng':rng_hash(torch.get_rng_state())})
 StatefulRNG.load_state_dict=traced_rng_load
