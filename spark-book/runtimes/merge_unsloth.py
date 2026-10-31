"""Target-only local adapter merge; conversion/quantization are separate steps."""
import argparse,json,os
from pathlib import Path
from common import fresh,tree_identity,ContractError,write_json
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=fresh(a.out);run=Path(a.run);contract=json.loads((run/'contract.json').read_text())
if tree_identity(a.base)[0]!=contract['base_tree_sha256']:raise ContractError('base changed since training')
adapter_cfg=json.loads((run/'adapter/adapter_config.json').read_text())
if Path(adapter_cfg['base_model_name_or_path']).resolve()!=Path(a.base).resolve():raise ContractError('adapter points to a different base path')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from unsloth import FastLanguageModel
model,tok=FastLanguageModel.from_pretrained(model_name=str((run/'adapter').resolve()),max_seq_length=256,load_in_4bit=True,local_files_only=True)
model.save_pretrained_merged(str(out/'merged'),tok,save_method='merged_16bit')
digest,files=tree_identity(out/'merged');write_json(out/'merge-receipt.json',{'base_tree_sha256':contract['base_tree_sha256'],'merged_tree_sha256':digest,'files':files})
print('MERGED local adapter; no GGUF conversion or quality claim')
