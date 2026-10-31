"""Target model exercise: masked mean pooling is an explicit model contract."""
import argparse,json
from pathlib import Path
import numpy as np
from lab import dump,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import torch,onnx,onnxruntime as ort
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(1)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    encoder=AutoModel.from_pretrained(a.model,local_files_only=True).cpu().eval()
    class MeanEncoder(torch.nn.Module):
        def __init__(self,encoder):super().__init__();self.encoder=encoder
        def forward(self,input_ids,attention_mask):
            h=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
            mask=attention_mask.unsqueeze(-1).to(h.dtype)
            pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled,p=2,dim=1,eps=1e-12)
    model=MeanEncoder(encoder).eval()
    texts=['collect a GPU trace','inspect device allocations','a sentence about gardening']
    sample=tokenizer(texts[:2],padding=True,truncation=True,max_length=128,return_tensors='pt')
    graph=a.out/'encoder.onnx'
    with torch.no_grad():
        torch.onnx.export(model,(sample['input_ids'],sample['attention_mask']),str(graph),dynamo=False,
          input_names=['input_ids','attention_mask'],output_names=['embeddings'],opset_version=18,
          dynamic_axes={'input_ids':{0:'N',1:'S'},'attention_mask':{0:'N',1:'S'},'embeddings':{0:'N'}})
    onnx.checker.check_model(str(graph))
    session=ort.InferenceSession(str(graph),providers=['CPUExecutionProvider']);results=[]
    for n in (1,2,3):
        tokens=tokenizer(texts[:n],padding=True,truncation=True,max_length=128,return_tensors='pt')
        with torch.inference_mode():reference=model(tokens['input_ids'],tokens['attention_mask']).numpy()
        actual=session.run(None,{k:tokens[k].numpy() for k in ['input_ids','attention_mask']})[0]
        error=float(np.max(np.abs(reference-actual)));cosine=np.sum(reference*actual,axis=1)
        assert error<1e-4 and float(np.min(cosine))>.9999
        np.savez(a.out/f'batch-{n}.npz',reference=reference,actual=actual)
        results.append(dict(batch=n,sequence=tokens['input_ids'].shape[1],max_error=error,min_cosine=float(cosine.min())))
    tokenizer.save_pretrained(a.out/'tokenizer')
    dump(a.out/'export.json',dict(model_path=str(a.model),graph_sha256=sha(graph),pooling='attention-masked mean plus L2',
          max_tokens=128,tests=results,note='Not equivalent to a model with a different pooling or projection head'))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
