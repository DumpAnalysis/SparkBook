"""Original teaching data. Families, not individual sentences, define splits."""
import argparse, json, hashlib
from pathlib import Path
FAMILIES = [
('allocation','train','A CUDA allocation failed while the model was loading. Reduce the resident weights or the context reservation.','Why does loading the model exhaust device memory?','How can I reduce memory needed at model startup?'),
('cache','train','The key value cache grows with token count and concurrent requests. Shorten context or reduce parallel slots.','Why does a longer chat consume more memory?','What makes memory rise when more conversations run?'),
('driver','train','WSL uses the Windows NVIDIA driver through the mapped guest interface. Do not install a Linux display driver in the guest.','Where does the WSL NVIDIA driver come from?','Should I install a second display driver inside WSL?'),
('copy','train','A large host to device copy can dominate a short GPU operation. Measure transfer time separately from kernel time.','Why is moving input slower than the kernel?','How do I account for host device transfer overhead?'),
('padding','train','Padding tokens must be excluded from mean pooling. Use the attention mask to compute the denominator.','Why does adding padding change my sentence vector?','How should padded tokens be handled in pooling?'),
('normalization','train','Normalize each nonzero embedding before using a dot product as cosine similarity. Reject zero vectors.','When does a dot product equal cosine similarity?','How do I compare vectors with different lengths?'),
('truncation','train','A tokenizer truncates text beyond the configured length. Inspect token counts and chunk long documents before encoding.','Why does the last part of my document disappear?','How can I detect text cut off before embedding?'),
('split','train','Split by source incident before forming pairs or chunks. Keep related versions in one partition to avoid leakage.','How can near duplicate reports leak into evaluation?','Should chunks from one incident cross data splits?'),
('checkpoint','train','A recovery checkpoint includes optimizer state and random number generator state as well as model parameters.','Why does resumed training diverge from uninterrupted training?','What must be saved for a faithful training restart?'),
('gradient','train','A missing parameter gradient can indicate a detached tensor or an unused branch. Inspect the graph and trainable flags.','Why does a trainable layer receive no gradient?','How can a detach operation prevent learning?'),
('stream','train','Work on independent CUDA streams needs an event dependency before one stream consumes the output of another.','How do I make one stream wait for data from another?','Why can asynchronous work read unfinished output?'),
('bounds','train','An invalid CUDA write can corrupt a neighboring allocation. Run a bounded reproducer under Compute Sanitizer.','How can I find a GPU buffer overrun?','Which diagnostic locates an out of bounds device write?'),
('temperature','dev','Sustained laptop load may change clock frequency and throughput. Record power mode and temperature for repeated benchmarks.','Why is a later benchmark slower after sustained load?','What operating conditions should accompany throughput results?'),
('index','dev','Changing an embedding model invalidates stored document vectors. Rebuild the index and retain the old model index pair for rollback.','Can I query an old index with a new embedding model?','What must be rebuilt after changing the sentence encoder?'),
('false_negative','dev','Two differently worded answers can both be relevant. In batch contrastive training must not label one as a negative for the other.','Why can a plausible answer become a false negative?','Do distinct strings guarantee unrelated training examples?'),
('projection','test','A two dimensional map can distort neighbors and distances. Evaluate retrieval in the original embedding space.','Does an attractive scatter plot prove good search quality?','Where should I measure retrieval before accepting a projection?'),
('tokenizer','test','A custom vocabulary changes token identifiers. A pretrained embedding table cannot be reused with arbitrary new identifiers.','Can I replace the tokenizer without changing the encoder?','Why do new token identifiers break old word embeddings?'),
('collapse','test','Nearly identical sentence vectors can make all cosine scores similar. Inspect vector variance and unrelated pair similarities.','Why do all unrelated documents receive the same score?','How can I detect collapsed sentence representations?'),
]
def records():
    return [dict(id=f,split=s,document=d,queries=[q1,q2]) for f,s,d,q1,q2 in FAMILIES]
def rows():
    out=[]
    for r in records():
        for kind,text in [('document',r['document']),('query0',r['queries'][0]),('query1',r['queries'][1])]:
            out.append(dict(id=r['id']+':'+kind,family=r['id'],split=r['split'],kind=kind,text=text))
    return out

def validate(rs=None):
    rs=records() if rs is None else rs
    seen=set(); texts=set()
    for r in rs:
        assert r['id'] not in seen, 'duplicate family'
        seen.add(r['id']); assert r['split'] in {'train','dev','test'}
        for t in [r['document'],*r['queries']]:
            n=' '.join(t.lower().split())
            assert n not in texts, 'duplicate text'; texts.add(n)
    return {s:sum(r['split']==s for r in rs) for s in ('train','dev','test')}

def digest():
    return hashlib.sha256(json.dumps(records(),sort_keys=True).encode()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='embeddings/data');a=p.parse_args()
    dst=Path(a.out);dst.mkdir(parents=True,exist_ok=True)
    for s in ('train','dev','test'):
        (dst/(s+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records() if r['split']==s))
    print('families',json.dumps(validate(),sort_keys=True))
    print('rows',len(rows()),'query_document_pairs',sum(len(r['queries']) for r in records()))
    print('sha256',digest())
    print('wrote',a.out)
