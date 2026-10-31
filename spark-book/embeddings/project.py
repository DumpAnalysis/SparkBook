"""CPU PCA, t-SNE and UMAP. Labels are never passed to fit."""
import argparse,json,importlib.metadata
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,trustworthiness
from sklearn.metrics import pairwise_distances
from common import normalize,save_json,sha_file

def overlap(x,z,k=5):
    def neighbors(v,metric):
        d=pairwise_distances(v,metric=metric);np.fill_diagonal(d,np.inf)
        return np.argsort(d,axis=1,kind='stable')[:,:k]
    a=neighbors(x,'cosine');b=neighbors(z,'euclidean')
    return float(np.mean([len(set(i)&set(j))/k for i,j in zip(a,b)]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--perplexity',type=float,default=10);p.add_argument('--neighbors',type=int,default=10);p.add_argument('--min-dist',type=float,default=.1);a=p.parse_args()
    import umap
    if a.input:
        src=Path(a.input);x=normalize(np.load(src/'vectors.npy',allow_pickle=False));rows=json.loads((src/'rows.json').read_text());labels=np.array([r['split'] for r in rows]);kind='Sentence embeddings';ids=[r['id'] for r in rows];source=sha_file(src/'vectors.npy')
    else:
        rng=np.random.default_rng(17);centers=normalize(rng.normal(size=(3,24)))
        x=normalize(np.repeat(centers,30,axis=0)+rng.normal(scale=.16,size=(90,24)));labels=np.repeat(['Group A','Group B','Group C'],30);ids=[str(i) for i in range(len(x))];kind='Synthetic vectors';source='synthetic centers seed 17; not text embeddings'
    n=len(x)
    if not 0<a.perplexity<n:raise ValueError('perplexity must be below sample count')
    if not 2<=a.neighbors<n:raise ValueError('neighbors must be between 2 and n-1')
    pca=PCA(n_components=2,svd_solver='full');z0=pca.fit_transform(x)
    z1=TSNE(n_components=2,perplexity=a.perplexity,metric='cosine',init='pca',learning_rate='auto',max_iter=1000,random_state=a.seed).fit_transform(x)
    reducer=umap.UMAP(n_components=2,n_neighbors=a.neighbors,min_dist=a.min_dist,metric='cosine',random_state=a.seed,transform_seed=a.seed,n_jobs=1)
    z2=reducer.fit_transform(x)
    dst=Path(a.out);dst.mkdir(parents=True,exist_ok=True)
    report=dict(source=source,kind=kind,n=n,dimension=x.shape[1],seed=a.seed,perplexity=a.perplexity,neighbors=a.neighbors,min_dist=a.min_dist,pca_variance=float(pca.explained_variance_ratio_.sum()),versions={k:importlib.metadata.version(k) for k in ['numpy','scikit-learn','umap-learn','numba']},methods={})
    for name,z in [('PCA',z0),('t-SNE',z1),('UMAP',z2)]:
        m=dict(trustworthiness_5=float(trustworthiness(x,z,n_neighbors=5,metric='cosine')),neighbor_overlap_5=overlap(x,z));report['methods'][name]=m
        fig,ax=plt.subplots(figsize=(6.1,3.8),layout='constrained')
        for lab,marker in zip(sorted(set(labels)),['o','^','s','D','v']):
            mask=labels==lab;ax.scatter(z[mask,0],z[mask,1],s=28,label=lab,marker=marker,alpha=.8,edgecolors='black',linewidths=.3)
        ax.set_title(kind+' — '+name,fontsize=12);ax.set_xlabel('Component 1' if name=='PCA' else 'Map coordinate 1');ax.set_ylabel('Component 2' if name=='PCA' else 'Map coordinate 2');ax.legend(fontsize=8,frameon=False);ax.spines[['top','right']].set_visible(False)
        fig.savefig(dst/(name.lower()+'.png'),dpi=180);plt.close(fig)
        np.savetxt(dst/(name.lower()+'.csv'),z,delimiter=',',header='x,y',comments='')
        print(name,'trustworthiness_5=%.4f'%m['trustworthiness_5'],'overlap_5=%.4f'%m['neighbor_overlap_5'])
    save_json(dst/'ids.json',ids);save_json(dst/'report.json',report)
    # Fit only training rows when studying deployment transforms.
    if a.input:
        train=labels=='train';test=~train
        fitted=PCA(n_components=2,svd_solver='full').fit(x[train]);new=fitted.transform(x[test])
        np.savez(dst/'pca-train-only.npz',mean=fitted.mean_,components=fitted.components_,heldout=new)
        u=umap.UMAP(n_neighbors=min(a.neighbors,int(train.sum())-1),metric='cosine',random_state=a.seed,transform_seed=a.seed,n_jobs=1).fit(x[train])
        np.save(dst/'umap-heldout.npy',u.transform(x[test]),allow_pickle=False)
    print('pca_variance=%.4f'%report['pca_variance']);print('saved',a.out)
if __name__=='__main__':main()
