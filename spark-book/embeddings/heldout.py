import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--maps',required=True);a=p.parse_args();root=Path(a.maps)
x=np.load(root/'pca-train-only.npz',allow_pickle=False);u=np.load(root/'umap-heldout.npy',allow_pickle=False)
print('PCA components',list(x['components'].shape),'heldout',list(x['heldout'].shape))
print('UMAP heldout',list(u.shape),'finite',bool(np.isfinite(u).all()))
print('t-SNE sklearn transform: unavailable; refitting changes the map')
