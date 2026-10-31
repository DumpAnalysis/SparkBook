"""TARGET ONLY: cuDF, cuML and RMM qualification in a fresh GPU process."""
import argparse,json,time
from pathlib import Path
import numpy as np
from lab import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--src',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import cupy as cp, cudf, cuml, rmm
    from cuml.decomposition import PCA
    import rmm.statistics as stats
    from rmm.allocators.cupy import rmm_cupy_allocator
    def counts():
        s=stats.get_statistics()
        if s is None:raise RuntimeError('RMM statistics disabled')
        return {k:int(getattr(s,k)) for k in ['current_bytes','current_count','peak_bytes','peak_count','total_bytes','total_count']}
    # Configure before the first CuPy/cuDF allocation; never reset with live buffers.
    stats.enable_statistics()
    cp.cuda.set_allocator(rmm_cupy_allocator)
    before=counts()
    free_before,total=cp.cuda.runtime.memGetInfo()
    t=time.perf_counter();df=cudf.read_csv(a.src/'requests.csv')
    good=df[df.ok==1]
    g=good.groupby('route').agg({'latency_ms':['count','mean']}).sort_index().to_pandas()
    cp.cuda.runtime.deviceSynchronize();elapsed=time.perf_counter()-t
    expected=json.loads((a.control/'analytics.json').read_text())['groups']
    actual=[]
    for r in expected:
        row=g.loc[r['route']];count=int(row[('latency_ms','count')]);mean=float(row[('latency_ms','mean')])
        assert count==r['count'] and abs(mean-r['mean_ms'])<1e-6
        actual.append(dict(route=r['route'],count=count,mean_ms=mean))
    x=cp.asarray(np.load(a.src/'fixture.npz')['vectors'])
    cp.cuda.runtime.deviceSynchronize();t=time.perf_counter()
    model=PCA(n_components=2,svd_solver='full',output_type='cupy')
    scores=model.fit_transform(x);cp.cuda.runtime.deviceSynchronize();pca_seconds=time.perf_counter()-t
    components=cp.asnumpy(model.components_)
    if components.shape != (2,x.shape[1]):raise RuntimeError('Expected row-form PCA components [2,n_features]')
    projector=components.T@components
    ref=np.load(a.control/'pca.npz')['projection'];error=float(np.linalg.norm(projector-ref))
    assert error<1e-3
    np.save(a.out/'scores.npy',cp.asnumpy(scores))
    free_after,_=cp.cuda.runtime.memGetInfo();after=counts()
    report=dict(cudf=cudf.__version__,cuml=cuml.__version__,rmm=rmm.__version__,
       groups=actual,projector_error=error,table_wall_seconds=elapsed,pca_wall_seconds=pca_seconds,
       rmm_before=before,rmm_after=after,driver_free_before=free_before,driver_free_after=free_after,
       driver_total=total,measurement='single cold diagnostic sample; not a speedup benchmark')
    dump(a.out/'rapids.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
