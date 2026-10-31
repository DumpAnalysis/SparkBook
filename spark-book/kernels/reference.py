"""CPU teaching oracles and index checks; this module executes no CUDA."""
import math

def grid_indices(n,block,grid):
    if min(n,block,grid)<1:raise ValueError('positive sizes required')
    return [list(range(t,n,block*grid)) for t in range(block*grid)]

def transpose_mapping(rows,cols):
    """Simulate the shared-tile addresses, retaining source identity, not values."""
    writes={}
    for by in range((rows+31)//32):
        for bx in range((cols+31)//32):
            shared={}
            for y in range(8):
                for x in range(32):
                    for j in range(0,32,8):
                        r,c=by*32+y+j,bx*32+x
                        if r<rows and c<cols:shared[y+j,x]=r*cols+c
            for y in range(8):
                for x in range(32):
                    for j in range(0,32,8):
                        r,c=bx*32+y+j,by*32+x
                        if r<cols and c<rows:
                            address=r*rows+c
                            if address in writes:raise ValueError('duplicate store')
                            writes[address]=shared[x,y+j]
    return writes

def tree_sum(values,block,grid):
    if block<1 or block&(block-1):raise ValueError('power of two required')
    partial=[]
    for b in range(grid):
        sums=[sum(values[b*block+t::block*grid]) for t in range(block)]
        step=block//2
        while step:
            for t in range(step):sums[t]+=sums[t+step]
            step//=2
        partial.append(sums[0])
    return partial

def matmul(a,b):
    m,k,n=len(a),len(a[0]),len(b[0])
    if len(b)!=k:raise ValueError('inner dimensions differ')
    return [[sum(a[r][q]*b[q][c] for q in range(k)) for c in range(n)] for r in range(m)]

def rms(values,gamma,epsilon=1e-5):
    if not values or len(values)!=len(gamma) or epsilon<=0:raise ValueError('invalid row')
    scale=1/math.sqrt(sum(x*x for x in values)/len(values)+epsilon)
    return [x*scale*g for x,g in zip(values,gamma)]

def softmax(values):
    if not values or not all(math.isfinite(x) for x in values):raise ValueError('finite nonempty row required')
    maximum=max(values);e=[math.exp(x-maximum) for x in values];total=sum(e)
    return [x/total for x in e]

def demo():
    print('evidence=CPU reference calculations; no GPU execution')
    visits=[i for group in grid_indices(1003,256,2) for i in group]
    print(f'grid_stride_count={len(visits)} unique={len(set(visits))}')
    mapping=transpose_mapping(37,65)
    print(f'transpose_elements={len(mapping)}')
    print('reduction_sum='+str(sum(i%17-8 for i in range(1003))))
    bins=[0]*16
    for i in range(1003):bins[i%16]+=1
    print('histogram='+','.join(map(str,bins)))
    print('matmul='+str(matmul([[1,2,3],[4,5,6]],[[1,0],[0,1],[1,1]])))
    print('rms_zero='+str(rms([0,0,0],[1,1,1])))
    print('softmax_row_sum='+format(sum(softmax([1000,1001,1002])),'.6f'))
if __name__=='__main__':demo()
