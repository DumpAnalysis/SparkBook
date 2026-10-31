import numpy as np
from common import pool,normalize
h=np.array([[[2.,0.],[0.,4.],[100.,100.]]]);m=np.array([[1,1,0]])
good=pool(h,m);bad=h.mean(1)
print('masked_mean',good.tolist());print('unmasked_mean',bad.tolist())
print('unit_vector',normalize(good).round(6).tolist())
try:normalize([[0.,0.]])
except ValueError as e:print('zero_vector_rejected',str(e))
