import scipy.sparse as sp
import numpy as np
import util
import scipy.linalg as la

import numpy as np

def find_closest_entries(large: np.ndarray, small: np.ndarray) -> np.ndarray:
    """
    For each element in `small`, find the element in `large` with the minimum absolute difference.
    Returns an array of the same shape as `small`.
    """
    # Flatten the large array for efficient searching
    large_flat = large.ravel()
    
    # Prepare an output array of the same shape as 'small'
    result = np.empty_like(small)
    
    # For each element in 'small', find the closest in 'large_flat'
    for idx, val in np.ndenumerate(small):
        diffs = np.abs(large_flat - val)
        min_idx = np.argmin(diffs)
        result[idx] = large_flat[min_idx]
    
    return result



seed=23439287
rng=np.random.default_rng(seed)

m=512
k=16
bands=[1,2,32]
A=sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
A=A.tocsr()
A=A+A.T

v=rng.uniform(-1,1,size=m)

V,eigs = util.smallest_eigenpairs(A,v,k)


print(np.linalg.norm(V.T @ V - np.eye(V.shape[1])))
print(np.linalg.norm(A@V[:,0] - eigs[0]*V[:,0]))
print(eigs)
print(find_closest_entries(la.eigvalsh(A.toarray()),eigs))
