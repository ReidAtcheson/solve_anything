import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import util
import scipy.linalg as la
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging



seed=23439287
rng=np.random.default_rng(seed)

m=2000
k=5
p=30
bands=[1,2,32]
A=sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
A=A.tocsr()
A=A+A.T

e=np.ones(m)
d=A@e
alpha=0.0
A = A - alpha*sp.diags([d],[0],shape=(m,m))


V,eigs = util.spectrum_near_zero(A,30,1e-6)

