import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import util
import scipy.linalg as la
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


seed=23439287
rng=np.random.default_rng(seed)

mx=256
my=256
m=mx*my
k=int(0.01*m)
bands=[1,mx]
A=sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
A=A.tocsr()
A=A+A.T


#util.fiedler(A,k,outer,npower)
v=util.fiedler_lanczos(A,k)

I=np.sign(v.reshape((mx,my)))
plt.imshow(I)
plt.savefig("part.svg")

