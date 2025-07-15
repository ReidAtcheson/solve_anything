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

m=5000
k=50
bands=[1,2,32]
A=sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
A=A.tocsr()
A=A+A.T

v=rng.uniform(-1,1,size=m)

V,eigs = util.smallest_eigenpairs(A,v,k)


x = rng.uniform(-1,1,size=m)
b = A@x


minres_err=[]
def minres_callback(xk):
    minres_err.append(np.linalg.norm(x-xk))


defl_err=[]
def defl_callback(xk):
    defl_err.append(np.linalg.norm(x-xk))


spla.minres(A,b,callback=minres_callback,maxiter=A.shape[0],rtol=1e-32)


defl = util.DeflatedMinres(A,V)
defl.solve(b,callback=defl_callback,maxiter=A.shape[0],tol=1e-32)


plt.semilogy(minres_err,label="minres")
plt.semilogy(defl_err,label="deflated minres")
plt.legend()
plt.xlabel("iterations")
plt.ylabel("||x-xh||")
plt.savefig("errs.svg")

