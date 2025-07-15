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

m=10000
k=20
bands=[1,2,32]
A=sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
A=A.tocsr()
A=A+A.T

x = rng.uniform(-1,1,size=m)
b = A@x

defl_err=[np.linalg.norm(x)]
def defl_callback(xk):
    defl_err.append(np.linalg.norm(x-xk))


minres_err=[np.linalg.norm(x)]
def minres_callback(xk):
    minres_err.append(np.linalg.norm(x-xk))


ps=[]
while min(defl_err)>1e-10:
    print(min(defl_err))
    if ps:
        P = np.column_stack(ps)
        P,_ = la.qr(P,mode="economic")
        P,_ = util.rayleigh_ritz(spla.aslinearoperator(A),P)
        defl = util.DeflatedMinres(A,P)
        xh,_ = defl.solve(b,callback=defl_callback,maxiter=2000,tol=1e-32)
        ps.append(xh)
    else:
        xh,_=spla.minres(A,b,callback=defl_callback,maxiter=2000,rtol=1e-32)
        ps.append(xh)


spla.minres(A,b,callback=minres_callback,maxiter=len(defl_err),rtol=1e-32)
plt.semilogy(defl_err,label="inc_deflation")
plt.semilogy(minres_err,label="minres")
plt.semilogy("incremental_deflation.svg")
plt.xlabel("iterations")
plt.ylabel("||x-xh||")
plt.legend()
plt.savefig("inc_errs.svg")
