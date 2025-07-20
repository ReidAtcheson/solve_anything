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

m=1000
k=10
bands=[1,2,32,128,256]
A=sp.diags([rng.uniform(-1,1,size=m) for _ in bands],bands,shape=(m,m))
A=A.tocsr()
A=A+A.T

e=np.ones(m)
d=A@e
alpha=0.9999
A = A - alpha*sp.diags([d],[0],shape=(m,m))

luA=spla.splu(A)
print(f"nnz(lu(A)) = {luA.L.nnz+luA.U.nnz}, nnz(defl(A)) = {m*k}")



v=rng.uniform(-1,1,size=m)

V,eigs = util.spectrum_near_zero(A,k,rtol=1e-12,p=5)
V,_ = la.qr(V,mode="economic")
print(np.amin(np.abs(eigs)))


x = rng.uniform(-1,1,size=m)
b = A@x


minres_err=[]
def minres_callback(xk):
    minres_err.append(np.linalg.norm(x-xk))


defl_err=[]
def defl_callback(xk):
    defl_err.append(np.linalg.norm(x-xk))


spla.minres(A,b,callback=minres_callback,maxiter=10*A.shape[0],rtol=1e-32)


defl = util.DeflatedMinres(A,V)
defl.solve(b,callback=defl_callback,maxiter=10*A.shape[0],tol=1e-32)

xe,_ = spla.gmres(A,b,rtol=1e-32,M=spla.LinearOperator((m,m),matvec=luA.solve),maxiter=10)


plt.semilogy(minres_err,label="minres")
plt.semilogy(defl_err,label="deflated minres")
plt.legend()
plt.title(f"||lu(A).solve(b) - x|| = {np.linalg.norm(xe - x)}")
plt.xlabel("iterations")
plt.ylabel("||x-xh||")
plt.savefig("errs.svg")

