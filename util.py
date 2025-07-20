# coding: utf-8
"""Experimental deflated MINRES solver.

This module provides a small class :class:`DeflatedMinres` implementing a
projection based deflation technique for the MINRES method.  The deflation
subspace must be supplied by the user.  The implementation mirrors the
``DeflatedCG`` example provided in the user prompt but uses ``minres`` and
does not assume that the matrix is positive definite (only symmetric).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from typing import Callable, Optional
import logging

logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)  


import numpy as np
import scipy.sparse as sp


class CountingLinearOperator(spla.LinearOperator):
    """
    A LinearOperator wrapper for a sparse matrix that counts matvec calls.
    """
    def __init__(self, A: sp.spmatrix):
        """
        Parameters
        ----------
        A : scipy.sparse.spmatrix
            The sparse matrix to wrap.
        """
        # Ensure CSR for efficient multiplication
        self.A = A.tocsr()
        # Counter for matvec calls
        self._counter = 0
        # Initialize the base LinearOperator
        super().__init__(dtype=A.dtype, shape=A.shape)

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Perform A @ x and increment the counter.
        """
        self._counter += 1
        return self.A.dot(x)

    def _rmatvec(self, x: np.ndarray) -> np.ndarray:
        """
        Optionally handle transpose multiplications and count them separately.
        """
        # You could track these separately if desired
        return self.A.T.dot(x)

    def get_counter(self) -> int:
        """
        Retrieve the number of times matvec has been called.
        """
        return self._counter

def graph_laplacian(A: sp.spmatrix) -> sp.spmatrix:
    """
    Compute the (combinatorial) graph Laplacian L for a directed or undirected graph,
    using out‐degrees on the diagonal and -1 for each edge.
    
    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Square adjacency matrix.  Any nonzero entry is treated as an edge of weight 1.
    
    Returns
    -------
    L : scipy.sparse.csr_matrix
        The unweighted graph Laplacian L = D_out - B, where B is the binary adjacency.
    """
    # Ensure CSR for efficient row‐ops
    A = A.tocsr()
    
    # Binarize: any nonzero entry → 1
    B = A.copy()
    B.data[:] = 1
    
    # Out‐degree = sum of each row of B
    deg = np.array(B.sum(axis=1)).ravel()
    
    # Build D_out
    D = sp.diags(deg)
    
    # Laplacian L = D - B
    L = D - B
    
    return L.tocsr()


class DeflatedMinres:
    """Solve ``Ax=b`` using deflated MINRES.

    Parameters
    ----------
    A:
        Real symmetric matrix or ``LinearOperator``.
    V:
        Orthonormal columns spanning the deflation space.
    """

    def __init__(self, A: np.ndarray | spla.LinearOperator, V: np.ndarray) -> None:
        self.A = A
        self.V = V
        self.VAV = V.T @ (A @ V)

    # projection onto range(A V)
    def _pi(self, x: np.ndarray) -> np.ndarray:
        Av = self.A @ x
        return self.V @ la.solve(self.VAV, self.V.T @ Av)

    # adjoint projection
    def _pistar(self, x: np.ndarray) -> np.ndarray:
        return self.A.T @ (self.V @ la.solve(self.VAV, self.V.T @ x))

    def _reconstruct(self, b: np.ndarray, xh: np.ndarray) -> np.ndarray:
        xl = self.V @ la.solve(self.VAV, self.V.T @ b)
        xr = xh - self._pi(xh)
        return xl + xr

    def solve(
        self,
        b: np.ndarray,
        *,
        tol: float = 1e-12,
        maxiter: Optional[int] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> tuple[np.ndarray, int]:
        """Solve the linear system ``Ax=b``.

        Parameters
        ----------
        b:
            Right-hand side vector.
        tol:
            Relative tolerance for ``minres``.
        maxiter:
            Maximum number of iterations.
        callback:
            Optional callback receiving the reconstructed iterate.
        """

        n = b.shape[0]
        dtype = getattr(self.A, "dtype", None) or b.dtype

        def mv(x: np.ndarray) -> np.ndarray:
            return self.A @ (x - self._pi(x))

        op = spla.LinearOperator((n, n), matvec=mv, dtype=dtype)

        def cb(x: np.ndarray) -> None:
            if callback is not None:
                callback(self._reconstruct(b, x))

        rhs = b - self._pistar(b)
        xh, info = spla.minres(op, rhs, rtol=tol, maxiter=maxiter, callback=cb)
        x = self._reconstruct(b, xh)
        return x, info



def arnoldi(A: spla.LinearOperator, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    m, n = A.shape
    if m != n:
        raise ValueError("A must be square")
    V = np.zeros((m, k + 1), dtype=v.dtype)
    H = np.zeros((k + 1, k), dtype=v.dtype)
    beta = np.linalg.norm(v)
    if beta == 0:
        raise ValueError("Starting vector must be nonzero")
    V[:, 0] = v / beta
    for j in range(k):
        w = A @ V[:, j]
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i].conj(), w)
            w = w - H[i, j] * V[:, i]
        for i in range(j + 1):
            h = np.dot(V[:, i].conj(), w)
            H[i, j] += h
            w = w - h * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] != 0:
            V[:, j + 1] = w / H[j + 1, j]
        else:
            V[:, j + 1] = 0
            H[j + 1:, j] = 0
            break
    return V, H



#Computes rayleigh ritz eigenvectors from a subspace V.
#Finds the most converged eigenvector and returns it
#alongside a new starting vector and all residuals
def rayleigh_ritz(A,V):
    VAV = V.T @ A.matmat(V)
    eigs,K = la.eigh(VAV)
    W = V @ K
    return W,eigs

def best_converged(A,W,eigs):
    residuals = [np.linalg.norm(A.matvec(W[:,i]) - eigs[i]*W[:,i])/np.abs(eigs[i]) for i in range(W.shape[1])]
    i = np.argmin(residuals)
    return W[:,i],eigs[i],residuals[i]


    

#Computes smallest absolute eigenpairs for a matrix
def smallest_eigenpairs(A: spla.LinearOperator, v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    m,n=A.shape
    assert m==n
    V,H = arnoldi(A,v,2*k)
    converged=0
    P = np.zeros((m,k))

    minres_its=0
    def solveA(b):
        nonlocal minres_its
        b=b.reshape(-1,1)
        if converged>0:
            b=b-P[:,:converged]@(P[:,:converged].T@b)
        #Determined this empirically. Seems to "mostly work"
        maxiter=2*b.shape[0]
        tol=1e-32

        def callback(xk):
            nonlocal minres_its
            minres_its+=1

        def evalA(y):
            return A@y
        x,_ = spla.minres(spla.LinearOperator((m,m),matvec=evalA),b,maxiter=maxiter,rtol=tol,callback=callback)
        return x

    invA = spla.LinearOperator((m,m),matvec=solveA)




    it=0
    while converged<k:
        W,eigs = rayleigh_ritz(spla.aslinearoperator(A),V)
        w,e,best_residual=best_converged(spla.aslinearoperator(A),W,eigs)
        #New starting vector for arnoldi
        es = np.zeros(eigs.size)
        es[np.abs(eigs)<=np.median(np.abs(eigs))]=1.0
        v=W@es
        logger.info("it=%s, Current eigenvalue: %s, current best residual: %s, converged: %s. minres_its: %s",it,e,best_residual,converged,minres_its)
        it+=1
        if best_residual<1e-5:
            P[:,converged] = w
            converged+=1
            #Re-orthogonalize
            P[:,:converged],_ = rayleigh_ritz(spla.aslinearoperator(A),P[:,:converged])
            P[:,:converged],_ = la.qr(P[:,:converged],mode="economic")
            if converged==k:
                break

        #If any eigenvectors have converged, project them out
        #of our starting vector
        if converged>0:
            v = v - P[:,:converged] @ (P[:,:converged].T @ v)
        #Note that this arnoldi factorization isn't a true
        #factorization because invA isn't linear. but the orthogonalization
        #of V is still correct
        V,_ = arnoldi(invA,v,2*k)
    V,eigs = rayleigh_ritz(spla.aslinearoperator(A),P)
    return V,eigs





#Simple qr/power iteration approach
def smallest_eigenpairs_power(A: spla.LinearOperator, k: int) -> tuple[np.ndarray, np.ndarray]:
    rng=np.random.default_rng(42)
    m,n=A.shape
    assert m==n
    converged=0
    V = rng.uniform(-1,1,size=(m,2*k))
    V,_ = la.qr(V,mode="economic")


    minres_its=0
    def solveA(b):
        nonlocal minres_its
        b=b.reshape(-1,1)
        maxiter=2*b.shape[0]
        tol=1e-32

        def callback(xk):
            nonlocal minres_its
            minres_its+=1

        def evalA(y):
            return A@y
        x,_ = spla.minres(spla.LinearOperator((m,m),matvec=evalA),b,maxiter=maxiter,rtol=tol,callback=callback)
        return x


    P,eigs = rayleigh_ritz(spla.aslinearoperator(A),V)
    residuals = np.array([np.linalg.norm(A@P[:,i]-eigs[i]*P[:,i])/abs(eigs[i]) for i in range(P.shape[1])])
    ids = np.argsort(np.abs(eigs))

    while np.amax(residuals[ids[:k]])>1e-6:
        logging.info("best residual: %s, worst residual %s, best eig %s, worst eig %s, minres iters %s",np.amin(residuals[ids[:k]]),np.amax(residuals[ids[:k]]),eigs[ids[0]],eigs[ids[k-1]],minres_its)

        for i in range(V.shape[1]):
            V[:,i]=solveA(V[:,i])
        V,_ = la.qr(V,mode="economic")
        P,eigs = rayleigh_ritz(spla.aslinearoperator(A),V)
        residuals = np.array([np.linalg.norm(A@P[:,i]-eigs[i]*P[:,i])/abs(eigs[i]) for i in range(P.shape[1])])
        ids = np.argsort(np.abs(eigs))

    return P[:,ids[0:k]],eigs[ids[0:k]]




# inverse iteration on (A-sigma*I).
# Keep p vectors and iterate until k eigenvalues 
# have converged. Optionally Keep K projected out
# Assume A is symmetric
def shifted_inverse_power(A,sigma,k,p,K,rtol):
    rng=np.random.default_rng(42)
    m,n=A.shape
    assert m==n
    assert p>=k
    V = rng.uniform(-1,1,size=(m,p))
    V,_ = la.qr(V,mode="economic")

    def solveA(b):
        if K is not None:
            b = b - K @ (K.T @ b)
        b=b.reshape(-1,1)
        maxiter=b.size
        tol=1e-32

        def evalA(y):
            return A@y - sigma*y
        x,_ = spla.minres(spla.LinearOperator((m,m),matvec=evalA),b,maxiter=maxiter,rtol=tol)
        return x
    P,eigs = rayleigh_ritz(spla.aslinearoperator(A),V)
    residuals = np.array([np.linalg.norm(A@P[:,i]-eigs[i]*P[:,i])/abs(eigs[i]) for i in range(P.shape[1])])
    ids = np.argsort(residuals)
    worst = np.amax(residuals[ids[0:k]])

    while worst > rtol:
        logger.info("shift: %s. worst residual %s",sigma,worst)
        worst = np.amax(residuals[ids[0:k]])
        for i in range(V.shape[1]):
            V[:,i]=solveA(V[:,i])
        V,_ = la.qr(V,mode="economic")
        P,eigs = rayleigh_ritz(spla.aslinearoperator(A),V)
        residuals = np.array([np.linalg.norm(A@P[:,i]-eigs[i]*P[:,i])/abs(eigs[i]) for i in range(P.shape[1])])
        ids = np.argsort(residuals)
        worst = np.amax(residuals[ids[0:k]])
        V = P

    return P[:,ids[0:k]],eigs[ids[0:k]]





# Iteratively apply shifted_inverse_power
# with successively increasing shifts expanding
# around 0 until reached desired number of eigenvectors
def spectrum_near_zero(A,k,rtol=2e-12,p=2):
    assert k>=p
    V,eigs = shifted_inverse_power(A,0.0,p,k,None,rtol)

    while eigs.size<k:
        ids=np.argsort(eigs)
        eigs = eigs[ids]
        V = V[:,ids]
        logger.info("eigs = %s",eigs)
        ipos = sum(eigs>0)
        ineg = sum(eigs<0)
        #decide whether to use positive or negative shift
        shift = 0.0
        if ipos>ineg and ineg>0:
            shift = eigs[0]
        if ineg > ipos and ipos>0:
            shift = eigs[eigs.size-1]
        if ipos==ineg:
            shift = eigs[0]

        W,neigs = shifted_inverse_power(A,shift,p,k,V,rtol=rtol)
        V = np.concatenate((V,W),axis=1)
        eigs = np.concatenate((eigs,neigs))
    ids=np.argsort(eigs)
    eigs = eigs[ids]
    V = V[:,ids]
    return V,eigs





#Warm up fiedler calculation with a fixed number of power iterations
#on shifted graph laplacian and follow up with some number of inverse iterations
def fiedler(A,k,outer,npower):
    assert k>=2
    rng=np.random.default_rng(42)
    L = graph_laplacian(A)
    m,_ = L.shape
    normL = spla.norm(L,ord=1)
    #Will use power iteration on this. smallest eigenvalues
    #of L become largest eigenvalues for L-norm(L)*I
    shiftL = L - normL*sp.eye(m)
    nevals=0
    V=rng.uniform(-1,1,size=(m,k))
    V,_ = la.qr(V,mode="economic")    
    eigs = None
    v = None
    u = np.ones((m,1))/np.sqrt(m)
    for i in range(outer):
        for _ in range(npower):
            #Project out nullspace
            V = V - u @ (u.T @ V)
            V = shiftL@V
            V,_ = la.qr(V,mode="economic")
            nevals+=k
        V,eigs = rayleigh_ritz(spla.aslinearoperator(L),V)
        ids = np.argsort(eigs)
        V = V[:,ids]
        eigs = eigs[ids]

        cL = CountingLinearOperator(L - eigs[0]*sp.eye(m))
        #Polynomially precondition the approximate fiedler vector
        for j in range(k):
            V[:,j],_ = spla.minres(cL,V[:,j] - u@(u.T@V[:,j]),maxiter=100)
        #This is just convenient but we can probably find a way to 
        #avoid this second orthgonalization and rayleigh-ritz projection
        V,_ = la.qr(V,mode="economic")
        V,eigs = rayleigh_ritz(spla.aslinearoperator(L),V)
        ids = np.argsort(eigs)
        V = V[:,ids]
        eigs = eigs[ids]


        nevals+=k+cL.get_counter()

        l = eigs[0]
        v = V[:,0]


        relres = np.linalg.norm(L@v - l*v)/np.abs(l)
        logger.info("Shifted power warmup phase iter: %s, fiedler eig: %s, relres: %s, laplacian evals: %s CG iters: %s",i,l,relres,nevals,cL.get_counter())



#Warm up fiedler calculation with a fixed number of power iterations
#on shifted graph laplacian and follow up with some number of inverse iterations
def fiedler_lanczos(A,k):
    assert k>=2
    rng=np.random.default_rng(42)
    L = graph_laplacian(A)
    m,_ = L.shape
    normL = spla.norm(L,ord=1)
    v=rng.uniform(-1,1,size=(m,1))
    #p=np.ones((m,1))/np.sqrt(m)
    #v = v - p @ (p.T @ v)
    cL = CountingLinearOperator(L)
    #v,_ = spla.cg(cL,v)
    #v = v - p @ (p.T @ v)
    cshiftL = CountingLinearOperator(L-normL*sp.eye(m))
    eigs,V = spla.eigsh(cshiftL,v0=v,k=2,ncv=k,tol=1e-6)
    V,_ = la.qr(V,mode="economic")
    V,eigs = rayleigh_ritz(spla.aslinearoperator(L),V)
    ids=np.argsort(eigs)
    eigs=eigs[ids]
    V=V[:,ids]

    l=eigs[1]
    v=V[:,1]
    relres = np.linalg.norm(L@v - l*v)/l
    logger.info("CG warmup evals %s lanczos evals %s. eig %s, relres %s",cL.get_counter(),cshiftL.get_counter(),l,relres)
    return v


