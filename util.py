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
    residuals = [np.linalg.norm(A.matvec(W[:,i]) - eigs[i]*W[:,i]) for i in range(W.shape[1])]
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
        #Determined this empirically. Seems to "mostly work"
        maxiter=2*b.shape[0]
        tol=1e-32

        def callback(xk):
            nonlocal minres_its
            minres_its+=1
        x,_ = spla.minres(A,b,maxiter=maxiter,rtol=tol,callback=callback)
        return x

    invA = spla.LinearOperator((m,m),matvec=solveA)




    it=0
    while converged<k:
        if converged>0:
            V = V - P[:,:converged] @ (P[:,:converged].T @ V)
            V,_ = la.qr(V,mode="economic")
        W,eigs = rayleigh_ritz(spla.aslinearoperator(A),V)
        w,e,best_residual=best_converged(spla.aslinearoperator(A),W,eigs)
        #New starting vector for arnoldi
        v = np.sum(W,axis=1)
        logger.info("it=%s, Current eigenvalue: %s, current best residual: %s, converged: %s. minres_its: %s",it,e,best_residual,converged,minres_its)
        it+=1
        if best_residual<1e-9:
            P[:,converged] = w
            converged+=1
            #Re-orthogonalize
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






















