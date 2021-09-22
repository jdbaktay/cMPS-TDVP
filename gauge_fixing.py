import scipy.linalg as la
import numpy as np
from TDVP import SuperOperator
from scipy.sparse.linalg import eigs

D = 3
d = 2

A = np.random.normal(0,1,size=((d,D,D)))+1j*np.random.normal(0,1,size=((d,D,D)))

E_op = SuperOperator(A[0],A[0].conj())
for s in range(1,d,1):
    E_op += SuperOperator(A[s],A[s].conj())


[e],l = eigs(E_op.T,k=1,which="LR")

A = A/np.sqrt(e)

l = l.reshape((D,D))
l /= np.trace(l)


L = la.cholesky(l)
L_inv = la.inv(L)


A = np.matmul(L,np.matmul(A,L_inv))

E_op = SuperOperator(A[0],A[0].conj())
for s in range(1,d,1):
    E_op += SuperOperator(A[s],A[s].conj())

[e],r = eigs(E_op,k=1,which="LR")

r = r.reshape((D,D))
r /= np.trace(r)

r_diag,U_r = la.eigh(r)

A = np.matmul(U_r.T.conj(),np.matmul(A,U_r))

l = np.ones_like(r_diag)
r = r_diag










