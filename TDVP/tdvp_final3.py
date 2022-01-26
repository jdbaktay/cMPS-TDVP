#cd Desktop/all/research/code/dMPS-TDVP

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt 

import scipy.sparse.linalg as sla
from scipy.sparse.linalg.interface import IdentityOperator
from tools import Projector,TransferMatrix
import cProfile

from quspin.basis import spin_basis_general,tensor_basis
from quspin.operators import hamiltonian

# dense solver
# from numpy.linalg import solve as solve_system
from scipy.sparse.linalg import bicgstab as solve_system




def check(x,y):
	return la.norm(np.subtract(x, y))

def normalize_lr(l,r):

	l = l/np.trace(l)
	r = r/np.trace(r)

	norm = np.sqrt(np.sum(l.conj()*r))

	r /= norm
	l /= norm

	return l,r

def left_canon(A):
	d, D, _ = A.shape


	#Compute transfer operator

	if d*D**3 < D**6:
		E = TransferMatrix(A)
	else:
		E = np.tensordot(A,A.conj(),axes=(0,0))
		E = E.transpose(0,2,1,3).reshape(D**2,D**2)




	#Compute fixed points and norm coefficient

	[eta] , r = sla.eigs(E,k=1,which="LR")
	[eta2], l = sla.eigs(E.T,k=1,which="LR")

	l = l.conj().reshape(D,D)
	r = r.reshape(D,D)


	l,r = normalize_lr(l,r)

	U,o,p = la.ldl(l,hermitian=False)
	L = (U[p,:].dot(np.sqrt(o))).T.conj()
	Li = calcinverse(L)
	# print('check Li ', check(np.eye(D),Li@L))

	#Redefine r and diagonalize
	r, elr = la.eig(r)

	r = np.diag(r)
	#Normalize and compute AL
	A = A/np.sqrt(eta)
	A = np.matmul(L,np.matmul(A,Li))
	#Recompute AL using unitary gauge freedom to use diagonal r
	A = np.matmul(elr.T.conj(),np.matmul(A,elr))

	#set new fixed points of AL and normalize
	l = np.eye(D,D)
	r /= np.trace(r)

	#Recompute transfer op with AL
	if d*D**3 < D**6:
		O = IdentityOperator(shape=E.shape) - TransferMatrix(A) + Projector(np.diagonal(r),np.diagonal(l))
	else:
		E = np.tensordot(A,A.conj(),axes=(0,0))
		E = E.transpose(0,2,1,3).reshape(D**2,D**2)
		rl = np.outer(r.ravel(),l.T.ravel())

		O = np.eye(*E.shape) - E + rl



	return A, O, l, r


def calcinverse(n):
	u,s,vh = la.svd(n)
	return vh.conj().T@np.diag(1/s)@u.conj().T

def calcnullspace(n):
	D = n.shape[0]
	u,s,vh = la.svd(n, full_matrices=True)
	VR = vh.conj().T[:,D:]
	return VR

def two_site_block(A):
	"""
	the code:
	t = np.tensordot(A, A, axes=(2,1))
	t = t.transpose(1,0,2,3)
	
	1. Given how tensordot works the tensor `t` is calculated as:
	t[s,a,r,b] = {sum over c} A[s,a,c]*A[r,c,b]
	where the new tensor is indexed using the left over indices ordered in terms of the inputs
	
	2. Then to move the s,r indices to middle of tensor the transpose maps:
	t[a,s,r,b] = t[s,a,r,b]

	but notice that if we simply transpose A in the beginning we can get the correct results from tensordot:

	t[a,s,r,b] = {sum over c} A.T[c,a,s]*A[r,c,b]

	"""

	return np.tensordot(A.T, A, axes=(0,1))

def apply_h(t,h):
	D,d = t.shape[:2]
	# move quantum indices to left and bond indices to the right
	# reshape to condense indices into matrix
	t = t.transpose(1,2,0,3).reshape((d*d,D*D))
	# multiply by h living on quantum indices f
	th = h.dot(t)
	# reshape back to two local H-space and two bond space indices
	th = th.reshape((d,d,D,D))
	# transpose back to standard form with quantum indices in the middle
	return th.transpose(2,0,1,3)


# def local_ham(A,h):
# 	t = two_site_block(A)
# 	tc= t.conj()
# 	th = apply_h(t,h)
# 	t = np.tensordot(th, tc,axes=([1,2],[1,2]))
# 	t = t.transpose(0,2,1,3)
# 	t = t.reshape(D*D,D*D)
# 	return np.ascontiguousarray(t)

def calcenergy(A,l,r,h):
	# l, r = l.T.reshape(D*D), r.reshape(D*D)

	t = two_site_block(A)
	tc= t.conj()
	th = apply_h(t,h)
	th = np.tensordot(l.T,th,axes=(1,0))
	th = np.tensordot(th,r,axes=(3,0))

	return np.sum(th*tc)

def projector(A, th, Lh, Rh, r, l, h):
	Ac, Lh, Rh = A.conj(), Lh.reshape(D,D).T, Rh.reshape(D,D)    # yeah, we do!

	F = np.tensordot(Lh, A, axes=(1,1))
	F = np.tensordot(F, r, axes=(2,0))
	
	t = np.tensordot(th, r, axes=(3,0))

	F += np.tensordot(t, Ac, axes=([2,3],[0,2]))
	F += np.tensordot(Ac, t, axes=([1,0],[0,1]))
	F += np.tensordot(A, Rh, axes=(2,0)).transpose(1,0,2)

	F = F.transpose(1,0,2)
	F = F.reshape(d*D,D)
	return F

def TDVP(A,dt,O,l,r,h):
	d, D, _ = A.shape

	ri = np.diag(1/(np.diagonal(r)))

	# this calculation is order D^2
	t = two_site_block(A)
	th = apply_h(t,h)
	tc = t.conj()

	# lth = np.tensordot(l.T,th,axes=(1,0))
	lth = th
	hL = np.tensordot(tc,lth,axes=([0,1,2],[0,1,2])).T.ravel()

	rth = np.tensordot(th,r,axes=(3,0))
	hR = np.tensordot(rth,tc,axes=([1,2,3],[1,2,3])).ravel()

	Lh,*_ = solve_system(O.T,hL)
	Rh,*_ = solve_system(O,hR)


	F = projector(A, th, Lh, Rh, r, l, h)

	VR = calcnullspace(A.conj().reshape(d*D,D).T)
	# print('R*VR', la.norm(A.conj().reshape(d*D,D).T.dot(VR)))
	# print('VR+*VR', check(VR.T.conj().dot(VR), np.eye(D*(d-1))))

	Adot = VR@(VR.conj().T)@F@ri
	Adot = Adot.reshape(d,D,D)

	# only works for left canonical form (FIX for general gauge)
	Adotnorm = np.tensordot(Adot, Adot.conj(), axes=([0,1],[0,1]))
	Adotnorm = np.tensordot(Adotnorm, r, axes=([0,1],[0,1]))
	print('(1|B|r) {:10.5e}    '.format(Adotnorm.real), end='')

	A -= dt*Adot



	return A,Adotnorm

L = 1
g = 0.01
basis_ring = spin_basis_general(L,pauli=True)
basis = tensor_basis(basis_ring,basis_ring)

J_ring_list = [[-g,i,(i+1)%L] for i in range(L-1)]
J_rung_list = [[-g,i,i] for i in range(L)]
h_list = [[-(1-g),i] for i in range(L)]

static = [["zz|",J_rung_list],["|zz",J_ring_list],["z|z",J_rung_list],
			["x|",h_list],["|x",h_list]]

h = hamiltonian(static,[],basis=basis,dtype=np.float64)
D, d, dt, N = 10, basis_ring.Ns, 0.001, 2500



A = np.random.normal(0,1,size=(d,D,D)) + 1j * np.random.normal(0,1,size=(d,D,D))


energy = []
Adotnorm = 1
i = 0
while(Adotnorm>1e-7):

	A,O,l,r = left_canon(A)
	lam = np.diagonal(r).real

	ene = calcenergy(A,l,r,h)
	A,Adotnorm = TDVP(A,dt,O,l,r,h)

	energy.append(ene)
	print('ene {} {:5.2e} {:10.7f} {:10.7f}'.format(i, dt, ene.real, -np.sum(lam*np.log(lam))))
	i += 1

print('Parameters:','D =',D,'d =',d,'dt =',dt,'N =',N)

energy = np.array(energy)
plt.plot(energy.real)
plt.show()