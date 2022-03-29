import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigs,bicg,gmres
from scipy.sparse.linalg.interface import IdentityOperator
from itertools import product
# local imports
from .tools import SuperOperator,TransferMatrix,Projector
# debugging
import matplotlib.pyplot as plt

__all__ = ["Lattice_TDVP"]

class Lattice_TDVP(object):
    def __init__(self,h_loc,D=10,d=2,A0=None):
        self._h_loc = h_loc.reshape((d,d,d,d))
        self._D = D
        self._d = d
        
        if A0 is None:
            A0 = np.random.normal(0,1,size=(d*D,D))+1j*np.random.normal(0,1,size=(d*D,D))
            A0,_ = la.qr(A0.reshape((-1,D)),mode="economic")
            A0 = A0.reshape((d,D,D))

        self._A = A0.copy()
        self._canonical_form=False
        
    @property
    def En(self):
        self.canonical_form()
        return self._En
    
    
    @property
    def d(self):
        """Hilbert space dimension."""
        return self._d
    
    @property
    def D(self):
        """Bond dimension"""
        return self._D

    @property
    def h_loc(self):
        """Local Hamiltonian"""
        
        h_view = self._h_loc[...]
        h_view.setflags(write=False)
        return h_view

    @property
    def A(self):
        self.canonical_form()
        
        A_view = self._A[...]
        
        A_view.setflags(write=False)
        
        return A_view
    
    @property
    def l(self):
        self.canonical_form()
        
        l_view = self._l[:]
        
        l_view.setflags(write=False)
        
        return l_view

    @property
    def r(self):
        self.canonical_form()
        
        r_view = self._r[:]
        
        r_view.setflags(write=False)
        
        return r_view

    def canonical_form(self):
        d = self.d
        D = self.D

        if not self._canonical_form:

            A = self._A

            E_op = TransferMatrix(A)


            [e],l = eigs(E_op.T,k=1,which="LR",maxiter=100000)
            [e],r = eigs(E_op  ,k=1,which="LR",maxiter=100000)

            A = A/np.sqrt(e)

            l = l.reshape((D,D))
            l /= np.trace(l)

            L = la.cholesky(l)
            L_inv = la.inv(L)

            A = np.matmul(L,np.matmul(A,L_inv))

            r = r.reshape((D,D))
            r /= np.trace(r)

            p,v = la.eigh(r)

            self._A = np.matmul(v.T.conj(),np.matmul(A,v))

            self._r = np.diag(p) / p.sum()
            self._l = np.diag(np.ones_like(p))

            self._En = np.einsum("qij,rjk,skm,tmi,qrst,k->",A,A,Ac,Ac,self.h_loc,p)

            self._canonical_form = True

    
    def solve_B(self):
        D = self.D
        d = self.d
        A = self.A
        Ac = A.conj()
        h_loc = self.h_loc

        r = self.r
        l = self.l


        if d*D**3 <= D**6:
            E = TransferMatrix(A)
            O = IdentityOperator(shape=E.shape) - E + Projector(np.diagonal(self.r),np.diagonal(self.l))
        else:
            E = np.tensordot(A,A.conj(),axes=(0,0))
            E = E.transpose(0,2,1,3).reshape(D**2,D**2)
            rl = np.outer(r.ravel(),l.T.ravel())

            O = np.eye(*E.shape) - E + rl


        th = np.tensordot(A.T, A, axes=(0,1))

        th = np.tensordot(th,self.h_loc,axes=([1,2],[0,1])).transpose(0,2,3,1)

        Ac = A.conj()
        lAc = np.tensordot(l.T,Ac,axes=(1,0))
        Acr = np.tensordot(Ac,r,axes=(2,0))

        F1 = np.tensordot(th,Acr,axes=([2,3],[0,2]))

        hR = np.tensordot(F1,Ac,axes=([1,2],[0,2]))
        F1 = np.tensordot(l.T,F1,axes=(1,0))

        F2 = np.tensordot(lAc,th,axes=([1,0],[0,1]))

        hL = np.tensordot(F2,Ac,axes=([0,1],[1,0]))
        F2 = np.tensordot(F2,r,axes=(2,0))



        self._En = np.sum(np.diagonal(hL)*np.diagonal(r))
        Lh,*_ = solve_system(O.T,hL)
        Rh,*_ = solve_system(O,hR)

        Rh = Rh.reshape(D,D)
        Lh = Lh.reshape(D,D)

        F3 = np.matmul(l.T,np.matmul(A,Rh))
        F4 = np.matmul(Lh.T,np.matmul(A,l))




"""
    def solve_B(self):
        D = self.D
        d = self.d
        A = self.A
        Ac = A.conj()
        h_loc = self.h_loc

        r = self.r
        l = self.l

        sqrt_r = np.sqrt(r)
        sqrt_l = np.sqrt(l)

        # getting null space
        L = ((Ac.transpose(0,2,1)*sqrt_l).transpose(1,0,2)).reshape((D,d*D))
        VL = la.null_space(L)
        VL = VL.reshape((d,D,D*(d-1)))
        # VL = VL.transpose((1,0,2))

        # getting transfer operator
        E = SuperOperator(A[0],Ac[0])
        for s in range(1,d,1):
            E += SuperOperator(A[s],Ac[s])

        # getting preliminary matrices
        C = np.zeros((d,d,D,D),dtype=np.complex128)
        lH = np.zeros((D,D),dtype=np.complex128)

        for s,t,u,v in product(*(4*[range(d)])):
            X = np.dot(Ac[s],Ac[t]).T # = (A[s],A[t]).T.conj()
            Y = np.dot(A[u],A[v])

            # np.dot(X*l,Y) equiv to np.dot(X,np.dot(np.diag(l),Y))
            lH += h_loc[s,t,u,v] * np.dot(X*l,Y)

            C[s,t] += h_loc[s,t,u,v] * Y

        h = np.sum(np.diagonal(lH)*r)
        diag_ind = np.diag_indices_from(lH)
        lH[diag_ind] -= h*l

        # getting K
        U = IdentityOperator(shape=E.shape) - E + Projector(r,l)

        # K,e = bicg(U.H,lH.ravel())
        K,e = gmres(U.H,lH.ravel())
        K = K.reshape((D,D))
        K = K.T.conj()


        # getting F
        F = np.zeros((D*(d-1),D),dtype=np.complex128)

        for s in range(d):
            Temp = (VL[t].T.conj()) * sqrt_l
            for t in range(d)
                
        for s,t in product(*(2*[range(d)])):
            
            Temp = np.dot(Temp,C[s,t])
            Temp *= r
            Temp = np.dot(Temp,Ac[s].T)
            Temp /= sqrt_r
            F += Temp

        for s in range(d):
            Temp = K.dot(A[s])
            for t in range(d):
                Temp += np.dot((Ac[t].T)*l,C[t,s])

            F += np.dot(VL[s].T.conj()/sqrt_l,Temp*sqrt_r)


        B = np.matmul(VL,F.T.conj())
        B = ((B / sqrt_r).transpose(0,2,1) / sqrt_l).transpose(0,2,1)

        return B
"""
    def imag_time_step(self,dtau):
        B = self.solve_B()

        self._A -= dtau*B

        self._canonical_form=False

        return dtau*B


