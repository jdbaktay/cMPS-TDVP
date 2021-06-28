import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg.interface import IdentityOperator
from itertools import product
# local imports
from .tools import SuperOperator
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

        self._A = A0.copy()
        self._canonical_form=False
        
    @property
    def En(self):
        return self._En.real
    
    
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
        if not self._canonical_form:
            self.canonical_form()
        
        A_view = self._A[...]
        
        A_view.setflags(write=False)
        
        return A_view
    
    @property
    def l(self):
        if not self._canonical_form:
            self.canonical_form()
        
        l_view = self._l[:]
        
        l_view.setflags(write=False)
        
        return l_view

    @property
    def r(self):
        if not self._canonical_form:
            self.canonical_form()
        
        r_view = self._r[:]
        
        r_view.setflags(write=False)
        
        return r_view

    def canonical_form(self):
        d = self.d
        D = self.D

        if not self._canonical_form:

            A,_ = la.qr(self._A.reshape((-1,D)),mode="economic")
            A = A.reshape((d,D,D))
            
            E_op = SuperOperator(A[0],A[0].conj())
            for s in range(1,d,1):
                E_op += SuperOperator(A[s],A[s].conj())

            [e],r = eigs(E_op,k=1,which="LR")

            r = r.reshape((D,D))
            r /= np.trace(r)
            p,v = la.eigh(r)

            self._A = np.matmul(v.T.conj(),np.matmul(A,v))/np.sqrt(e)

            self._r = p.ravel() / p.sum()
            self._l = np.ones_like(p)

            self._En = 0

            A = self._A
            for s,t,u,v in product(*(4*[range(d)])):
                X = np.dot(A[s],A[t])
                Y = np.dot(A[u],A[v])
                self._En += self._h_loc[s,t,u,v] * np.trace(np.dot(X.T.conj(),Y) * self._r)

            self._canonical_form = True

    
                
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
        L = (Ac.transpose(0,2,1) * sqrt_l).reshape((D,d*D))
        VL = la.null_space(L)
        VL = VL.reshape((d,D,D*(d-1)))

        # getting transfer operator
        E = SuperOperator(A[0],Ac[0])
        for s in range(1,d,1):
            E += SuperOperator(A[s],Ac[s])

        # getting preliminary matrices
        C = np.zeros((d,d,D,D),dtype=np.complex128)
        lH = np.zeros((D,D),dtype=np.complex128)

        for s,t,u,v in product(*(4*[range(d)])):
            X = np.dot(Ac[t].T,Ac[s].T)
            Y = np.dot(A[u],A[v])

            # np.dot(X*l,Y) equiv to np.dot(X,np.dot(np.diag(l),Y))
            lH += h_loc[s,t,u,v] * np.dot(X*l,Y)

            C[s,t] += h_loc[s,t,u,v] * Y

        h = np.sum(np.diag(lH)*r)

        diag_ind = np.diag_indices_from(lH)

        K0 = lH.copy()

        K0[diag_ind] -= h*l

        # getting K
        K = np.random.normal(0,1,size=(D,D))
        # np.trace(K.dot(np.diag(r)))
        TrKr = np.sum(np.diag(K)*r)

        while( np.abs(TrKr) > 1e-7):

            K = E.dot(K.ravel()).reshape((D,D)) + K0 
            K[diag_ind] -= TrKr*l
            TrKr = np.sum(np.diag(K)*r)
            # print(TrKr,K)

        # getting F
        F = np.zeros((D*(d-1),D),dtype=np.complex128)

        for s,t in product(*(2*[range(self.d)])):
            Temp = (VL[t].T.conj()) * sqrt_l
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


        B = (np.matmul(VL,F.T.conj()/sqrt_r).transpose(0,2,1) / sqrt_l).transpose(0,2,1)

        return B

    def imag_time_step(self,dtau):
        B = self.solve_B()

        self._A -= dtau*B

        self._canonical_form=False
        self.canonical_form()

        return dtau*B


