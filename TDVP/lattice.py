import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg.interface import IdentityOperator
from itertools import product
# local imports
from .tools import SuperOperator

__all__ = ["Lattice_TDVP"]

class Lattice_TDVP(object):
    def __init__(self,h_loc,D=10,d=2,A0=None):
        self._h = h.reshape((d,d,d,d))
        self._D = D
        self._d = d
        
        if A0 is not None:
            self._A = A0.copy()
        else:
            self._A = np.random.normal(0,1,size=(d,D,D))
        
        self.canonical_form()
        
        
    
    @property
    def d(self):
        """Hilbert space dimension."""
        return self._d
    
    @property
    def D(self):
        """Bond dimension"""
        return self._D

    @property
    def h(self):
        """Local Hamiltonian"""
        
        h_view = self._h[...]
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
        u,lam,vh = la.svd(self._A.reshape((d*D,D)),full_matrices=False)

        # # how to normalize singular values?
        # lam /= np.linalg.norm(lam)

        gamma = np.matmul(vh,u.reshape((d,D,D)))

        self._A = ((gamma * np.sqrt(lam)).transpose((0,2,1)) * np.sqrt(lam)).transpose((0,2,1))
        
        self._l = lam
        self._r = lam
        
        self._canonical_form = True       
        
    def solve_B(self):
        D = self.D
        d = self.d
        A = self.A
        Ac = A.conj()
        
        r = self.r
        l = self.l
        
        sqrt_r = np.sqrt(r)
        sqrt_l = np.sqrt(l)
        
        # getting null space
        L = (Ac.transpose(0,2,1) * sqrt_l).reshape((D,d*D))
        VL = la.null_space(L)
        VL = VL.reshape((d,D,D*(d-1)))
        
        # getting transfer operator
        E = IdentityOperator((D**2,D**2)) 
        for s in range(d):
            E += SuperOperator(A[s],Ac[s])


        # getting preliminary matrices
        C = np.zeros((d,d,D,D))
        
        lH = np.zeros(D,D)
        
        for s,t,u,v in product(*(4*[range(d)])):
            X = np.dot(Ac[t].T,Ac[s].T)
            Y = np.dot(A[u],A[v])
            
            # np.dot(X*l,Y) equiv to np.dot(X,np.dot(np.diag(l),Y))
            lH += h[s,t,u,v] * np.dot(X*l,Y)
            
            C[s,t] += h[s,t,u,v] * Y

        h = np.sum(np.diag(lH),r)
        
        diag_ind = np.diag_indices_from(lH)
        
        K0 = lH
        
        K0[diag_ind] -= h*l
        
        # getting K
        K = K0
        TrKr = np.sum(np.diag(K)*r)
        
        while( TrKr < 1e-7):
            K = E.dot(K.ravel()).reshape((D,D)) + K0 
            K[diag_ind] -= TrKr*l
            TrKr = np.sum(np.diag(K)*r)
        
        # getting F
        F = np.zeros((D*(d-1),D))
        
        for s,t in product(*(2*[range(self.d)])):
            Temp = Ac[s].T/sqrt_r
            Temp = (Temp.T * r).T
            Temp = np.dot(C[s,t],Temp)
            Temp = (Temp.T * sqrt_l).T
            Temp = np.dot(VL[t].T.conj(),Temp)
            F += Temp
                
        
        for s in range(d):
            Temp = K.dot(A[s])
            for t in range(d):
                Temp += np.dot(l*Ac[t].T,C[t,s])
            
            Temp = ((Temp*sqrt_r).T / sqrt_l).T
            F += np.dot(VL[s].T.conj(),Temp)


        B = (np.dot(VL,F.T.conj()/sqrt_r).T / sqrt_l).T
        
        return B


