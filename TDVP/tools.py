import numpy as np
from scipy.sparse.linalg import LinearOperator

__all__ = ["SuperOperator","Projector"]


class SuperOperator(LinearOperator):
    # operator Kronecker(A,B)
    
    def __init__(self,A=None,B=None):
        self._A = A
        self._B = B

        try:
            self._N = A.shape[0]**2
            self._D = A.shape[0]
        except AttributeError:
            self._D = B.shape[0]
            self._N = B.shape[0]**2
        
    @property
    def dtype(self):
        return np.result_type(self._A.dtype,self._B.dtype)
        
    @property
    def shape(self):
        return (self._N,self._N)
  
    def _matvec(self,other):
        other = other.reshape((self._D,self._D))
        
        if self._B is None:
            Temp = other
        else:
            Temp = np.dot(other,self._B.T)
            
        if self._A is None:
            pass
        else:
            Temp = np.dot(self._A,Temp).ravel()

        return Temp
    
    def _rmatvec(self,other):
        other = other.reshape((self._D,self._D))
        
        if self._B is None:
            Temp = other
        else:
            Temp = np.dot(other,self._B.conj())
            
        if self._A is None:
            pass
        else:
            Temp = np.matmul(self._A.T.conj(),Temp).ravel()
        
        return Temp



class Projector(LinearOperator):
    # operator |l)(r|
    
    def __init__(self,l,r):
        self._l = l
        self._r = r

        self._l_diag = l.ndim <= 1
        self._r_diag = r.ndim <= 1

        self._N = l.shape[0]**2
        self._D = l.shape[0]

        
    @property
    def dtype(self):
        return np.result_type(self._l.dtype,self._r.dtype)
        
    @property
    def shape(self):
        return (self._N,self._N)
  
    def _matvec(self,other):
        # implement |l)(r|other)
        other = other.reshape((self._D,self._D))

        if self._r_diag:
            r_other = np.sum(self._r.conj() * np.diagonal(other))
        else:
            r_other = np.sum(self._r.conj() * other)

        if self._l_diag:
            return np.diag(self._l * r_other).ravel()
        else:
            return (self._l * r_other).ravel()
        
    
    def _rmatvec(self,other):
        # implement (other|l)(r|
        other = other.reshape((self._D,self._D))

        if self._l_diag:
            l_other = np.sum(self._l * np.diagonal(other).conj())
        else:
            l_other = np.sum(self._l * other.conj())

        if self._r_diag:
            return np.diag(self._r.conj() * l_other).ravel()
        else:
            return (self._r.conj() * l_other).ravel()