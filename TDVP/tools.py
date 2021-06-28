import numpy as np
from scipy.sparse.linalg import LinearOperator

__all__ = ["SuperOperator"]


class SuperOperator(LinearOperator):
    # operator a * Kronecker(A,B)
    
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
