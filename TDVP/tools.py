import numpy as np
import numba as nb

from scipy.sparse.linalg import LinearOperator
import cProfile

__all__ = ["SuperOperator","Projector","TransferMatrix"]


class SuperOperator(LinearOperator):
    # operator Kronecker(A,B)
    
    def __init__(self,A=None,B=None):
        self._A = A
        self._B = B
        self._Ac = A.conj()
        self._Bc = B.conj()

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
            Temp = np.dot(other,self._Bc)
            
        if self._A is None:
            pass
        else:
            Temp = np.dot(self._Ac.T,Temp).ravel()
        
        return Temp

@nb.njit
def _Transfer_matrix_core(other,A):
    d = A.shape[0]

    out = np.dot(A[0],np.dot(other,A[0].T.conj()))
    for s in range(1,d):
        out += np.dot(A[s],np.dot(other,A[s].T.conj()))

    return out

@nb.njit
def _Transfer_matrix_core_hc(other,A):
    d = A.shape[0]

    out = np.dot(A[0].T.conj(),np.dot(other,A[0]))
    for s in range(1,d):
        out += np.dot(A[s].T.conj(),np.dot(other,A[s]))

    return out

class TransferMatrix(LinearOperator):
    # operator sum(np.kron(A[s],A[s].conj()) for s in range(A.shape[0]))
    
    def __init__(self,A):
        if A.ndim != 3:
            raise ValueError("expecting a 3-tensor for A")

        self._D = A.shape[1]
        self._N = self._D**2
        self._A = A

        
    @property
    def dtype(self):
        return np.dtype(self._A.dtype)
        
    @property
    def shape(self):
        return (self._N,self._N)
  
    def _matvec(self,other):
        other = other.reshape(self._A[0].shape)
        other = other.astype(np.result_type(other.dtype,self.dtype))
        return _Transfer_matrix_core(other,self._A).ravel()
    
    def _rmatvec(self,other):
        other = other.reshape(self._A[0].shape)
        other = other.astype(np.result_type(other.dtype,self.dtype))
        return _Transfer_matrix_core_hc(other,self._A).ravel()    


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



if __name__ == '__main__':
    d = 2**4
    D = 10

    A = np.random.normal(0,1,size=(d,D,D))+1j*np.random.normal(0,1,size=(d,D,D))

    E = SuperOperator(A[0],A[0].conj())
    for s in range(1,d,1):
        E += SuperOperator(A[s],A[s].conj())

    E_2 = TransferMatrix(A)

    v = np.random.normal(0,1,size=(D**2,))

    r = E.dot(v)
    r_2 = E_2.dot(v)

    print(r-r_2)

    r = E.T.dot(v)
    r_2 = E_2.T.dot(v)

    print(r-r_2)    