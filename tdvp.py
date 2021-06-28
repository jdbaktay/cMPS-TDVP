#!/usr/bin/env python
# coding: utf-8

# # Lattice TDVP calculation

# In[1]:


import numpy as np
from TDVP import SuperOperator,Lattice_TDVP
import matplotlib.pyplot as plt

np.random.seed(0)
d = 2
s_x = np.array([[0,1],[1,0]])
s_z = np.array([[0,1],[0,-1]])

h = 0*np.kron(s_z,s_z)+1*(np.kron(np.eye(*s_x.shape),s_x)+np.kron(s_x,np.eye(*s_x.shape)))


tdvp = Lattice_TDVP(h,d=2,D=4)


for i in range(10000):
	step = tdvp.imag_time_step(0.0001)
	print(tdvp.En,np.linalg.norm(step))









# In[ ]:




