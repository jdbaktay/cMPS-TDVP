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


tdvp = Lattice_TDVP(h,d=2,D=10)

Es = []
Ss = []


for i in range(100):
	step = tdvp.imag_time_step(1e-5)
	Es.append(tdvp.En.real)
	Ss.append(np.linalg.norm(step))


plt.plot(Es)
plt.figure()
plt.plot(Ss)
plt.show()










# In[ ]:




