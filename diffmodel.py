#!/usr/bin/env python
# coding: utf-8

# # A 1D diffusion model

# Here we develop one-deminsional model of diffusion.
# Assuming constant diffusivity
# Regular grid
# Step function for intial condition 
# Fixed boundary condtions

# Here is the diffusion equation

# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$

# Here is the discrtetized version of the diffusion equation we will solve with our model

# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$

# This is the FTCS scheme as described by Slingerland and Kump (2011).

# Lets use two libraries, NumPy and Matplotlib 

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt


# Start by setting two model parmaters, the diffusivity and the size of the model domain 

# In[ ]:


D = 100
Lx = 300


# Next, set up the model grid using a NumPy array

# In[ ]:


dx = 0.5
x = np.arange(start=0,stop=Lx, step=dx)
nx = len(x)


# Intial conditions, inital shape of the ice cream cake
# The cake 'C' is a step function with a high value on the left and a low value on thr right, and a step at the center of the domain

# In[ ]:


C = np.zeros_like(x)
C_left = 500
C_right = 0
C[x<=Lx / 2] = C_left
C[x> Lx / 2] = C_right


# Plot the inital profile

# In[ ]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial Profile")


# Set the number of timesteps in the model
# Calculare a stable timestep using a stability criterion

# In[ ]:


nt = 5000
dt = 0.5*dx**2/D


# Loop over the time steps of the model solving the diffusion equation using the FTCS scheme shown above
# Note the use of array operations on the variable 'C'. 
# The boundary condtions remain fixed at each time step

# In[ ]:


for t in range(0, nt):
	C[1:-1] += D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:])
# C[1:-1] ensures first and lat vlaue (boudnary conditions) are nor touched
# += takes the quantity (D * dt / dx ** 2 * (C[:-2] - 2*C[1:-1] + C[2:]) and ads the quntitify evertime to the loop


# In[ ]:


plt.figure()
plt.plot(x, C, "b")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Final Profile")

