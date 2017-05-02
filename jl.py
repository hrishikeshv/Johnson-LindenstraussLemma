
# coding: utf-8

# In[3]:

import numpy as np
import math
from itertools import combinations
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


# In[129]:

from scipy import stats
xk = np.array([0, 1])
pk = (0.6, 0.4)
custm = stats.rv_discrete(name='custm', values=(xk, pk))


# In[133]:

def randomSubspaceGuassian(subspaceDimension, ambientDimension):
    return np.random.normal(0, 1, size=(subspaceDimension, ambientDimension))

def randomSubspaceSparse(subspaceDimension, ambientDimension):
    x = np.vstack([np.random.choice([0, 1, -1], size=ambientDimension, p=[2/3, 1/6, 1/6])]*subspaceDimension)
    for i in range(0, subspaceDimension):
        x[i] = np.random.choice([0, 1, -1], size=ambientDimension, p=[2/3, 1/6, 1/6])
    return x
  
def randomSubspaceSparse1(subspaceDimension, ambientDimension):
    x = np.vstack([custm.rvs(size=ambientDimension)]*subspaceDimension)
    for i in range(0, subspaceDimension):
        x[i] = custm.rvs(size=ambientDimension)
    return x
    
def checkTheorem(oldData, newData, epsilon):
    numBadPoints = 0
    count=0
    error = []
    for (x,y), (x2,y2) in zip(combinations(oldData, 2), combinations(newData, 2)):
        oldNorm = np.linalg.norm(x2-y2)**2 
        newNorm = np.linalg.norm(x-y)**2 
        if newNorm == 0 or oldNorm == 0:
            continue
        count+=1
        error.append(oldNorm/ newNorm)         #distortion
        if abs(oldNorm / newNorm- 1) > epsilon:
            numBadPoints += 1

    plt.hist(error, 20)
    plt.show()
    return (1.0*numBadPoints)/count

def jl(data, subspaceDim):
    origDim = len(data[0])
    A = randomSubspaceSparse1(subspaceDim, origDim)
    transformed = (1 / math.sqrt(subspaceDim)) * A.dot(data.T).T
    return transformed

if __name__ == '__main__':
    size = 1000 #data points
    n = 4096 #original dimension
    eps = 0.25
    subspaceDim = int(math.ceil(2.0*math.log(n)/eps**2))  #logn/e^2
    print(subspaceDim)
    data = np.random.randn(size,n) 
    trans = jl(data,subspaceDim)
    print(checkTheorem(data, trans, eps))


# In[130]:

randomSubspaceSparse1(5, 7)


# In[77]:

randomSubspaceGuassian(5, 6)


# In[111]:




# In[112]:

xk


# In[113]:

custm.xk


# In[118]:

R = custm.rvs(size=10)


# In[119]:

R


# In[ ]:



