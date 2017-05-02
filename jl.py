import numpy as np
import math
from itertools import combinations
from matplotlib import pyplot as plt
from scipy.linalg import circulant

def randomSubspace(subspaceDimension, ambientDimension, method="gaussian"):
    if method == "gaussian":
        return np.random.normal(0, 1, size=(subspaceDimension, ambientDimension))
    elif method == "circulant":
        cmatrix = circulant(np.random.normal(0, 1, ambientDimension))[:subspaceDimension]
        dmatrix = np.diag(bernoulli.rvs(0.5,ambientDimension))
        return cmatrix.dot(dmatrix)

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
        error.append(oldNorm/ newNorm)
        if abs(oldNorm / newNorm- 1) > epsilon:
            numBadPoints += 1

    plt.hist(error, 20)
    plt.show()
    return (1.0*numBadPoints)/count

def jl(data, subspaceDim):
    origDim = len(data[0])
    A = randomSubspace(subspaceDim, origDim)
    transformed = (1 / math.sqrt(subspaceDim)) * A.dot(data.T).T
    return transformed

if __name__ == '__main__':
    size = 1000
    n = 4096
    eps = 0.05
    method = "circulant"
    datatype = "dense"
    if method == "gaussian":
        subspaceDim = int(math.ceil(2.0*math.log(size)/eps**2))
    elif method == "circulant":
        subspaceDim = int(math.ceil((math.log(size)**3)/eps**2))
    if datatype == "dense":
        data = np.random.uniform(-1,1,size=(size,n))
    elif datatype == "sample":
        indlist = np.random.choice(n, 10, replace=False)
        data = np.zeros((size,n))
        for ind in indlist:
            data[ind] = np.random.uniform(-1,1,n)
    trans = jl(data,subspaceDim)
    print(checkTheorem(data, trans, eps))
