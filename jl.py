import numpy as np
import math
from itertools import combinations
from matplotlib import pyplot as plt
from scipy.linalg import circulant
from scipy.linalg import hadamard
from scipy import stats

def randomSubspace(subspaceDimension, ambientDimension, method="hadamard"):
    if method == "gaussian":
        return np.random.normal(0, 1, size=(subspaceDimension, ambientDimension))
    elif method == "circulant":
        cmatrix = circulant(np.random.normal(0, 1, ambientDimension))[:subspaceDimension]
        custm = stats.rv_discrete(values=([-1,1],[1.0/2, 1.0/2]))
        dmatrix = np.diag(custm.rvs(size=ambientDimension))
        return cmatrix.dot(dmatrix)
    elif method == "sparse":
        custm = stats.rv_discrete(values=([-1,0,1],[1.0/6,2.0/3,1.0/6]))
        unnorm_matrix = custm.rvs(size=(subspaceDimension, ambientDimension))
        return unnorm_matrix
    elif method == "hadamard":
        P = np.random.normal(0, 1, size=(subspaceDimension, ambientDimension))
        H = hadamard(ambientDimension)
        custm = stats.rv_discrete(values=([-1,1],[1.0/2, 1.0/2]))
        D = np.diag(custm.rvs(size=ambientDimension))
        return P.dot(H.dot(D))
        # unnorm_matrix =  P.dot(H.dot(D))
        # l2norm = np.sqrt((unnorm_matrix * unnorm_matrix).sum(axis=1))
        # ret_matrix = unnorm_matrix / l2norm.reshape(unnorm_matrix.shape[0],1)
        # return ret_matrix

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
    n = 2048
    eps = 0.4
    method = "gaussian"
    datatype = "dense"
    subspaceDim = 0
    if method == "gaussian":
        subspaceDim = int(math.ceil(2.0*math.log(size)/eps**2))
    elif method == "sparse":
        subspaceDim = int(math.ceil(2.0*math.log(size)/eps**2))
    elif method == "circulant":
        subspaceDim = int(math.ceil(2.0*(math.log(size))/eps**2))

    data = np.random.uniform(-1,1,size=(size,n))
    if datatype == "dense":
        data = np.random.uniform(-1,1,size=(size,n))
    elif datatype == "sparse":
        indlist = np.random.choice(n, 10, replace=False)
        data = np.zeros((size,n))
        for ind in indlist:
            data[ind] = np.random.uniform(-1,1,n)

    # l2norm = np.sqrt((data * data).sum(axis=1))
    # data = data / l2norm.reshape(data.shape[0],1)
    
    trans = jl(data,subspaceDim)
    print(subspaceDim)
    print(checkTheorem(data, trans, eps))