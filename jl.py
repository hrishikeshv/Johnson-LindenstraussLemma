import numpy as np
import math
from itertools import combinations
from matplotlib import pyplot as plt
from scipy.linalg import circulant
from scipy import stats
from scipy.linalg import hadamard
import argparse

def randomSubspace(subspaceDimension, ambientDimension, method="gaussian", q=3.0):
    print(method)
    if method == "gaussian":
        return np.random.normal(0, 1, size=(subspaceDimension, ambientDimension))
    elif method == "circulant":
        cmatrix = circulant(np.random.normal(0, 1, ambientDimension))[:subspaceDimension]
        custm = stats.rv_discrete(values=([-1,1],[1.0/2, 1.0/2]))
        dmatrix = np.diag(custm.rvs(size=ambientDimension))
        return cmatrix.dot(dmatrix)
    elif method == "sparse":
        custm = stats.rv_discrete(values=([-1,0,1],[1.0/(2*q),1-(1.0/q),1.0/(2*q)]))
        return math.sqrt(q)*custm.rvs(size=(subspaceDimension, ambientDimension))
    elif method == "hadamard":
        P = np.random.normal(0, 1, size=(subspaceDimension, ambientDimension))
        H = hadamard(ambientDimension)
        custm = stats.rv_discrete(values=([-1,1],[1.0/2, 1.0/2]))
        D = np.diag(custm.rvs(size=ambientDimension))
        return (1/math.sqrt(ambientDimension))*P.dot(H.dot(D))

def checkTheoremSingle(oldData, newData, epsilon):
    numBadPoints = 0
    count=0
    error = []
    for (x,x2) in zip(oldData, newData):
        oldNorm = np.linalg.norm(x2)**2 
        newNorm = np.linalg.norm(x)**2 

        if newNorm == 0 or oldNorm == 0:
            continue

        count+=1
        error.append(oldNorm/ newNorm)
        if abs(oldNorm / newNorm- 1) > epsilon:
            numBadPoints += 1

#    plt.hist(error, 20)
#    plt.show()
    return error
#    return (1.0*numBadPoints)/count

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
#return error
    return (1.0*numBadPoints)/count

def jl(data, subspaceDim, method, q=3.0):
    origDim = len(data[0])
    A = randomSubspace(subspaceDim, origDim, method, q)
    transformed = (1 / math.sqrt(subspaceDim)) * A.dot(data.T).T
    return transformed

def get_hyperparams(size, n, eps, method, datatype):
    if not method == "circulant":
        subspaceDim = int(math.ceil(2.0*math.log(size)/eps**2))
    elif method == "circulant":
        subspaceDim = int(math.ceil(2.0*(math.log(size))/eps**2))
    if datatype == "dense":
        data = np.random.uniform(-1,1,size=(size,n))
    elif datatype == "sparse":
        indlist = np.random.choice(n, 10, replace=False)
        data = np.zeros((size,n))
        for ind in indlist:
            data[:,ind] = np.random.uniform(-1,1,size)
    return [subspaceDim, data]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="gaussian")
    parser.add_argument('--data', type=str, default="dense")
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--eps', type=float, default=0.25)
    args = parser.parse_args()
    size = args.size
    n = args.n
    eps = args.eps
    method = args.method
    datatype = args.data
    if not method == "circulant":
        subspaceDim = int(math.ceil(2.0*math.log(size)/eps**2))
    elif method == "circulant":
        subspaceDim = int(math.ceil(2.0*(math.log(size))/eps**2))
    if datatype == "dense":
        data = np.random.uniform(-1,1,size=(size,n))
    elif datatype == "sparse":
        indlist = np.random.choice(n, 10, replace=False)
        data = np.zeros((size,n))
        for ind in indlist:
            data[:,ind] = np.random.uniform(-1,1,size)
    print(subspaceDim)
    trans = jl(data,subspaceDim,method,3.0)
    print(checkTheorem(data, trans, eps))
