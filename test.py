from jl import *
import numpy as np
from matplotlib import pyplot as plt

def distovereps():

	eps = np.linspace(0.5,0,10,endpoint=False)
	minerr=[1.0]
	maxerr=[1.0]
	meanerr=[1.0]
	for e in eps[::-1]:
		print(e)
		[subspaceDim, data] = get_hyperparams(1000, 4096, e, 'gaussian','dense')
		trans = jl(data,subspaceDim,'gaussian')
		err = checkTheorem(data,trans,e)
		minerr.append(np.min(err))
		maxerr.append(np.max(err))
		meanerr.append(np.mean(err))
	
	eps = np.append(eps,[0.0])

	plt.plot(eps[::-1], minerr)
	plt.plot(eps[::-1], maxerr)
	plt.plot(eps[::-1], meanerr)
	plt.title('Pairwise distortions')
	plt.legend(['min err','max err','mean err'], loc='upper left')
	plt.xlabel('$\epsilon$')
	plt.ylabel('Distortion')
	plt.show()

def stabsparse():

	qlist = np.linspace(3,150,20)
	minerr = []
	maxerr = []
	meanerr = []
	for q in qlist:
		print(q)
		[subspaceDim, data] = get_hyperparams(10000, 4096, 0.25, 'sparse','sparse')
		trans = jl(data,subspaceDim,'sparse',q)
		err = checkTheorem(data,trans,0.25)
		minerr.append(np.min(err))
		maxerr.append(np.max(err))
		meanerr.append(np.mean(err))

	plt.plot(qlist, minerr)
	plt.plot(qlist, maxerr)
	plt.plot(qlist, meanerr)
	plt.title('Dense data')
	plt.legend(['min err','max err','mean err'], loc='upper left')
	plt.xlabel('q')
	plt.ylabel('Distortion')
	plt.show()

if __name__ == "__main__":
	stabsparse()
#distovereps()
