'''
Functions to load,pre-process and visualize data and calculate log likelihood
Author: Athira
'''
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.core.umath_tests import inner1d
from scipy.misc import logsumexp

class DataLoader():
	'''
	Data loader class that loads data, divides it into training, validation and testing sets and returns them
	Also visualize the data
	Param  dataset: name of the dataset (mnist/cifar-10)
	'''
	def __init__(self,dataset):
		if dataset == 'mnist':
			self.dataset_name = '../mnist.pkl'
		elif dataset == 'cifar-100':
			self.dataset_name = 'a'
		else: 
			print('Unrecognized dataset!')

	def unpickle(self): 
		#Unpickle the data file 
		fo = open(self.dataset_name, 'rb')
		dict = pickle.load(fo) #Object pickled in Python 2, unpickling now in python 3
		fo.close()
		return dict

	def load_data(self):
		 train,valid,test = self.unpickle()
		 #Use only 10k from trainset as training set, and another 10k as validation set
		 N = 10000
		 nSamp = train[0].shape[0]
		 indx = np.random.permutation(nSamp) #Shuffle
		 train_new = (train[0][indx[:N]],train[1][indx[:N]]) 
		 valid_new = (train[0][indx[N:2*N]],train[1][indx[N:2*N]])
		 return train_new,valid_new,test

	def visualize(self,data, n = 20):
		#Visualize first n*n images of the data in a nxn grid
		plt.figure(figsize=(n,n))
		gs1 = gridspec.GridSpec(n, n)
		gs1.update(wspace=0.8, hspace=0.0005) # set the spacing between axes. 

		for i in range(n*n):
			img = data[0][i].reshape((28, 28))
			label = data[1][i]
			ax = plt.subplot(gs1[i])
			plt.axis('off')
			plt.title(str(label))
			plt.imshow(img, cmap='gray')
		plt.show()

	def pre_process(self,X):
		X = (X-np.min(X))


def log_likelihood(X,D,sigma):
	'''
	Calculate log likelihood of data samples in X, given a dataset D
	Inputs-	  X : mxd matrix
			  D : kxd data matrix
			  sigma: smoothing parameter
	Outputs-  ll: mx1 array of log likelihoods log(p(x))
	'''
	
	N = D.shape[0]
	K = np.arange(1,N+1).reshape([1,N])
	d = X.shape[1]
	nSamp = X.shape[0]

	ll = np.zeros([nSamp,1])
	k = 0
	batchSize = 1   #Number of samples to be calculated simultaneously 
	while k<nSamp:
		t = min(k+batchSize,nSamp)
		x = X[k:t]
		x = x.reshape([x.shape[0],x.shape[1],1])
		temp = np.einsum('ijk,ijk->ki',np.transpose(x-np.transpose(D)),np.transpose(x-np.transpose(D)))/(2*sigma**2)
		a = np.max(-temp)   #Constant term to take out of exponential for better numerical behaviour
		ll[k:t] = (a + np.log(np.sum(np.exp(-temp - a)/K,1))- np.log(2*np.pi*sigma**2)*d/2).reshape([batchSize,1])
		k = k+batchSize
		if k%100 == 0:
	 		print(''.join([str(k),' samples done..']))


	return ll





		

