'''
Functions to load,pre-process and visualize data and calculate log likelihood
Author: Athira
'''
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class DataLoader():
	'''
	Data loader class that loads data, divides it into training, validation and testing sets and returns them
	Also visualize the data
	Param  dataset : name of the dataset (mnist/cifar-10)
	'''
	def __init__(self,dataset):
		'''
		Store dataset name and path in self. Assumes that the data files are stored in parent directory
		'''
		self.dataset = dataset
		if dataset == 'mnist':
			self.file_path = '../mnist.pkl'
		elif dataset == 'cifar-100':
			self.file_path = ['../train','../test']
		else: 
			print('Unrecognized dataset!')

	def unpickle(self, file): 
		#Unpickle the data file 
		fo = open(file, 'rb')
		dict = pickle.load(fo) 
		fo.close()
		return dict

	def load_data(self):
		#Load and unpickle datasets stored in self
		if self.dataset == 'mnist':
			train,valid,test = self.unpickle(self.file_path)
			train = train[0]
			test = test[0]
		elif self.dataset == 'cifar-100':
			train = self.unpickle(self.file_path[0])['data']
			test = self.unpickle(self.file_path[1])['data']
			train = train/float(np.max(train))
			test = test/float(np.max(test))
		#Use only 10k from trainset as training set, and another 10k as validation set
		N = 10000
		nSamp = train.shape[0]
		indx = np.random.permutation(nSamp) #Shuffle
		return train[indx[:N]],train[indx[N:2*N]],test

	def visualize(self,data, n = 20):
		#Visualize first n*n images of the data in a nxn grid
		plt.figure(figsize=(n,n))
		gs1 = gridspec.GridSpec(n, n)
		gs1.update(wspace=0.008, hspace=0.05) # set the spacing between axes. 

		if self.dataset == 'mnist':
			for i in range(n*n):
				ax = plt.subplot(gs1[i])
				plt.axis('off')
				img = data[i].reshape((28,28))
				plt.imshow(img,'gray')
			plt.show()
		elif self.dataset == 'cifar-100':
			for i in range(n*n):
				ax = plt.subplot(gs1[i])
				plt.axis('off')
				img = data[i].reshape((32,32,3),order = 'F')
				img = np.transpose(img,(1,0,2))
				plt.imshow(img)
			plt.show()



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
	batchSize = 50   #Number of samples to be calculated simultaneously 
	while k<nSamp:
		t = min(k+batchSize,nSamp)
		x = X[k:t]
		x = x.reshape([x.shape[0],x.shape[1],1])
		temp = np.einsum('ijk,ijk->ki',np.transpose(x-np.transpose(D)),np.transpose(x-np.transpose(D)))/(2*sigma**2)
		a = np.max(-temp,1).reshape([t-k,1])   #Constant term to take out of exponential for better numerical behaviour
		ll[k:t] = a + np.log(np.sum(np.exp(-temp - a)/K,1)).reshape([t-k,1])- np.log(2*np.pi*sigma**2)*d/2
		k = k+batchSize
		if k%500 == 0:
	 		print(''.join([str(k),' samples done..']))


	return ll





		

