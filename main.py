'''
Main script to run 

Created 23 March 2017

Author: Athira
'''

import numpy as np
import argparse
import time

from functions import *

#Parse input arguments
FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="mnist",help='Dataset to use: mnist or cifar-100')
parser.add_argument('--visualize', type=int, default=0,help='Visualize? If yes, input number of images to show')
FLAGS, unparsed = parser.parse_known_args()


#Load and split data
data_loader = DataLoader(FLAGS.dataset)
train,valid,test = data_loader.load_data()

#Visualize?
if FLAGS.visualize is not 0: 
	data_loader.visualize(train,n = FLAGS.visualize)

nSamp_train = train[0].shape[0]  #Number of training samples
nSamp_valid = valid[0].shape[0]  #Number of validation samples
d = valid[0].shape[1]   #Number of features

# Search for optimal sigma through grid search
# sigmas = np.array([0.05,0.08,0.1,0.2,0.5,1.0,1.5,2])
sigmas = np.array([0.05])
#Store results in a dictionry
results = dict()

for sigma_ in sigmas:
	
	print('Calculating for sigma = ' + str(sigma_) + '...')
	#Find log likelihood
	start = time.time()
	ll = log_likelihood(valid[0][0:500],train[0],sigma_)
	end = time.time()
	timing = end - start
	results[sigma_] = {'log-likelihood': np.mean(ll),'timing': timing}

print(results)
