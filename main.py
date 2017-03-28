'''
Main script to run. This script constructs the model and runs the validation dataset through the model using different sigma values to find the best sigma. The best sigma is then used to run the test set. 
Input parameters: dataset to use and number of images to display (if needed)

Sample commands to run the script: 
python main.py                                         (Runs with default settings: MNIST dataset, no visualization )
python main.py --dataset "cifar-100" --visualize 10    (Runs with CIFAR100 dataset, displays a grid of 10 x 10 images)

The dataset to be loaded is assumed to be present in the parent directory with the following paths:  
MNIST:         '../mnist.pkl'
CIFAR100:      '../train' and '../test'

Created 23 March 2017
The code is written on Linux 3.13.0-113-generic , using Python 2.7.6

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
	data_loader.visualize(train,n = FLAGS.visualize) #Visualize training dataset

# Search for optimal sigma through grid search
sigmas = np.array([0.05,0.08,0.1,0.2,0.5,1.0,1.5,2])
results = dict()  #Store results in a dictionry

#Loop through sigma's and store the results in a dictionary
for sigma_ in sigmas:
	
	#Find log likelihood
	start = time.time()
	ll = log_likelihood(valid[0:2],train,sigma_)
	end = time.time()
	timing = end - start
	results[sigma_] = {'log-likelihood': np.mean(ll),'timing': timing}


print(results)

#Find the best sigma by checking log-likelihood of each sigma
best_ll = -np.inf
for key,value in results.items():
	if value['log-likelihood'] > best_ll:
		best_sigma = key

#Calculate log-likelihood of test set using best sigma
print('Calculating log-likelihood of test data with the best sigma..')
start = time.time()
test_ll = log_likelihood(test,train,best_sigma)
end = time.time()
print('Best sigma = ' + str(best_sigma) + ' with log-likelihood = ' + str(np.mean(test_ll)) + ' and timing = ' + str(end - start))
