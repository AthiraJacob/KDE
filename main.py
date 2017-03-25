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

if FLAGS.visualize is not 0: #Visualize?
	data_loader.visualize(train,n = FLAGS.visualize)

nSamp_train = train[0].shape[0]  #Number of training samples
nSamp_valid = valid[0].shape[0]  #Number of validation samples
d = valid[0].shape[1]   #Number of features

sigma = 1.0

#Find log likelihood
start = time.time()
ll = log_likelihood(valid[0][0:500],train[0],sigma)
end = time.time()

print(end - start)

