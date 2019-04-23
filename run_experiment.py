# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:54:15 2019

@author: Nick
"""

##############################################################################
#  Running the experiment
#   Exit models are not tuneable
#   Number of Validation samples is not an input
#   Number of epochs is not a input
#   These and other parameters can be specified in the ciphar_sim_func.py file
#   ciphar_sim_func.py is called directly in this script (from directory of this file)
##############################################################################

import numpy as np
import pandas as pd
from ciphar_sim_func_0414 import exit_sim

# Specify training sample sizes
start = 100
stop = 5050
step = 100
samples = np.arange(start,stop,step)


iters = samples.shape[0] 

results = pd.DataFrame({'samplesize': samples,
                        'NN_acc': np.zeros([iters]),
                        'RF_acc': np.zeros([iters]),
                        'SVM_acc': np.zeros([iters]),
                        'nntime': np.zeros([iters]),
                        'tottime': np.zeros([iters])})

# Run the experiment
row = 0
for i in range(0,samples.shape[0]):
        
        out = exit_sim(n = samples[i])
        
        results.at[row,'NN_acc'] = out[0]
        results.at[row,'RF_acc'] = out[1]
        results.at[row,'SVM_acc'] = out[2]
        results.at[row,'nntime'] = out[3]
        results.at[row,'tottime'] = out[4]
        row +=1
        print("Progress {:2.1%}".format(row / iters))  
        
        
results.to_csv('results_5000_5.csv',index=False)