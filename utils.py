# coding: utf-8
'''
some utils functions
'''
import csv
import numpy as np
import pandas as pd
import pdb
import os
import pdb
import argparse
from sklearn.model_selection import KFold
import random
from numpy.random import seed
seed(1)
import random
random.seed(10)
def create_CV_list(X_matrix, Y_matrix, kf_num = 5, test = True, val_split = 0.1):
    # X_matrix: X_matrix is the total input data, shape = samples*time_slices*input_dim
    # Y_maxtrix: Y_maxtrix is the total label data, shape = samples*time_slices*output_dim
    # test: Whether we have a test set. default True.
    assert X_matrix.shape[:-1] == Y_matrix.shape[:-1]
    total_sample_num = X_matrix.shape[0]
    print(total_sample_num)
    total_list = list(range(total_sample_num))
    kf = KFold(n_splits= kf_num, random_state=None, shuffle=True)
    total_set = []
    for train_val_index, test_index in kf.split(total_list):
        print ('train_val_set: ',len(train_val_index),'test_set: ', len(test_index))
        random.shuffle(train_val_index) # shuffle before select val_set
        train_index = train_val_index[:int((1-val_split)*len(train_val_index))]
        val_index = train_val_index[int((1-val_split)*len(train_val_index)):]
        print('train_set ',len(train_index),' val_set ',len(val_index), ' test_set ',len(test_index))
        total_set.append([train_index,val_index,test_index])
    total_set = np.array(total_set)
    np.save("./CV_remove.npy",total_set)
    print('CV list created succesfully!')
    return total_set


if __name__ == "__main__":
    from DataProcessing import *
    from chop import *
    remove = True
    
    datasets = load_data('/research/datasets/telemetry_touch_ratio/')
    datasets = negative(datasets, remove)
    datasets = majority_label(datasets)

    datasets = NA_to_0_instructions(datasets)
    X_matrix, Y_matrix = train_matrix_chop(datasets)  
    a = create_CV_list(X_matrix, Y_matrix, kf_num = 5, test = True, val_split = 0.1)
