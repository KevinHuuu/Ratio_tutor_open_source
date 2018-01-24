# coding: utf-8
import numpy as np
import pandas as pd
import csv
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Merge
from theano import tensor as T
from keras.preprocessing import sequence
import pdb
import os
from tutor_model import *
import argparse
from DataProcessing import *
from chop import *
import datetime

import os
os.environ['KERAS_BACKEND'] = 'theano'

from numpy.random import seed
seed(1)
now = datetime.datetime.now()
now_time = now.strftime('%Y-%m-%d %H:%M:%S')
from utils import create_CV_list

def str2bool(v):
  #susendberg's function
    return v in ("yes", "True", "t", "1")

parser = argparse.ArgumentParser()
parser.register('type', 'bool', str2bool)
parser.add_argument('--train_mode', type=str, default= 'per_student')
parser.add_argument('--with_RedGreen',type='bool', default= False)
parser.add_argument('--with_prompt',type='bool', default= False)
parser.add_argument('--file_name', type=str, default= ('output_file'))
parser.add_argument('--output_file_path', type=str, default= './')
parser.add_argument('--batch_size', type= int, default= 32)
parser.add_argument('--hidden_layer_size', type= int, default= 256)
#parser.add_argument('--learning_rate', type= float, default= 0.001)
parser.add_argument('--optimizer_mode', type= str, default= 'RMSprop')
parser.add_argument('--RNN_mode', type=str, default= 'SimpleRNN')
parser.add_argument('--remove',type='bool', default= True)
parser.add_argument('--with_id',type='bool', default= False)
parser.add_argument('--with_touch',type='bool', default= False)
parser.add_argument('--fixed_len',type=int, default= -1)
parser.add_argument('--sample_rate', type=int, default = 1)

parser.add_argument('--target', type=int)
parser.add_argument('--cut_mode', type=str)


args = parser.parse_args()
file_name = args.file_name + now_time + 'target'+str(args.target) + args.cut_mode +'.csv'
output_file_path = args.output_file_path
batch_size = args.batch_size
hidden_layer_size = args.hidden_layer_size
#learning_rate = args.learning_rate
optimizer_mode = args.optimizer_mode
RNN_mode = args.RNN_mode
epoch = 50
remove = args.remove
with_id = args.with_id
with_touch = args.with_touch
fixed_len = args.fixed_len
with_prompt = args.with_prompt
with_RedGreen = args.with_RedGreen
sample_rate = args.sample_rate

'''per chop'''
#X_matrix, Y_matrix = train_matrix_chop(datasets)
class train_RNN():
    def __init__(self, file_name, cut_mode, output_file_path, batch_size, hidden_layer_size, optimizer_mode,\
                RNN_mode, epoch, datasets, with_prompt, with_RedGreen, with_id, with_touch, fixed_len):
        self.file_name = file_name
        self.output_file_path = output_file_path
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.optimizer_mode = optimizer_mode
        self.RNN_mode = RNN_mode
        self.epoch = epoch
        self.datasets = datasets
        self.with_prompt = with_prompt
        self.with_RedGreen = with_RedGreen
        self.with_id = with_id
        self.with_touch = with_touch
        self.fixed_len = fixed_len
        self.cut_mode = cut_mode

    def avg_matrix(self, info_matrix):
        avg_mat = sum(info_matrix)
        correct = list(np.diag(avg_mat))[0:-1]
        s = list(avg_mat['sum'])[0:-1]
        accu = []
        for i, correct_num in enumerate(correct):
            accu.append(correct_num/s[i])
        accu.append(sum(correct)/sum(s))
        avg_mat.loc['accu'] = accu
        return avg_mat


    def RNN_per_chop(self, CV_list):
        print('RNN per chop')
        datasets = NA_to_0_instructions(self.datasets,with_prompt = self.with_prompt, with_RedGreen = self.with_RedGreen)
        X_matrix, Y_matrix = train_matrix_chop(datasets, self.cut_mode,with_prompt = self.with_prompt, with_RedGreen = self.with_RedGreen,\
                                              with_id = self.with_id, with_touch = self.with_touch, fixed_len = self.fixed_len)

        X_matrix = list(X_matrix)
        Y_matrix = list(Y_matrix)

        accuracy = []
        info_matrix = []
        for CV_fold in CV_list:
            train_index,val_index,test_index = CV_fold
            train_input = []
            val_input = []
            test_input = []

            train_label = []
            val_label = []
            test_label = []

            for i in train_index:
                train_input.append(X_matrix[i])
                train_label.append(Y_matrix[i])
            for i in val_index:
                val_input.append(X_matrix[i])
                val_label.append(Y_matrix[i])
            for i in test_index:
                test_input.append(X_matrix[i])
                test_label.append(Y_matrix[i])

            train_input = np.array(train_input)
            train_label = np.array(train_label)

            val_input = np.array(val_input)
            val_label = np.array(val_label)

            test_input = np.array(test_input)
            test_label = np.array(test_label)

            '''
            print('Test return_sequence ==False! Reduce label dim, -1')
            train_label = train_label[:,-1,:]
            val_label = val_label[:,-1,:]
            test_label = test_label[:,-1,:]
            '''

            # print('Test return_sequence ==False! Reduce label dim, 0')
            # train_label = train_label[:,0,:]
            # val_label = val_label[:,0,:]
            # test_label = test_label[:,0,:]


            input_dim = train_input.shape[-1]
            output_dim = train_label.shape[-1]
            net = tutor_net(args.cut_mode, self.batch_size, self.epoch, self.hidden_layer_size, \
                              input_dim, output_dim,\
                               self.optimizer_mode, self.RNN_mode)
            model = net.build(train_input,  train_label,  val_input,  val_label, self.file_name)
            accu = net.predict(model, test_input, test_label)
            info_mat = net.predict_label(model, test_input, test_label)
            accuracy.append(accu)
            info_matrix.append(info_mat)
        info_matrix.append(self.avg_matrix(info_matrix))
        print ("\nResult:")
        print (accuracy)
        print (info_matrix)
        for df in info_matrix:
            df.to_csv(self.output_file_path+file_name, mode='a')
        print ("\n", file_name, "output finished!\n")

        '''
        print('Saving for plotting')
        np.save('pred_per_chop.npy', net.predict_distribution(model, test_input))
        np.save('true_per_chop.npy', test_label)
        '''
    def RNN_per_chop_student(self):
        print('RNN per chop student')
        datasets = NA_to_0_instructions(self.datasets,with_prompt = self.with_prompt, with_RedGreen = self.with_RedGreen)
        X_set, Y_set = train_matrix_chop_student(datasets,self.cut_mode, with_prompt = self.with_prompt, with_RedGreen = self.with_RedGreen,with_id = self.with_id, with_touch = self.with_touch, fixed_len = self.fixed_len)

        accuracy = []
        info_matrix = []
        for i in range(len(X_set)):
            train_input = []
            val_input = []
            train_label = []
            val_label = []
            for j in range(len(X_set)):
                if i == j:
                    val_input = X_set[j]
                    val_label = Y_set[j]
                else:
                    if train_input ==[]:
                        train_input = X_set[j]
                        train_label = Y_set[j]
                    else:
                        train_input = np.concatenate((train_input, X_set[j]), axis=0)
                        train_label = np.concatenate((train_label, Y_set[j]), axis=0)
            train_input = np.array(train_input)
            train_label = np.array(train_label)

            val_input = np.array(val_input)
            val_label = np.array(val_label)        
                    
                    
            #sub_accuracy = []
            # accuracy = []
            # info_matrix = []

            # print('Test return_sequence ==False! Reduce label dim')
            # train_label = train_label[:,0,:]
            # val_label = val_label[:,0,:]



            input_dim = train_input.shape[-1]
            output_dim = train_label.shape[-1]
            net = tutor_net(args.cut_mode,self.batch_size, self.epoch, self.hidden_layer_size, \
                              input_dim, output_dim,\
                               self.optimizer_mode, self.RNN_mode)
            if self.cut_mode =='no_cut':
                model = net.build(train_input,  train_label, np.zeros(train_input.shape),np.zeros(train_label.shape), self.file_name)
            else:
                model = net.build(train_input[:int(0.8*len(train_input))], train_label[:int(0.8*len(train_label))],\
                                 train_input[int(0.8*len(train_input)):], train_label[int(0.8*len(train_label)):],self.file_name)

            accu = net.predict(model, val_input, val_label)
            accuracy.append(accu)
            print ("\nValidation confusion matrix:")
            info_mat = net.predict_label(model, val_input, val_label)
            info_matrix.append(info_mat)
            print('Saving for plotting')
            np.save('high_predict.npy', net.predict_distribution(model, val_input))
            np.save('high_label.npy', val_label)
            
            
        info_matrix.append(self.avg_matrix(info_matrix))
        print ("\nResult:")
        print (accuracy)
        print (info_matrix)
        for df in info_matrix:
            df.to_csv(self.output_file_path+file_name, mode='a')
        print ("\n", file_name, "output finished!\n")

        print('Saving for plotting')
        np.save('pred_per_chop_student.npy', net.predict_distribution(model, val_input))
        np.save('true_per_chop_student.npy', val_label)

if __name__ == "__main__":
    datasets = load_data('./datasets/')
    datasets = rm_negative(datasets)
    datasets = majority_label(datasets)

    if args.target == 1:
        datasets = target_labelset(datasets)
        if args.cut_mode == 'cut_label':
            CV_list = np.load('./2CV_strategy_label.npy')
        elif args.cut_mode == 'cut_prompt':
            CV_list = np.load('./2CV_strategy_prompt.npy')
        elif args.cut_mode =='no_cut':
            pass
        else: 
            print('Wrong cut_mode')
        print('Target Labels')

    elif args.target == 0:
        if args.cut_mode == 'cut_label':
            CV_list = np.load('./2CV_expanded_label.npy')
        elif args.cut_mode == 'cut_prompt':
            CV_list = np.load('./2CV_expanded_prompt.npy')
        elif args.cut_mode =='no_cut':
            pass
        else: 
            print('Wrong cut_mode')
        print('Expanded Labels')
    else:
        print('Wrong input for --target', args.target)    


    print('fixed_len == ',fixed_len, 'Using random 5-fold CV_list')
    NA_to_0_datasets = NA_to_0_instructions(datasets,with_prompt = with_prompt, with_RedGreen = with_RedGreen)
    #NA_to_0_datasets = del_x(NA_to_0_datasets)
    X_matrix, Y_matrix = train_matrix_chop(NA_to_0_datasets,args.cut_mode,with_prompt = with_prompt, with_RedGreen = with_RedGreen,\
                                           with_id = with_id, with_touch = with_touch, fixed_len = fixed_len)
    #CV_list = create_CV_list(X_matrix, Y_matrix, kf_num = 5, test = True, val_split = 0.1)
    
    output_file_path = './'

    training = train_RNN(file_name,args.cut_mode, output_file_path, batch_size, hidden_layer_size, optimizer_mode,\
                RNN_mode, epoch, datasets, with_prompt, with_RedGreen, with_id, with_touch, fixed_len)
    if args.train_mode == 'per_student':
        training.RNN_per_student()
    elif args.train_mode == 'per_chop':
        training.RNN_per_chop(CV_list)
    elif args.train_mode == 'per_chop_student':
        training.RNN_per_chop_student()
    elif args.train_mode == 'per_student_no_val':
        training.RNN_per_student_no_val()

