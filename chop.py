'''
This script chops sequence per prompt or per label.
'''
from DataProcessing import *
import numpy as np
import pandas as pd
import keras
from keras.utils import to_categorical
import pdb
import csv
from numpy.random import seed
from DataProcessing import *
seed(1)
def cut(datasets):
    print ("Cutting into sequences by label...")
    seq = []
    for data_index, data in enumerate(datasets):
        labels = list(data['label'])
        index = []
        for i in range(data.shape[0]):
            if i < data.shape[0] - 1 and labels[i] != labels[i+1]:
                index.append(i + 1)
        if index[-1] != data.shape[0]:
            index.append(data.shape[0])
        for i in range(len(index) - 1):
            item = data.iloc[index[i]:index[i + 1]]
            seq.append(item)
    print ("Finished.")
    print ("Total number of sequences is "+str(len(seq)))
    return seq


def cut_prompt(datasets):
    print ("Cutting into sequences by prompt...")
    seq = []
    for data_index, data in enumerate(datasets):
        prompt = list(data['instruction'].str.strip())
        index = []
        if prompt[0] != "'NA'":
            index.append(0)
        pr_len = len(prompt)
        for i, item in enumerate(prompt):
            if i < pr_len - 1 and prompt[i + 1] != item and prompt[i + 1] != "'NA'":
                index.append(i + 1)
        index.append(pr_len)
        for i in range(len(index) - 1):
            item = data.iloc[index[i]:index[i + 1]].copy()
            item['instruction'] = item.iloc[0]['instruction']
            seq.append(item)
    print ("Finished.")
    print ("Total number of sequences is "+str(len(seq)))
    return seq


def fixed_cut(datasets, length):
    print ("Cutting into sequences by fixed length", str(length), "...")
    seq = []
    for data_index, data in enumerate(datasets):
        index = [i*length for i in range(int(len(data)/length)+1)]
        if index[-1] != len(data):
            index.append(len(data))
        for i in range(len(index) - 1):
            item = data.iloc[index[i]:index[i + 1]]
            seq.append(item)
    print ("Finished.")
    print ("Total number of sequences is "+str(len(seq)))
    return seq


def cut_NA(datasets):
    print ("Cutting into sequences by NA...")
    seq = []
    for data_index, data in enumerate(datasets):
        ly = list(data['ly'])
        ry = list(data['ry'])
        index = [0]
        for i in range(data.shape[0]):
            if i < data.shape[0] - 1 and (ly[i + 1] != ly[i] or ry[i + 1] != ry[i])\
                    and ((ly[i + 1] == 0 and ry[i + 1] == 0) or (ly[i] == 0 and ry[i] == 0)):
                index.append(i + 1)
        if index[-1] != data.shape[0]:
            index.append(data.shape[0])
        for i in range(len(index) - 1):
            item = data.iloc[index[i]:index[i + 1]]
            seq.append(item)
        for i, item in enumerate(seq):
            seq[i] = item[(item['ly'] != 0) | (item['ry'] != 0)]
            if seq[i].empty:
                del(seq[i])
    print ("Finished.")
    print ("Total number of sequences is "+str(len(seq)))
    return seq


def cut_label(datasets):
    print ("Cutting into sequences by label...")
    seq = []
    for data_index, data in enumerate(datasets):
        labels = list(data['label'])
        index = [0]
        for i in range(data.shape[0]):
            if i < data.shape[0] - 1 and labels[i] != labels[i+1]:
                index.append(i + 1)
        if index[-1] != data.shape[0]:
            index.append(data.shape[0])
        for i in range(len(index) - 1):
            item = data.iloc[index[i]:index[i + 1]]
            seq.append(item)
    print ("Finished.")
    print ("Total number of sequences is "+str(len(seq)))
    return seq


def cut_IT_RT(datasets):
    print ("Test: Cutting into sequences by IT RT...")
    seq = []
    for data_index, data in enumerate(datasets):
        prompt = list(data['instruction'].str.strip())
        index = []
        pr_len = len(prompt)
        for i, item in enumerate(prompt):
            if i < pr_len - 1 and prompt[i + 1] != item and item == "'NA'":
                index.append(i + 1)
        index.append(pr_len)
        for i in range(len(index) - 1):
            item = data.iloc[index[i]:index[i + 1]]
            seq.append(item)
    print ("Finished.")
    print ("Total number of sequences is "+str(len(seq)))
    return seq


def padding(seq, with_id, with_touch, with_prompt, with_RedGreen, max_len=0):
    print ("Padding...")
    # feature_len = 4
    # if with_id:
    #     feature_len += 5
    # if with_touch:
    #     feature_len += 2
    # if with_prompt:
    #     print('with prompt')
    #     dict_instruction = np.load("dict_instruction.npy").item()
    #     prompt_len = len(dict_instruction)
    #     feature_len += prompt_len
    # if with_RedGreen:
    #     print('with RedGreen')
    #     feature_len += 1

    feature_len = seq[0].shape[1] - 2
    print ("Length of feature is", feature_len)
    #print (seq[0])
    lens = [item.shape[0] for _, item in enumerate(seq)]
    if max_len == 0:
        max_len = max(lens)
    print ("Max sequence length is "+str(max_len))

    train_x = []
    for i, item in enumerate(seq):
        if lens[i] < max_len:
            train_x.append(np.vstack((np.array(item.iloc[:, 0:feature_len]), np.array([[-1]*feature_len]*(max_len-lens[i])))))
        else:
            train_x.append(np.array(item.iloc[:, 0:feature_len]))
    train_x = np.array(train_x)
    
    # total_seq = pd.concat(seq)
    # count = 0
    # label_to_integer = {}
    # '''Generate label_to_integer'''
    # for i in set(total_seq['label']):
    #     label_to_integer.update({i: count})
    #     count += 1
    # with open('label_to_integer.csv', 'w') as f:  # Just use 'w' mode in 3.x
    #     w = csv.DictWriter(f, label_to_integer.keys())
    #     w.writeheader()
    #     w.writerow(label_to_integer)
    # print(label_to_integer)
    
        
    label_to_integer = pd.read_csv("label_to_integer.csv", sep=",")
    label_to_integer.columns = label_to_integer.columns.str.strip()
    label_to_integer = label_to_integer.iloc[0].to_dict()

    one_hots_seq = []
    for i, item in enumerate(seq):
        labels = list(item['label'])
        one_hots = []
        for _, label in enumerate(labels):
            if label is 'negative':
                one_hot = [-1]*len(label_to_integer)
            else:
                one_hot = list(to_categorical(label_to_integer[label], num_classes=len(label_to_integer)))
            one_hots.append(one_hot)
        one_hots = np.array(one_hots)
        one_hots_seq.append(one_hots)

    train_y = []
    for i, item in enumerate(one_hots_seq):
        if lens[i] < max_len:
            train_y.append(np.vstack((item, np.array([[-1] * len(label_to_integer)] * (max_len - lens[i])))))
        else:
            train_y.append(item)
    train_y = np.array(train_y)
    print ("Finished.")
    return train_x, train_y

def find_max_len(data):
    max_len = 0
    for i in data:
        max_len = max(max_len,len(i))
    print('max_len is ',max_len)
    return max_len
    
def train_matrix_chop(datasets,cut_mode, with_id = False, with_touch = False, with_prompt = False, with_RedGreen = False, fixed_len = -1):
    if fixed_len == -1:
        if cut_mode == 'cut_label':
            seqs = cut_label(datasets)
        elif cut_mode =='cut_prompt': 
            seqs = cut_prompt(datasets)
        elif cut_mode == 'no_cut':
            seqs = datasets
    else:
        seqs = fixed_cut(datasets, length=fixed_len)
    train_x, train_y = padding(seqs, with_id, with_touch, with_prompt, with_RedGreen)
    return train_x, train_y

def train_matrix_chop_student(datasets,cut_mode, with_id = False, with_touch = False, with_prompt = False, with_RedGreen = False, fixed_len = -1):
    if fixed_len == -1:
        if cut_mode == 'cut_label':
            seqs = cut_label(datasets)
        elif cut_mode =='cut_prompt': 
            seqs = cut_prompt(datasets)
        elif cut_mode == 'no_cut':
            seqs = datasets   
        max_len = find_max_len(seqs)
    else:
        max_len = fixed_len
        
    X_set = []
    Y_set = []
    for stu in datasets:
        if fixed_len == -1:
            if cut_mode == 'cut_label':
                seq = cut_label([stu])
            elif cut_mode =='cut_prompt': 
                seq = cut_prompt([stu])
            elif cut_mode == 'no_cut':
                seq = [stu]
        else:
            seq = fixed_cut([stu], length=fixed_len)
        train_x, train_y = padding(seq, with_id, with_touch, with_prompt, with_RedGreen, max_len)
        X_set.append(train_x)
        Y_set.append(train_y)
    return np.array(X_set), np.array(Y_set)


if __name__ == "__main__":
    datasets = load_data('/research/kevin/ratio_tutor/datasets/')
    datasets = majority_label(datasets)
    datasets = remove_NT(datasets)
    datasets = NA_to_0_instructions(datasets)
    X_set, Y_set = train_matrix_chop_student(datasets)
