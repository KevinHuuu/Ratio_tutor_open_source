#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data precessing process in this script
"""

print('Using this file to preprocess data')
import pandas as pd
from keras.utils import to_categorical
import numpy as np
import copy
from numpy.random import seed
seed(1)
def del_x(datasets):
    #This function is after NA_to_0_instructions
    new_datasets = []
    for dataset in datasets:
        tmp_dataset = dataset.drop(['lx'],axis = 1)
        tmp_dataset = tmp_dataset.drop(['rx'],axis = 1)
        new_datasets.append(tmp_dataset)
    print('delete both x position information')
    return new_datasets

def sampling(datasets,sample_rate):
    #subsampling from original datasets
    new_datasets = []
    for dataset in datasets:
        index = dataset.index
        new_index = np.arange(min(index),max(index),sample_rate)
        new_datasets.append(dataset.iloc[new_index-new_index[0],:].copy())
    print('Using new datasets with sample rate 1/',sample_rate)
    return new_datasets

def load_data(root):
    '''Load the whole datasets, no modification'''
    print('start to load data from:', root)

    data_EL = pd.read_csv(root+'pandas_EL.csv', sep=',')
    data_ER = pd.read_csv(root+'pandas_ER.csv', sep=',')
    data_KN = pd.read_csv(root+'pandas_KN.csv', sep=',')
    data_MS = pd.read_csv(root+'pandas_MS.csv', sep=',')
    data_ND = pd.read_csv(root+'pandas_ND.csv', sep=',')

    datasets = [data_EL, data_ER, data_KN, data_MS, data_ND]
    print('\n')
    print('Finish load')
    print('\n')
    print('The datasets is:', type(datasets), 'and the elements type is:', type(datasets[0]))
    return datasets

# Remove negative labels list
import copy
def rm_negative(datasets):
    # Negative labels include: NT, -B, 0
    data = copy.deepcopy(datasets)
    for i in range(len(datasets)):
        data[i] = data[i][data[i].label.str.contains('NT') == False]
        data[i] = data[i][data[i].label.str.contains('-B') == False]
        data[i] = data[i][data[i].label.str.contains('0') == False]
    return data

# Using strategy labels OR expended labels
def target_labelset(datasets, target = True):
    strat_labels = ['AB', 'SP', 'D']
    if target:
        # Strategy labels are AB, SP, D
        for i in range(len(datasets)):
            datasets[i] = datasets[i].loc[datasets[i]['label'].isin(strat_labels)]
    else:
        return datasets
    
    return datasets

def negative(datasets, remove=False):
    print('Test remove many labels!!!')
    # Replace or remove label rows containing 'NT', '0', 'B', -B
    if not remove:
        print ("Replacing invalid labels with 'negative'...")
    else:
        print ("Removing invalid labels...")
    data = copy.deepcopy(datasets)
    for i in range(len(datasets)):

        #        df.loc[df['sport'].str.contains('ball', case=False), 'sport'] = 'ball sport'
        if not remove:

            # data[i].loc[(data[i]['label'].str.contains('IT', case=False)==False) & \
            # (data[i]['label'].str.contains('RT', case=False)==False)\
            # , 'label'] = 'negative'


            data[i].loc[data[i]['label'].str.contains('NT', case=False), 'label'] = 'negative'
            data[i].loc[data[i]['label'].str.contains('0', case=False), 'label'] = 'negative'
            data[i].loc[data[i]['label'].str.strip().str.strip("'") == 'B', 'label'] = 'negative'

            data[i].loc[data[i]['label'].str.contains('-B', case=False), 'label'] = 'negative'
            data[i].loc[data[i]['label'].str.contains('IT', case=False), 'label'] = 'negative'
            data[i].loc[data[i]['label'].str.contains('RT', case=False), 'label'] = 'negative'
            data[i].loc[data[i]['label'].str.contains('DBH', case=False), 'label'] = 'negative'
            data[i].loc[data[i]['label'].str.contains('T', case=False), 'label'] = 'negative'
            data[i].loc[data[i]['label'].str.contains('D', case=False), 'label'] = 'negative'
            data[i].loc[data[i]['label'].str.contains('NV', case=False), 'label'] = 'negative'
        else:
            data[i] = data[i][data[i].label.str.contains('NT') == False]
            data[i] = data[i][data[i].label.str.contains('0') == False]
            data[i] = data[i][data[i].label.str.strip().str.strip("'") != 'B']
            data[i] = data[i][data[i].label.str.contains('-B') == False]
            data[i] = data[i][data[i].label.str.contains('IT') == False]
            data[i] = data[i][data[i].label.str.contains('RT') == False]
            data[i] = data[i][data[i].label.str.contains('DBH') == False]
            data[i] = data[i][data[i].label.str.contains('T') == False]
            # data[i] = data[i][data[i].label.str.contains('D') == False]
            data[i] = data[i][data[i].label.str.contains('NV') == False]
    data = [item for item in data if not item.empty]
    print ("Finished.")
    return data


def NA_to_0(datasets):
    data_df = []
    for data_index in range(len(datasets)):
        data = datasets[data_index][['left touch', 'right_touch']]
        print('Begin to process ', data_index, '-th data in datasets')
        for i in range(len(data)):
            row_left_touch = str(data.iloc[i]['left touch']).strip()
            row_right_touch = str(data.iloc[i]['right_touch']).strip()
            if row_left_touch == "' (NA NA NA) '" :
                row_left_touch = "'(0 0 0)'"
            if row_right_touch == "' (NA NA NA) '" :
                row_right_touch = "'(0 0 0)'"
            data.iloc[i]['left touch'] = row_left_touch
            data.iloc[i]['right_touch'] = row_right_touch
        def f(x):
            return list(map(float , x.strip()[2:-2].split(' ')))
        
        left_touch_x = list(map(lambda x:x[0], list(data['left touch'].map(f))))
        left_touch_y = list(map(lambda x:x[1], list(data['left touch'].map(f))))
        right_touch_x = list(map(lambda x:x[0], list(data['right_touch'].map(f))))
        right_touch_y = list(map(lambda x:x[1], list(data['right_touch'].map(f))))
        training_data = pd.DataFrame()
        training_data['original line num'] = datasets[data_index]['original line num']
        training_data['lx'] = left_touch_x
        training_data['ly'] = left_touch_y
        training_data['rx'] = right_touch_x
        training_data['ry'] = right_touch_y
        training_data['color'] = datasets[data_index]['color']
        training_data['instruction'] = datasets[data_index]['instruction']
        training_data['time'] = datasets[data_index]['time']
        training_data['label'] = datasets[data_index]['label']
        
        
        data_df.append(training_data)
    print('Finished!')
    print('\n')
    print('tips: The type of data_df is:',type(data_df),'  . Length = ', len(data_df), '   .The type of the first element',type(data_df[0]))
    return data_df
               

def NA_to_0_instructions(datasets, with_id = False, with_touch = False, with_prompt = False, with_RedGreen = False):
    # Replace NAs by zeros values
    print ("Replacing NAs with Zeros...")
    data_df = []

    for data_index in range(len(datasets)):
        data = datasets[data_index][['left touch', 'right_touch', 'color', 'instruction', 'label']].copy()
        # replace all NAs by 0s
        left_NA_index = data["left touch"].str.strip() == "' (NA NA NA) '"
        right_NA_index = data["right_touch"].str.strip() == "' (NA NA NA) '"
        data.loc[left_NA_index, "left touch"] = "'(0 0 0)'"
        data.loc[right_NA_index, "right_touch"] = "'(0 0 0)'"

        def f(x):
            return list(map(float, x.strip()[2:-2].split(' ')))

        left_touch_x = list(map(lambda x: x[0], list(data['left touch'].map(f))))
        left_touch_y = list(map(lambda x: x[1], list(data['left touch'].map(f))))
        right_touch_x = list(map(lambda x: x[0], list(data['right_touch'].map(f))))
        right_touch_y = list(map(lambda x: x[1], list(data['right_touch'].map(f))))
        instructions = data['instruction']
        labels = data['label'].map(lambda x: x.strip())

        training_data = pd.DataFrame()
        training_data['lx'] = left_touch_x
        training_data['ly'] = left_touch_y
        training_data['rx'] = right_touch_x
        training_data['ry'] = right_touch_y

        if with_id:
            #add student id one-hot columns as feature
            one_hot = list(to_categorical(data_index, num_classes=len(datasets))[0])
            stu_id_columns = pd.DataFrame([one_hot]*len(data))
            stu_id_columns.columns = ['stu_id'+str(i) for i, _ in enumerate(datasets)]
            training_data = training_data.join(stu_id_columns)

        if with_touch:
            #add 2 touch judge columns, 1 if NA, 0 if not
            training_data['left_isNA'] = 0
            training_data['right_isNA'] = 0
            training_data.loc[left_NA_index, 'left_isNA'] = 1
            training_data.loc[right_NA_index, 'right_isNA'] = 1

        if with_prompt:
            #add prompt one-hot columns as feature
            dict_instruction = np.load("dict_instruction.npy").item()
            training_data['instruction'] = list(instructions)
            func = lambda x: pd.Series(list(to_categorical(dict_instruction[x.instruction], num_classes=len(dict_instruction))[0]))
            prompt_columns = training_data.apply(func, axis=1)
            prompt_columns.columns = ['prompt'+str(i) for i in range(len(dict_instruction))]
            training_data = training_data.drop('instruction', axis=1).join(prompt_columns)

        if with_RedGreen:
            #add a binary column to judge whether screen color is Green
            threshold = 0.65
            func = lambda x: 0 if float(x.color.strip().split(' ')[1])<threshold else 1
            training_data['isGreen'] = list(pd.Series(data.apply(func, axis=1)))

        training_data['instruction'] = list(instructions)
        training_data['label'] = list(labels)

        print ("Round " + str(data_index + 1) + " finished.")
        data_df.append(training_data)
    print ("Finished.")
    return data_df


def NA_to_0_instructions1(datasets, with_id = False, with_touch = False, with_prompt = False, with_RedGreen = False):
    # Replace NAs by zeros values, diff one hot
    print ("Replacing NAs with Zeros...")
    data_df = []

    for data_index in range(len(datasets)):
        data = datasets[data_index][['left touch', 'right_touch', 'color', 'instruction', 'label']].copy()
        # replace all NAs by 0s
        left_NA_index = data["left touch"].str.strip() == "' (NA NA NA) '"
        right_NA_index = data["right_touch"].str.strip() == "' (NA NA NA) '"
        data.loc[left_NA_index, "left touch"] = "'(0 0 0)'"
        data.loc[right_NA_index, "right_touch"] = "'(0 0 0)'"

        def f(x):
            return list(map(float, x.strip()[2:-2].split(' ')))

        left_touch_x = list(map(lambda x: x[0], list(data['left touch'].map(f))))
        left_touch_y = list(map(lambda x: x[1], list(data['left touch'].map(f))))
        right_touch_x = list(map(lambda x: x[0], list(data['right_touch'].map(f))))
        right_touch_y = list(map(lambda x: x[1], list(data['right_touch'].map(f))))
        instructions = data['instruction']
        labels = data['label'].map(lambda x: x.strip())

        training_data = pd.DataFrame()
        #training_data['lx'] = left_touch_x
        training_data['ly'] = left_touch_y
        #training_data['rx'] = right_touch_x
        training_data['ry'] = right_touch_y
        data_df.append(training_data['ly'].diff().drop(0))
        data_df.append(training_data['ry'].diff().drop(0))

    counts = pd.concat(data_df).value_counts().drop(0).sort_index()
    area = np.trapz(list(counts.index), x=list(counts))
    n = 6
    rate_p = 0
    index_sep = [-30]
    for i in range(len(list(counts.index))):
        rate_c = int(np.trapz(list(counts.index)[0:i+1], x=list(counts)[0:i+1])/(area/n))
        if rate_c > rate_p:
            index_sep.append(list(counts.index)[i])
            rate_p = rate_c
    index_sep.pop()
    index_sep.append(30)
    index_sep = [-30, -10, -5, -2, 0, 2, 5, 10, 30]
    index_sep = [-30, -10, -5, -1, -0.5,  -0.25, 0.15, -0.05, 0, 0.05, 0.15, 0.25, 0.5, 1, 5, 10, 30]
    print (index_sep)

    data_df = []

    for data_index in range(len(datasets)):
        data = datasets[data_index][['left touch', 'right_touch', 'color', 'instruction', 'label']].copy()
        # replace all NAs by 0s
        left_NA_index = data["left touch"].str.strip() == "' (NA NA NA) '"
        right_NA_index = data["right_touch"].str.strip() == "' (NA NA NA) '"
        data.loc[left_NA_index, "left touch"] = "'(0 0 0)'"
        data.loc[right_NA_index, "right_touch"] = "'(0 0 0)'"

        def f(x):
            return list(map(float, x.strip()[2:-2].split(' ')))

        left_touch_x = list(map(lambda x: x[0], list(data['left touch'].map(f))))
        left_touch_y = list(map(lambda x: x[1], list(data['left touch'].map(f))))
        right_touch_x = list(map(lambda x: x[0], list(data['right_touch'].map(f))))
        right_touch_y = list(map(lambda x: x[1], list(data['right_touch'].map(f))))
        instructions = data['instruction']
        labels = data['label'].map(lambda x: x.strip())

        training_data = pd.DataFrame()
        #training_data['lx'] = left_touch_x
        training_data['ly'] = left_touch_y
        #training_data['rx'] = right_touch_x
        training_data['ry'] = right_touch_y


        training_data = training_data.diff()
        training_data.iloc[0] = [0, 0]
        def func(x):
            if x==0:
                num = 0
            else:
                for num, item in enumerate(index_sep):
                    if x<item:
                        break
            return list(to_categorical(num, num_classes=len(index_sep))[0])

        ly_columns = pd.DataFrame(list(training_data['ly'].map(func)))
        ly_columns.columns = ['ly' + str(i) for i, _ in enumerate(index_sep)]
        ry_columns = pd.DataFrame(list(training_data['ry'].map(func)))
        ry_columns.columns = ['ry' + str(i) for i, _ in enumerate(index_sep)]
        training_data = training_data.join(ly_columns).join(ry_columns).drop('ly', axis=1).drop('ry', axis=1)

        if with_id:
            #add student id one-hot columns as feature
            one_hot = list(to_categorical(data_index, num_classes=len(datasets))[0])
            stu_id_columns = pd.DataFrame([one_hot]*len(data))
            stu_id_columns.columns = ['stu_id'+str(i) for i, _ in enumerate(datasets)]
            training_data = training_data.join(stu_id_columns)

        if with_touch:
            #add 2 touch judge columns, 1 if NA, 0 if not
            training_data['left_isNA'] = 0
            training_data['right_isNA'] = 0
            training_data.loc[left_NA_index, 'left_isNA'] = 1
            training_data.loc[right_NA_index, 'right_isNA'] = 1

        if with_prompt:
            #add prompt one-hot columns as feature
            dict_instruction = np.load("dict_instruction.npy").item()
            training_data['instruction'] = list(instructions)
            func = lambda x: pd.Series(list(to_categorical(dict_instruction[x.instruction], num_classes=len(dict_instruction))[0]))
            prompt_columns = training_data.apply(func, axis=1)
            prompt_columns.columns = ['prompt'+str(i) for i in range(len(dict_instruction))]
            training_data = training_data.drop('instruction', axis=1).join(prompt_columns)

        if with_RedGreen:
            #add a binary column to judge whether screen color is Green
            threshold = 0.65
            func = lambda x: 0 if float(x.color.strip().split(' ')[1])<threshold else 1
            training_data['isGreen'] = list(pd.Series(data.apply(func, axis=1)))

        training_data['instruction'] = list(instructions)
        training_data['label'] = list(labels)

        print ("Round " + str(data_index + 1) + " finished.")
        data_df.append(training_data)
    print ("Finished.")
    return data_df


def majority_label(datasets):
    '''transform some labels to their parent label(majority label)'''
    label_set = []
    for data_index in range(len(datasets)):
        data = datasets[data_index]['label'].str.strip().str.strip("'").str.strip('"')
        data.loc[data.str.contains('AB')] = 'AB'
        data.loc[data.str.contains('A:B')] = 'AB'
        data.loc[data.str.contains('D-')] = 'D'
        data.loc[data.str.contains('DBH')] = 'DBH'
        data.loc[data.str.contains('IT-')] = 'IT'
        data.loc[data.str.contains('SP-')] = 'SP'
        data.loc[data.str.contains('O-')] = 'O'
        data.loc[data.str.contains('T-')] = 'T'
        data.loc[data.str.contains('NT-')] = 'NT'
    
        data_1 = pd.DataFrame()
        
        data_1['original line num'] = datasets[data_index]['original line num']
        data_1['left touch'] = datasets[data_index]['left touch']
        data_1['right_touch'] = datasets[data_index]['right_touch']
        data_1['color'] = datasets[data_index]['color']
        data_1['instruction'] = datasets[data_index]['instruction']
        data_1['time'] = datasets[data_index]['time']
        data_1['label'] = data
        
        label_set.append(data_1)
    print('the type of the return data:',type(label_set),'the shape of the first element',len(label_set[0]))
    return label_set


def NA2avg(datasets):
    '''Replace (NA NA NA)s with the average value'''
    print ("start replacing NAs with average...")
    data_df = []

    for data_index in range(len(datasets)):
        data = datasets[data_index][['left touch', 'right_touch', 'label']].copy()
        # replace all NAs by NAN, in order to convert 'NAN' to nan float value
        data.loc[data["left touch"].str.strip() == "' (NA NA NA) '", "left touch"] = "'(NAN NAN NAN)'"
        data.loc[data["right_touch"].str.strip() == "' (NA NA NA) '", "right_touch"] = "'(NAN NAN NAN)'"

        def f(x):
            return list(map(float, x.strip()[2:-2].split(' ')))

        left_touch_x = list(map(lambda x: x[0], list(data['left touch'].map(f))))
        left_touch_y = list(map(lambda x: x[1], list(data['left touch'].map(f))))
        right_touch_x = list(map(lambda x: x[0], list(data['right_touch'].map(f))))
        right_touch_y = list(map(lambda x: x[1], list(data['right_touch'].map(f))))

        labels = list(data['label'].map(lambda x: x.strip()))
        training_data = pd.DataFrame()
        training_data['lx'] = left_touch_x
        training_data['ly'] = left_touch_y
        training_data['rx'] = right_touch_x
        training_data['ry'] = right_touch_y


        # save_data is for saving data to csv
        save_data = datasets[data_index].copy()
        # for each label, calculate the average value
        for label in set(labels):
            selected_label = pd.Series(labels) == label
            mean = training_data[selected_label].mean()
            mean = [round(mean.lx, 1), round(mean.ly, 1), round(mean.rx, 1), round(mean.ry, 1)]
            selected_index_left = pd.isnull(training_data.lx) & pd.isnull(training_data.ly) & selected_label
            selected_index_right = pd.isnull(training_data.rx) & pd.isnull(training_data.ry) & selected_label
            training_data.loc[selected_index_left, ["lx", "ly"]] = [mean[0], mean[1]]
            training_data.loc[selected_index_right, ["rx", "ry"]] = [mean[2], mean[3]]
            #save_data.loc[selected_index_left, "left touch"] = "'(" + str(mean[0]) + " " + str(mean[1]) + " -10.0)'"
            #save_data.loc[selected_index_right, "right_touch"] = "'(" + str(mean[2]) + " " + str(mean[3]) + " -10.0)'"
            #print (label)

        index_not_nan = pd.notnull(training_data.lx) & pd.notnull(training_data.ly) & pd.notnull(
            training_data.rx) & pd.notnull(training_data.ry)
        bin_index = [int(i) for i in index_not_nan]

        training_data['label'] = labels

        training_data = training_data.loc[index_not_nan]
        print (len(bin_index) - sum(bin_index), "rows still containing NA are removed.")
        #save_data = save_data.loc[index_not_nan]
        #save_data.to_csv("pandas_" + str(data_index) + ".csv", index=False)
        print ("Round " + str(data_index + 1) + " finished.")

        data_df.append(training_data)
    return data_df


def bin(datasets):
    bin_columns_set = []
    for data_index in range(len(datasets)):
        data = datasets[data_index][['left touch', 'right_touch']]
        data["left_is_NA"] = 0
        data["right_is_NA"] = 0
        data.loc[data["left touch"].str.strip() == "' (NA NA NA) '", "left_is_NA"] = 1
        data.loc[data["right_touch"].str.strip() == "' (NA NA NA) '", "right_is_NA"] = 1
        bin_columns = data[["left_is_NA", "right_is_NA"]]
        bin_columns_set.append(bin_columns)
    return bin_columns_set

def prompt(datasets):
    '''turning prompt into one_hot features'''
    datasets_ = [np.array(data['instruction']) for data in datasets]
    prompt_matrix_integer = []
    dict_instruction = np.load('dict_instruction.npy').item()
    for j in range(len(datasets_)):
        integer_matrix = np.zeros(shape = datasets_[j].shape)

        for i in range(datasets_[j].shape[0]):
            integer_matrix[i] = int(dict_instruction[datasets_[j][i]])
        prompt_matrix_integer.append(integer_matrix)

    one_hots = []
    for i in range(len(prompt_matrix_integer)):   
        one_hot = to_categorical(prompt_matrix_integer[i], num_classes=len(dict_instruction))
        one_hots.append(one_hot)
   
    for k in range(len(datasets)):
        onehot_prompt = pd.DataFrame(one_hots[k], columns = list(range(len(dict_instruction))))
        datasets[k] = datasets[k].join(onehot_prompt)
    return datasets


def RedGreen(datasets):
    '''Create another feature to represent whether the screen is Green('1' for green, '0' for red)'''
    datasets_color = [data['color'] for data in datasets]
    for i in range(len(datasets_color)):
        data = datasets_color[i]
        bool_ = np.zeros(shape = (len(data), ))
        for j in range(len(data)):
            sample = data.iloc[j]
            sample = float(sample[7:-2].split(' ')[1])
            if sample > 0.65:
                bool_[j] = 1
        datasets[i]['Green'] = bool_
    return datasets
    
