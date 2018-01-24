'''
create label dictionary
'''
from DataProcessing import *
import pandas as pd
datasets = load_data('./datasets/')
label_dict = {}
count = 0
maj_datasets = majority_label(datasets)
for stu in maj_datasets:
    for i in list(set(stu.label)):
        if i not in label_dict:
            label_dict.update({i:[count]})
            count += 1
df = pd.DataFrame(label_dict)
df.to_csv('./label_to_integer.csv',index = False)

'''
str_datasets = strategy_label(datasets)
for stu in str_datasets:
    for i in list(set(stu.label)):
        if i not in label_dict:
            label_dict.update({i:[count]})
            count += 1
df = pd.DataFrame(label_dict)
df.to_csv('./label_to_integer.csv',index = False)
'''
