import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.groupby import GroupBy


def is_string(dataframe, element):
    column = dataframe[element]
    c = np.array(column)
    tmp = c[0]
    return isinstance(tmp, str)


def prob(dataframe, col):
    group = dataframe.groupby(col)
    num_group = group.size()
    return num_group / len(dataframe)

def joint_prob(dataframe,col1,col2):
    p=dataframe.groupby([col1,col2])
    return (p.size())/len(dataframe)

def cond_prob(dataframe,col1,col2):
    join = (joint_prob(dataframe,col1,col2))
    join_arr = np.array(join)
    p1 = np.array(prob(dataframe,col1))
    p2 = np.array(prob(dataframe,col2))
    for i in range(len(p1)):
        for k in range(len(p2)):
            join_arr[(i*len(p2))+k] = join_arr[(i*len(p2))+k]/p2[k]
    cond= pd.DataFrame(join)
    cond.rename(columns = {0:"Joint"},inplace=True)
    cond.insert(1,"Conditional",join_arr)
    return cond




data = pd.read_csv('cell2celltrain.csv')
data.dropna(how='any', inplace=True)
columns = list(data)
total = len(data)
prob_df_yn = pd.DataFrame()
prob_df_str = pd.DataFrame()
keys = []
frames = []
for col in columns:
    if (is_string(data, col) and
            (str(data[col][0]) == "Yes" or str(data[col][0]) == "No")):
        prob_col = (prob(data, col))
        prob_df_yn[col] = prob_col
    elif is_string(data, col):
        keys.append(col)
        tmp = pd.DataFrame(prob(data, col))
        frames.append(tmp)
prob_df_str = pd.concat(frames, keys=keys)
p = (cond_prob(data,columns[1],columns[len(columns)-1]))
x = prob(data,columns[len(columns)-1])