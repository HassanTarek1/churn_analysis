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
