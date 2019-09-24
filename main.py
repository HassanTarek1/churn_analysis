import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def is_string(dataframe, element):
    column = dataframe[element]
    c = np.array(column)
    tmp = c[0]
    return isinstance(tmp, str)


def prob(dataframe, col):
    group = dataframe.groupby(col)
    num_group = group.size()
    return num_group / len(dataframe)


def joint_prob(dataframe, col1, col2):
    p = dataframe.groupby([col1, col2])
    return (p.size()) / len(dataframe)


def cond_prob(dataframe, col1, col2):
    join = (joint_prob(dataframe, col1, col2))
    join_arr = np.array(join)
    p1 = np.array(prob(dataframe, col1))
    p2 = np.array(prob(dataframe, col2))
    for i in range(len(p1)):
        for k in range(len(p2)):
            join_arr[(i * len(p2)) + k] = join_arr[(i * len(p2)) + k] / p2[k]
    cond = pd.DataFrame(join)
    cond.rename(columns={0: "Joint"}, inplace=True)
    cond.insert(1, "Conditional", join_arr)
    return cond

def histo(dataframe,columns):
    sns.set()
    for col in columns:
        if not is_string(dataframe,col):
            dataframe.hist(col,bins=20)
            plt.show()
            dataframe[col].plot.kde()
            plt.show()
            plt.hist(dataframe[col], cumulative=True, density=1, label='CDF',
                     histtype='step', alpha=1, color='r')
            plt.show()

def mean_var(df,columns):
    res = []
    for col in columns:
        if not is_string(df,col):
            n = np.array(df[col])
            arr = [n.mean(),n.var()]
            res.append(arr)
    return res

data = pd.read_csv('cell2celltrain.csv')
data.dropna(how='any', inplace=True)
columns = list(data)
total = len(data)
# data frame for the probability of all yes/no columns
prob_df_yn = pd.DataFrame()
# data frame for the probabilty of the other string columns
prob_df_str = pd.DataFrame()
keys = []
frames = []
for col in columns:
    if (is_string(data, col) and
            (str(data[col][0]) == "Yes" or str(data[col][0]) == "No")):
        prob_col = (prob(data, col))
        prob_df_yn[col] = prob_col
    elif is_string(data, col):
        # keys is an array of columns names
        keys.append(col)
        tmp = pd.DataFrame(prob(data, col))
        # frames is an array of data frames containing probabilty of each column
        frames.append(tmp)

# concatinating the columns in one data frame and can be accessed by prob_df_str.loc[key]
prob_df_str = pd.concat(frames, keys=keys)
# the cond_prob() function returns a data frame containg the joint probability and the conditional
y = data.groupby("PrizmCode").size()
cond1 = data.groupby("PrizmCode")["Occupation"].value_counts()/y
cond2 = cond_prob(data,"Occupation","PrizmCode")["Conditional"]
sns.set()
# histo(data,columns)
x = np.array(data["MonthlyRevenue"])
y = np.array(data["MonthlyMinutes"])
mv = mean_var(data,columns)

