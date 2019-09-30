import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns





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
        dataframe.hist(col,bins=25)
        plt.show()
        pdf, bins = np.histogram(data[col], bins=25, density=True)
        bin_centers = (bins[1:] + bins[:-1]) * 0.5
        plt.plot(bin_centers, pdf)
        plt.show()
        plt.hist(dataframe[col], cumulative=True, density=1, label='CDF',
            histtype='step', alpha=1, color='k')
        plt.show()

def mean_var(df,columns):
    res = []
    for col in columns:
        n = np.array(df[col])
        arr = [n.mean(),n.var()]
        res.append(arr)
    return res


# 1 task
data = pd.read_csv('cell2celltrain.csv')
# 2,3 tasks
data.dropna(inplace=True)
columns = list(data)
total = len(data)
number_data = data.select_dtypes(exclude=['object'])
string_data = data.select_dtypes(exclude=['int','float'])
# data frame for the probabilty of string columns
# task 4
print(prob(data,"CreditRating"))
# task 5
print(joint_prob(data,"Churn","ChildrenInHH"))
# task 6
y = data.groupby("Occupation").size()
cond = data.groupby(["PrizmCode","Occupation"]).size()/y
print(cond)
sns.set()
# task 7,8,9
histo(number_data,list(number_data))
# task 10
join_pdf = plt.hist2d(data["MonthlyRevenue"],data["MonthlyMinutes"],bins=50,range=[[0,150],[0,1500]])
plt.show()
# task 11
mv = mean_var(number_data,list(number_data))