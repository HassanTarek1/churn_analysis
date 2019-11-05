import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import beta


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

def normal_fit(data,col):
    plt.figure()
    plt.hist(data[col], bins=25, density=True, alpha=0.6, color='g')
    mu, std = norm.fit(data[col])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'y', linewidth=2)
    title = "Normal fitting of "+str(col)+" with Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.show()


def exponential_fit(data,col):
    plt.figure()
    plt.hist(data[col], bins=25, density=True, alpha=0.6, color='g')
    loc, scale = expon.fit(data[col])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = expon.pdf(x, loc, scale)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Exponential fitting of "+str(col)
    plt.title(title)
    plt.show()

def beta_fit(data,col):
    plt.figure()
    plt.hist(data[col], bins=25, density=True, alpha=0.6, color='g')
    a,b,loc,scale = beta.fit(data[col])
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p=beta.pdf(x,a,b,loc,scale)
    plt.title("Beta fitting of "+str(col))
    plt.plot(x, p, 'r', linewidth=2)
    plt.show()

def pdf(data,col):
    plt.figure()
    pdf, bins = np.histogram(data[col], bins=25, density=True)
    bin_centers = (bins[1:] + bins[:-1]) * 0.5
    plt.plot(bin_centers, pdf)
    plt.show()

def bayes(data,col,given1,given2):
    y = data.groupby(col).size()
    cond1 = data.groupby([given1, col]).size() / y
    cond2 = data.groupby([given2, col]).size() / y
    p=prob(data,col)
    p1 =prob(data,given1)
    p2 =prob(data,given2)

def cond(data,col,given):
    y = data.groupby(given).size()
    cond = data.groupby([col, given]).size() / y
    return pd.DataFrame(cond)

# milestone1
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
y = data.groupby("CreditRating").size()
cond = data.groupby(["PrizmCode","CreditRating"]).size()/y
print(cond)
sns.set()
# task 7,8,9
# histo(number_data,list(number_data))
# task 10
# join_pdf = plt.hist2d(data["MonthlyRevenue"],data["MonthlyMinutes"],bins=50,range=[[0,150],[0,1500]])
# plt.show()
# task 11
mv = mean_var(number_data,list(number_data))

# milestone 2 ...
# task 1
data_corr= data.corr()
data_cov = data.cov()

# task 2


# task 3
# normal_fit(data,"ReceivedCalls")
# pdf(data,"ReceivedCalls")
# exponential_fit(data,"ReceivedCalls")
# beta_fit(data,"ReceivedCalls")

# task 4
y2 = data.groupby("UnansweredCalls").size()
cond2 = data.groupby(["DroppedCalls","UnansweredCalls"]).size()/y2
cond2 = pd.DataFrame(cond2)
exponential_fit(cond2, 0)
beta_fit(cond2,0)
y = data.groupby(["DroppedCalls","ReceivedCalls"]).size()
cond1 = data.groupby(["Churn","DroppedCalls","ReceivedCalls" ]).size() / y
p=prob(data,"Churn")
y = data.groupby("Churn").size()
cond3 = data.groupby(["DroppedCalls", "Churn"]).size() / y
cond4=data.groupby(["ReceivedCalls", "Churn"]).size() / y
p1=prob(data,"DroppedCalls")
p2=prob(data,"ReceivedCalls")