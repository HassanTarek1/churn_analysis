import pandas as pd
import numpy as np

data = pd.read_csv('cell2celltrain.csv')
data.dropna(how='any',inplace=True)
columns = list(data)
print(data)
