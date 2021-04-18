
import pandas as pd
import numpy as np 

def set_missing_value(data, n_neighbors=5): # n_neighbors为缺失值相近的5个值来确定缺失值
    from sklearn.impute import KNNImputer
    data = pd.DataFrame(KNNImputer(n_neighbors=n_neighbors).fit_transform(data), columns=data.columns)
    return data

data = pd.read_csv('titanic_train.csv')
df = data[['Pclass','SibSp','Fare','Parch','Age']]
set_missing_value(df, n_neighbors=6)

