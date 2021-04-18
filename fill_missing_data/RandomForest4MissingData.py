def set_missing_value(data, input_columns, output_column):
    from sklearn.ensemble import RandomForestRegressor
    #把数值型特征都放到随机森林里面去
    input_columns.insert(0, output_column) # 缺失字段与不缺失字段拼接 缺失字段固定放在首位
    input_data = data[input_columns]
    know_output_column = input_data[input_data[output_column].notnull()].values
    unknow_output_column = input_data[input_data[output_column].isnull()].values
    y = know_output_column[:, 0] #y是年龄，第一列数据
    x = know_output_column[:, 1:] #x是特征属性值，后面几列
    rfr = RandomForestRegressor(random_state=0, n_estimators=500, n_jobs=-1).fit(x, y)
    #填补缺失值
    data.loc[(data[output_column].isnull()), output_column] = rfr.predict(unknow_output_column[:, 1:]).astype('int32')
    return data

import numpy as np
import pandas as pd
data = pd.read_csv('titanic_train.csv')
set_missing_value(data, input_columns=['Pclass','SibSp','Fare','Parch'], output_column='Age')