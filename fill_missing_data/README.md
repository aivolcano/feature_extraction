

缺失值填充是特征工程的重要部分

深度学习、决策树、众数、中位数、KNN是填充缺失值的4种处理方案

在考虑模型效果和准确率优先时，可以使用深度学习填充缺失值，datawig提供的缺失值预测相当准确，更提供GPU和CPU计算

使用Titanic数据集进行工具开发。

* 相比简单的中位数、众数、均值填充Nan值，使用算法找到相应规律并预测缺失值的策略更为准确。

* 上手快：KNN算法最符合“最近邻法”预测缺失值，利用欧氏距离找到缺失值附近的值并预测缺失值，只需3行代码就可以完成，继承SKlearn中的fit_transform。

* 随机森林的代码量比KNN多，和深度学习预测缺失值的代码量差不多，在考虑模型效果和准确率优先时，可以使用深度学习。并且datawig提供的缺失值预测相当准确，可以选择。

```python 
def set_missing_value(raw_data, input_columns, output_column, num_epochs):
    import datawig
    rd_train, rd_test = datawig.utils.random_split(raw_data)
    # 初始化并拟合一个简单的imputer模型
    imputer = datawig.SimpleImputer(
        input_columns = input_columns,
        output_column = output_column,
        output_path = 'imputer_model').fit(rd_train, num_epochs=num_epochs)#存储模型数据和度量
    imputed_test = imputer.predict(rd_test)
#     print('MSE:{.4lf}', mean_squared_error())
    imputed = imputer.predict(raw_data)
    raw_data.loc[(data[output_column].isnull()), output_column] = imputed.loc[(imputed[output_column].isnull()), str(output_column + '_imputed')].apply(lambda x: float(round(x, 1)))
    return raw_data

import pandas as pd
data = pd.read_csv('titanic_train.csv')
set_missing_value(data, input_columns=['Pclass','SibSp','Fare','Parch'], output_column='Age', num_epochs=100)
# 输入原始数据，input_columns是用哪些列去预测有缺失值的列，output_column是缺失值的列，num_epochs是迭代次数
```