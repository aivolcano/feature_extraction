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