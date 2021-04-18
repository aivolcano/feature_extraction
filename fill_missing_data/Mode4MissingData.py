import pandas as pd

data = pd.read_csv('titanic_train.csv')

data.Embarked.describe() # .mode()
data['Embarked'].fillna('S', inplace=True)

'''人工LabelEnocder，人工形成有序规则'''
#更改分类变量对应的值
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
#同理，更改Embarked对应的值
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
te=data[data['Embarked'].notnull()]#非空的embarked对应的行