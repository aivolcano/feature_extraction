
## 总览
* 思路：数据驱动的方式构造一部分特征
* 技术实现：依靠XGBoost + SHAP的特征依赖图、特征交互图、召回单个样本看特征如何发挥作用
的事后解释反推特征是如何发挥作用的
* 框架优势：
  * 找到连续特征中哪个区间范围内对模型贡献为正 => 构建0-1特征
  * 找到哪些特征做组合可以提升模型效果 => 构建交互特征
  * 召回几个样本，查看每个特征如何发挥作用，可综合召回率、精确率查看效果


### 使用部分依赖图构建0-1特征
以年龄为例，图中显示20-32岁的球星能提升身价，因此我们可以构建【是否处于黄金年龄】这一特征。

该新构建的特征还可以跟其他特征组合。比如【是否黄金年龄】和【速度、射门】等等特征做交互，

```python 
import xbgoost as xgb
model = xgb.XGBClassifier().fit(train_X, train_y)

def get_01_features(model, data, all_feature_cols, continous_features):  
  import shap
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(data[all_feature_cols])
  return shap.dependence_plot(continous_features, shap_values, data[all_feature_cols],interaction_index=None, show=True)
  # shap.dependence_plot('age', shap_values, data[cols],interaction_index=None, show=True)

get_01_features(model, train_X, all_features_cols, 'age')

data['is_gold_age'] = data['age'].apply(lambda r: [1 if r.isin([20, 32]) else 0])

```
![image](https://user-images.githubusercontent.com/68730894/115808595-51e68200-a41d-11eb-9098-0ac1ee9ebc80.png)



### 特征交叉使用特征交互图得知哪些特征值得做交互
图中粉色和蓝色突出的特征适合做特征交互，比如：potential_cf 和 potential_cam 两个特征中粉色部分均突出。但是potential_cf 和 st 做特征交互的意义不大，因为粉色部分没有明显突出。
```python 
def get_interact_features(model, data, all_features_cols):
  import shap
  shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data[all_features_cols])
  return shap.summary_plot(shap_interaction_values, data[all_features_cols], max_display=10) # max_display是展示top-k几个特征的效果

get_interact_features(model, train_X, all_features_cols)

# 使用多项式构建交互特征
data['potential_cf_potential_cf'] = data['potential_cf'] ** 2
data['potential_cf_potential_cam'] = data['potential_cf'] * data['potential_cam']

```
![image](https://user-images.githubusercontent.com/68730894/115808176-932a6200-a41c-11eb-801a-ba61d8d3685c.png)


### 召回样本查看特征是如何发挥作用
模型评价时，我们更侧重召回率、精确率、f1、准确率四个指标评价模型，实际上我们召回样本查看每个特征是如何发挥作用，每个特征的shap值是多少，更利于我们理解模型和数据。

shap值除了能解释树模型，还可以解释神经网络，此时，我们可以使用小样本来做可解释性，减少计算成本。神经网络本身是黑箱，shap本身也是黑箱，两者结合起来颇有魔法对抗魔法的意思。

```python 
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data[cols])

import random
# range(low, high)在哪个范围内生产，4为生成个数
sample_index = random.sample(range(10, 100), 1):
sample_explainer = pd.DataFrame({'feature':cols, 'feature_values':data[cols].iloc[sample_index].values, 'shap_value':shap_values[j]})

print(sample_explainer)
```
![image](https://user-images.githubusercontent.com/68730894/115810365-62e4c280-a420-11eb-930e-3e0e1ca7ca27.png)
