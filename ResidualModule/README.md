

LSTM、GRU、Transformer、DNN等结构的残差网络
```python 
'''TensorFlow'''
class ResidualModule(tf.keras.Model):
	def __init__():
		super().__init__()
		self.model = model
	def call(inputs, *args, **kwargs):
    # 只输出一个值
		# delta = self.model(inputs, *args, **kwargs)
    # RNN 输出2个值，所以
    delta = self.model(inputs, *args, **kwargs)[0]
		return inputs + delta

# 放在 def __init__ 部分
self.residual_gru = ResidualModule(tf.keras.Sequential([
                  tf.keras.layers.GRU(32, return_sequences=True),
                  # 使用线性层统一维度
		  tf.keras.layers.Dense(num_features, kernel_initializer=tf.initializers.zeros())
# delta刚开始需要很小，所以使用0来初始化output层
]))

# 放在 call 部分
output = self.residual_gru(inputs)

'''PyTorch'''
class ResidualModule(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = model
	def forward(inputs, *args, *kwargs):
		delta = self.model(inputs, *args, **kwargs)
		return inputs + delta

self.residualmodule = ResidualModule(nn.Sequential(
                nn.TransfomerEncoder(nn.TransformerEncoderLayer(
                        d_model=512, n_head=8, dropout=0.2, activation='gelu'
                        ), num_layers=2))

# 放在forward 部分
output = self.residualmodule(inputs, attention_mask)
```
例子详见：https://github.com/aivolcano/BERT_MRC_CLS/blob/main/text_classification/model.py


### 第二种写法：向量拼接
高维非线性特征和没有经过神经网络的低纬线性特征拼接。
```python 
'''TensorFlow'''
tf.concat([nn_output, ori_inputs], axis=-1)

'''PyTorch'''
torch.cat([nn_output, ori_input], dim=-1)

```
例子详见：https://github.com/aivolcano/RecSys_tf2/blob/main/DeepFM/model.py

### 残差网络中的非线性泛化
非线性特征的实现方式有：
   * 激活函数：ReLU、GeLU、leakyrelu、tanh、sigmoid、softmax
   * 分段函数：MLR、ReLU
   * 多项式：x^2, x ** 0.5, log(x)
   * 神经网络：GRU、LSTM、CNN、Transformer、自定义神经网络W_2 σ(W_1 x)
   * 向量哈达玛积、向量笛卡尔积、特征组合

```python 
'''向量外积 + 残差网络'''
# TensorFlow
out_product = tf.multiply(inputs_from_user_activation, inputs from ad)
attention_weight = tf.keras.layers.Prelu()(tf.concat([inputs_from_user_activation, out_product, inputs from ad], axis=-1)

# PyTorch
Out_product = torch.mm(inputs_from_user_activation, inputs from ad)
attention_weight = F.prelu((torch.cat([inputs_from_user_activation, out_product, inputs from ad], dim=-1))
```
![image](https://user-images.githubusercontent.com/68730894/115814147-1355c500-a427-11eb-9e86-45880194eb07.png)


### 控制信息流量：F(x) = βf(x) + wx
对于残差连接x+f(x)，x 和 f(x)权重各占0.5，我们也可以使用加权控制信息的流量，由神经网络自己控制遗忘多少信息，保留多少信息。这可以通过设置标量参数β实现，x + βf(x)表达式可以自动学习β的权重。
![image](https://user-images.githubusercontent.com/68730894/115814241-40a27300-a427-11eb-9781-52940c59e845.png)

