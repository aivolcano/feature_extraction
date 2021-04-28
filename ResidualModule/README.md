原始论文中已经说得很清楚了，笔者在这里说一说自己对Wide & Deep的理解：

从模型内容Pooling的思路出发，如果Wide和Deep部分同时喂相同的特征，Wide & Deep是可以用残差网络解释的，并且从一定程度上说，Wide & Deep 自带残差网络。为什么？

F(x) = f(x) + x， f(x)可视为DNN（非线性），x可视为LR（线性）

研究者发现：深层神经网络会让模型退化的主要原因是激活函数ReLU的存在。数据经过激活函数之后是无法再反推回到原始状态的，整个过程是不可逆的。即当使用ReLU等激活函数时，会导致信息丢失：

低维（2维）的信息嵌入到n维的空间中，并通过随机矩阵T对特征进行变换，再加上ReLU激活函数，之后在通过 T^(-1)（反变换）进行反变换。当n=2，3时，会导致比较严重的信息丢失，部分特征重叠到一起了；当n=15到30时，信息丢失程度降低。这是因为非线性激活函数（Relu）的存在，每次输入到输出的过程都几乎是不可逆的（信息损失），所以很难从输出反推回完整的输入。

![image](https://user-images.githubusercontent.com/68730894/116391762-2535ce80-a852-11eb-9998-e4e0bf48ae9b.png)
![image](https://user-images.githubusercontent.com/68730894/116391781-2830bf00-a852-11eb-95ed-775d568b7240.png)


从数学公式看，如果没有非线性激活函数，残差网络存在与否意义不大。如果残差网络存在，则只是做了简单的平移： 

![image](https://user-images.githubusercontent.com/68730894/116391848-3da5e900-a852-11eb-9770-cd868184e411.png)

增加非线性激活函数之后，上述式子发生改变，模型的特征表达能力大幅提升。这也是为什么Residual Block有2个权重（W_1,W_2）的原因。

![image](https://user-images.githubusercontent.com/68730894/116391900-50b8b900-a852-11eb-9a67-cfb9b91b90b8.png)


残差网络升级扩展，数学上证明： 原始残差网络

![image](https://user-images.githubusercontent.com/68730894/115329746-557fcc00-a1c5-11eb-859e-a0871c94a291.png)

增加非线性激活

![image](https://user-images.githubusercontent.com/68730894/115329737-50228180-a1c5-11eb-9b83-280ac2fbe376.png)

为了实现一直堆叠网络而不发生网络退化的需要，何凯明让模型内部结构具备恒等映射能力：将上一层（或几层）之前的输出与本层计算的输出相加，可以将求和的结果输入到激活函数中做为本层的输出。

![image](https://user-images.githubusercontent.com/68730894/116392071-92496400-a852-11eb-81a7-918735f88483.png)

### 代码结构
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


### 使用残差网络解释 Wide & Deep 和 DeepFM
* Wide & Deep
作者认为 Wide & Deep 是自带残差网络的，只是可能我们暂时没有发现，针对我们改进后的残差网络 F(x) = βf(x) + wx ，是 Wide & deep 的数学表达，f(x) 表示 DNN，赋予数据非线性特征， x 表示LR，赋予数据线性特征

![image](https://user-images.githubusercontent.com/68730894/116390304-79d84a00-a850-11eb-829d-c5f39a3983f0.png)

更详细的介绍：https://github.com/aivolcano/RecSys_tf2/tree/main/Wide%26Deep

* DeepFM
如果我们按照类别特征喂给FM，类别特征+连续特征喂给DNN，我们就可以对DNN部分使用残差网络。!

![v2-3e4ab11c3d125df0d3659691b2b116da_1440w](https://user-images.githubusercontent.com/68730894/116390796-fbc87300-a850-11eb-94ea-b5f76367333a.jpg)

并且我们认为FM是自带残差网络效果的。

[v2-3e62201e619ea3452de06ee9ed0ac0cc_1440w](https://user-images.githubusercontent.com/68730894/116391019-3c27f100-a851-11eb-93d5-ab3f46976304.jpg)

更详细的介绍：https://github.com/aivolcano/RecSys_tf2/tree/main/DeepFM


