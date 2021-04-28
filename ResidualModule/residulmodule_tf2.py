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


# 例子：https://github.com/aivolcano/RecSys_tf2/blob/main/DeepFM/model.py
