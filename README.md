# feature_extraction

该板块包括残差网络泛化后的模块，该模块可直接在feature_extraction 和 MLP使用

数学支撑： F(x) = f(x) + x  -->  F(x) = βf(x) + wx

w的存在让残差网络不受维度限制的使用，因为增加线性层nn.Linear() / tf.keras.layers.Dense() 将 x 的维度调整到与 f(x) 并不违反残差网络 非线性特征 + 线性特征的初衷

更多残差网络的使用：https://github.com/aivolcano/feature_extraction/ResidualModule


