# Create time: 2021.4.12 13:23
# Authod: ChenYancan
# E-mail: ican22@foxmail.com


import torch 
import torch.nn as nn

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

  
                                     
# 例子详见：https://github.com/aivolcano/BERT_MRC_CLS/blob/main/text_classification/model.py
