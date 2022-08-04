# -*- coding: utf-8 -*-
# 使用pandas 读取上面介绍的数据，并进行简单的缺失值填充
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.models import *
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_fixlen_feature_names
import torch
 
# 使用pandas 读取上面介绍的数据，并进行简单的缺失值填充
data = pd.read_csv('./criteo_sample.txt')
# 上面的数据在：https://github.com/shenweichen/DeepCTR-Torch/blob/master/examples/criteo_sample.txt
 
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]
 
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']
 
#这里我们需要对特征进行一些预处理，对于类别特征，我们使用LabelEncoder重新编码(或者哈希编码)，对于数值特征使用MinMaxScaler压缩到0~1之间。
 
for feat in sparse_features:
   lbe = LabelEncoder()
   data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])
 
# 这里是比较关键的一步，因为我们需要对类别特征进行Embedding，所以需要告诉模型每一个特征组有多少个embbedding向量，我们通过pandas的nunique()方法统计。
 
 
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                                                           for feat in dense_features]
 
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
 
fixlen_feature_names = get_fixlen_feature_names(
   linear_feature_columns + dnn_feature_columns)
 
#最后，我们按照上一步生成的特征列拼接数据
 
train, test = train_test_split(data, test_size=0.2)
train_model_input = [train[name] for name in fixlen_feature_names]
test_model_input = [test[name] for name in fixlen_feature_names]
 
# 检查是否可以使用gpu
 
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
   print('cuda ready...')
   device = 'cuda:0'
 
# 初始化模型，进行训练和预测
 
model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, task='binary',
               l2_reg_embedding=1e-5, device=device)
 
model.compile("adagrad", "binary_crossentropy",
               metrics=["binary_crossentropy", "auc"],)
model.fit(train_model_input, train[target].values,
           batch_size=256, epochs=10, validation_split=0.2, verbose=2)
 
pred_ans = model.predict(test_model_input, 256)
print("")
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))