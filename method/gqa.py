import torch
import torch.nn as nn

# dropout_layer = nn.Dropout(p=0.3)
#
# t1 = torch.Tensor([1,2,3])
# t2 = dropout_layer(t1)
# print(t2) # 在训练模式下，随机将部分元素置为0，其余元素按比例放大

# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.Tensor([1,2,3])
# t2 = torch.Tensor([[1,2,3]])
# output2 = layer(t2)
# print(t2.shape)
# print(output2)
# print(output2.shape)
# # 线性变换 对应用的张量乘以一个w矩阵再加上偏置b

# t = torch.tensor([[1,2,3,4,5,6], [7,8,9,10,11,12]])
# t_view1 = t.view(3,4)
# print(t_view1)
# t_view2 = t.view(4,3)
# print(t_view2)
# # view不会改变数据，只是改变了张量的形状

t = torch.tensor([[1,2,3,4,5,6], [7,8,9,10,11,12], [5,6,4,7,8,4], [92, 12, 3,4,3,2]])
t1 = t.transpose(-2,0)
print(t)
print(t1)
# transpose交换指定的两个维度 从2维6列转换成6维2列

# x = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(torch.triu(x)) # 上三角矩阵 包括对角线
# print(torch.triu(x, 1)) # 上三角矩阵 不包括对角线

# x = torch.arange(1, 7)
# y = torch.reshape(x, (2,3))
# z = torch.reshape(x, (3,-1))
# print(y)
# print(z)
# # reshape不会改变数据，只是改变了张量的形状，-1表示自动计算该维度的大小


