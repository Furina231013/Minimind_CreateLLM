import torch

# # 条件选择 x中不符合要求的由y中数据填充对应位置
# x = torch.tensor([1, 2, 3, 4, 5])
# y = torch.tensor([10, 20, 30, 40, 50])
# condition = x > 3
# result = torch.where(condition, x, y)
# print(result)  # 输出: tensor([10, 20, 30,  4,  5])

# t = torch.arange(0, 10, 2)
# print(t)  # 输出: tensor([0, 2, 4, 6, 8])
# t2 = torch.arange(5, 0, -1)
# print(t2)  # 输出: tensor([5, 4, 3, 2, 1])

# v1 = torch.tensor([1,2,3])
# v2 = torch.tensor([4,5,6])
# result = torch.outer(v1, v2)
# print(result)
# 输出：
# tensor([[ 4,  5,  6],
#         [ 8, 10, 12],
#         [12, 15, 18]])

# t1 = torch.tensor([[[1,2,3], [4,5,6]],[[7,8,9],[10,11,12]]])
# t2 = torch.tensor([[[1,0,0], [0,1,0]],[[0,0,1],[1,0,0]]])
# result = torch.cat((t1,t2), dim=0)
# print(result) # 生成一个新的张量，在第一维度上进行拼接

t1 = torch.Tensor([1,2,3])
t2 = t1.unsqueeze(0) # 在第0维增加一个维度
print(t1.shape)
print(t2)
print(t2.shape)