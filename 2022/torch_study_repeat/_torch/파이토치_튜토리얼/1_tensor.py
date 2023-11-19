#https://tutorials.pytorch.kr/beginner/basics/tensorqs_tutorial.html

import torch
import numpy as np

print(torch.__version__)

# data = [[1, 2],[3, 4]]
# x_data = torch.tensor(data)

# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
# print(f"x_np: \n {x_np} \n")

# x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
# print(f"Ones Tensor: \n {x_ones} \n")

# x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
# print(f"Random Tensor: \n {x_rand} \n")

# ''' x_np: 
#  tensor([[1, 2],
#         [3, 4]]) 
 
# Ones Tensor: 
#  tensor([[1, 1],
#         [1, 1]]) 

# Random Tensor: 
#  tensor([[0.8097, 0.6581],
#         [0.6501, 0.8227]]) '''
 
# shape = (2,3,)
# rand_tensor = torch.rand(shape)
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")


# ''' Random Tensor: 
#  tensor([[0.1501, 0.3660, 0.2072],
#         [0.4758, 0.4551, 0.1728]]) 

# Ones Tensor: 
#  tensor([[1., 1., 1.],
#         [1., 1., 1.]]) 

# Zeros Tensor: 
#  tensor([[0., 0., 0.],
#         [0., 0., 0.]]) '''

# tensor = torch.rand(3,4)

# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")

# ''' Shape of tensor: torch.Size([3, 4])
# Datatype of tensor: torch.float32
# Device tensor is stored on: cpu '''


# tensor = torch.tensor(np.arange(0,16).reshape(4,4))#적용안됨
tensor = torch.FloatTensor(np.arange(0,16).reshape(4,4))#응용
# tensor = torch.IntTensor(np.arange(0,16).reshape(4,4))#적용 안 됨

# tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[:, -1])
# print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

# GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
y1 = tensor * tensor.T
print(f"y1:{y1}")
y2 = tensor.matmul(tensor.T)

print(f"y2:{y2}")

y3 = torch.rand_like(tensor)
print(torch.matmul(tensor, tensor.T, out=y3))


# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item()
print(f"agg:{agg}, agg_item:{agg_item}, type(agg_item):{type(agg_item)}")
# agg:92.0, agg_item:92.0, type(agg_item):<class 'float'>


#바꿔치기 연산
print(tensor, "\n")
tensor.add_(5)
print(tensor)

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")