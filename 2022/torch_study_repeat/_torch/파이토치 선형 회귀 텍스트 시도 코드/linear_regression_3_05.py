#05. 클래스로 파이토치 모델 구현하기

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# # 2. 단순 선형 회귀 클래스로 구현하기

# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear=nn.Linear(1,1)

#     def forward(self,x):
#         return self.linear(x)
    
# model = LinearRegressionModel()

# torch.manual_seed(1)

# x_train = torch.Tensor(np.reshape(range(1,4),(-1,1)))
# y_train = torch.Tensor(np.reshape(range(2,7,2),(-1,1)))
# print(x_train)
# print(y_train)

# model = LinearRegressionModel()

# optimizer = optim.SGD(model.parameters(),lr=1e-2)

# nb_epoch = 2000

# for epoch in range(nb_epoch+1):

#     prediction=model(x_train)

#     cost = F.mse_loss(prediction,y_train)

#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()

#     if epoch % 100 ==0:
#         print(f"epoch : {epoch:4d}/{nb_epoch} cost : {cost:.6f}")


#3. 다중 선형 회귀 클래스로 구현하기

class Multivariate_LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,1)

    def forward(self,x):
        return self.linear(x)
    

x_train = torch.FloatTensor(np.reshape(([73,93,89,96,73],[80,88,91,98,66],[75,93,90,100,70]),(-1,3)))
y_train = torch.FloatTensor(np.reshape([152,185,180,196,142],(-1,1)))

model = Multivariate_LinearRegressionModel()

optimizer = optim.SGD(model.parameters(),lr=1e-5)

nb_epoch=2000
for epoch in range(nb_epoch):
    prediction = model(x_train)

    cost = F.mse_loss(prediction,y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch == nb_epoch//10:
        print("y_pred")
        print(prediction)
        print("y_train")
        print(y_train)
        print(cost)

    if epoch%100==0:
        print(f"epoch : {epoch:4d}/{nb_epoch}, Cost : {cost:.6f}")
        