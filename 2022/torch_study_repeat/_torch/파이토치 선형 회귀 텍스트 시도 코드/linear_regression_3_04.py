""" # 04. nn.Module로 구현하는 선형 회귀


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

x_train = torch.FloatTensor(np.reshape(range(1,4),(-1,1)))
y_train = torch.FloatTensor(np.reshape(range(2,7,2),(-1,1)))

# print(x_train)
# print(y_train)
# print(x_train.shape)
# print(y_train.shape)

# W = torch.zeros((3,1),requires_grad = True)
# b = torch.zeros(1,requires_grad = True)


# model  = nn.Linear(input_dim,output_dim)
model  = nn.Linear(1,1)

# print(list(model.parameters()))

optimizer = optim.SGD(model.parameters(),lr=1e-2)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    prediction = model(x_train)

    cost = F.mse_loss(prediction,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch:4d}/{nb_epochs} Cost: {cost:.6f}")


new_var = torch.Tensor([4])
pred_y = model(new_var)

print(f"훈련 후 입력이 4일 때의 예측값: {pred_y}")

print(list(model.parameters()))
 """
# 2. 다중 선형 회귀 구현하기

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

x_train = torch.FloatTensor(np.reshape(([73,93,89,96,73],[80,88,91,98,66],[75,93,90,100,70]),(-1,3)))
y_train = torch.FloatTensor(np.reshape([152,185,180,196,142],(-1,1)))

model = nn.Linear(3,1)

print(list(model.parameters()))

optimizer = optim.SGD(model.parameters(),lr=1e-5)

nb_epochs = 3000
for epoch in range(nb_epochs+1):

    prediction = model(x_train)

    cost = F.mse_loss(prediction,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100==0:
        print(f"epoch : {epoch:4d}/{nb_epochs}, Cost: {cost:.6f}")
        # print(f"epoch : {epoch:4d}/{nb_epochs}, Cost: {cost.item():.6f}")




x_test = torch.Tensor([[73,80,75]])

y_predict = model(x_test)
print(f"{y_predict}")