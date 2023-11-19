# 03. 다중 선형 회귀(Multivariable Linear regression)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

x1_train = torch.FloatTensor(np.reshape([73,93,89,96,73],(-1,1)))
x2_train = torch.FloatTensor(np.reshape([80,88,91,98,66],(-1,1)))
x3_train = torch.FloatTensor(np.reshape([75,93,90,100,70],(-1,1)))
y_train = torch.FloatTensor(np.reshape([152,185,180,196,142],(-1,1)))

print(x1_train)
print(x2_train)
print(x3_train)
print(y_train)

w1 = torch.zeros(1,requires_grad = True)
w2 = torch.zeros(1,requires_grad = True)
w3 = torch.zeros(1,requires_grad = True)
b = torch.zeros(1,requires_grad = True)

optimizer = optim.SGD([w1,w2,w3,b],lr = 1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    hypothesis = x1_train*w1+x2_train*w2+x3_train*w3+b

    cost = torch.mean((hypothesis-y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%50==0:
        print(f"Epoch : {epoch}/{nb_epochs} w1:{w1.item():.3f} w2:{w2.item():.3f} w3:{w3.item():.3f} b:{b.item():.3f} Cost:{cost.item():.6f}")


# 4. 행렬 연산을 고려하여 파이토치로 구현하기

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

x_train = torch.FloatTensor(np.reshape(([73,93,89,96,73],[80,88,91,98,66],[75,93,90,100,70]),(-1,3)))
# x2_train = torch.FloatTensor(np.reshape([80,88,91,98,66],(-1,1)))
# x3_train = torch.FloatTensor(np.reshape([75,93,90,100,70],(-1,1)))
y_train = torch.FloatTensor(np.reshape([152,185,180,196,142],(-1,1)))

print(x_train)
print(y_train)
print(x_train.shape)
print(y_train.shape)

W = torch.zeros((3,1),requires_grad = True)
b = torch.zeros(1,requires_grad = True)

optimizer = optim.SGD([W,b],lr = 1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    hypothesis = x_train.matmul(W)+b

    cost = torch.mean((hypothesis-y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%50==0:
        print(f"Epoch : {epoch}/{nb_epochs} hypothesis:{hypothesis.squeeze().detach()}  b:{b.item():.3f} Cost:{cost.item():.6f}")