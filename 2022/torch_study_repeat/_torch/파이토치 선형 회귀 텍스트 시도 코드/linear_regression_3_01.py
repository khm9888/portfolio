
#경사하강법


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor(np.reshape(range(1,4),(-1,1)))
y_train = torch.FloatTensor(np.reshape(range(2,7,2),(-1,1)))

W = torch.zeros(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)

optimizer = optim.SGD([W,b],lr=0.01)

nb_epochs = 2000
for epoch in range(nb_epochs+1):
    hypothesis = x_train*W+b
    cost = torch.mean((hypothesis-y_train)**2)

    optimizer.zero_grad()#기울기를 0으로 초기화
    cost.backward()#역순으로 자동미분해준다고 함
    optimizer.step()#W,b값을 바꾸어주고

    if epoch%100==0:
        print(f"Epoch {epoch:4d}/{nb_epochs} W: {W.item():.3f}, b: {b.item():.3f}, Cost: {cost.item():.6f}")