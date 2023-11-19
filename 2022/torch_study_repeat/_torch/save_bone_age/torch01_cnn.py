import itertools
from IPython.display import image
from IPython import display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
# import torch.utils.data as data_utils
from torchvision import datasets, transforms

#학습 데이터 다운로드

trn_dataset = datasets.MNIST('../mnist_data/',
                             download=True,
                             train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(), # image to Tensor
                                 transforms.Normalize((0.1307,), (0.3081,)) # image, label
                             ])) 

val_dataset = datasets.MNIST("../mnist_data/", 
                             download=False,
                             train=False,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.N

batch_size = 64


# 초기화 함수에 구조를 다 만들고, forward 함수에서는 optim 함수 정도만 손댄 걸로 보임
# 210524 생각


class MyModel(nn.Module):
    
    def __init__(self, X_dim, y_dim):
        super(MyModel, self).__init__()
        layer1 = nn.Linear(X_dim, 128)
        activation1 = nn.ReLU()
        layer2 = nn.Linear(128, y_dim)
        self.module = nn.Sequential(
            layer1,
            activation1,
            layer2
        )
        
    def forward(self, x):
        out = self.module(x)
        result = F.softmax(out, dim=1)
        return result     
    
##############################학습 뼈대코드#####################################

# model = MyModel()

# 준비재료
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-5
optimizer = optim.SGD(model.paraeters(), lr=learning_rate)
num_epochs = 2
num_batches = len(train_loader)

for epoch in range(num_epochs):
	for i, data in enumerate(train_loader):
		x, x_labels = data # x.size() = [batch, channel, x, y]
		# init grad
		optimizer.zero_grad() # step과 zero_grad는 쌍을 이루는 것이라고 생각하면 됨
		# forward
		pred = model(x)
		# calculate loss
		loss = criterion(pred, x_labels)
		# backpropagation
		loss.backward()
		# weight update
		optimizer.step()
		# 학습과정 출력
		running_loss += loss.item()
		if (i+1)%2000 == 0: # print every 2000 mini-batches
			print("epoch: {}/{} | step: {}/{} | loss: {:.4f}".format(epoch, num_epochs, i+1, num_batches, running_loss/2000))
			running_loss = 0.0

print("finish Training!")