import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# x_train = torch.FloatTensor(np.reshape(([73,93,89,96,73,80,88,91,98,66,75,93,90,100,70]),(5,3)))
# y_train = torch.FloatTensor(np.reshape([152,185,180,196,142],(5,1)))

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train,y_train)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(),lr = 1e-5)

# with open("/home/con/torch_test/record.txt","a") as file:

#     file.write("-------date--------"+"\n")
#     file.write(str(datetime.date)+"\n")
#     file.write("-------------------"+"\n")

nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train,y_train = samples
        prediction = model(x_train)

        cost = F.mse_loss(prediction,y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


        sentence = f"Epoch {epoch:4d}/{nb_epochs} Batch {batch_idx+1}/{len(dataloader)} Cost: {cost.item():.6f}"
        print(sentence)
        # with open("/home/con/torch_test/record.txt","a") as file:
        #     file.write(sentence+"\n")