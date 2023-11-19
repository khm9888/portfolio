import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import torchaudio

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#데이터입력


voice = torchaudio.datasets.YESNO(root,
  url='http://www.openslr.org/resources/1/waves_yesno.tar.gz',
  folder_in_archive='waves_yesno',
  download=False,
  transform=None,
  target_transform=None))

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]]) 

y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train,y_train)#데이터 

dataloader = DataLoader(dataset,batch_size=2, shuffle=True)

#모델구성
model = nn.Linear(3,1)#선형모델

optimizer = torch.optim.SGD(model.parameters(),lr = 1e-5)#활성화 함수

# +@ 기록 하기
with open("/home/con/torch_test/record.txt","a") as file:

    file.write("------date-----------\n")
    file.write(str(datetime.datetime.today())+"\n")
    file.write("------date-----------\n")
#훈련

nb_epochs = 20
for epoch in range(nb_epochs+1):
    for batch_idx,samples in enumerate(dataloader):
        x_train,y_train = samples#이번 epoch에서 실행될 데이터
        
        prediction = model(x_train)#모델 예상값

        cost = F.mse_loss(prediction,y_train)#코스트(낮아야 좋음)

        optimizer.zero_grad()#0으로 초기화
        cost.backward()#역전파
        optimizer.step()# 뭐라구 설명하지...음

        sentence = f"Epoch {epoch:4d}/{nb_epochs} Batch {batch_idx+1}/{len(dataloader)} Cost: {cost.item():.6f}"
        # print(samples)
        print(sentence)
        with open("/home/con/torch_test/record.txt","a") as file:
            # file.write(str(samples)+"\n")
            file.write(sentence+"\n")
#평가 및 예측