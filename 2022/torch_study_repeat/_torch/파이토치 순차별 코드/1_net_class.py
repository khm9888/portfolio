#PYTORCH에서 STATE_DICT란 무엇인가요?

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import datetime
import torchaudio

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# +@ 기록 하기
location_now = "/home/con/tutorial_torch/"

with open(f"{location_now}/record.txt","a") as file:

    file.write("------date-----------\n")
    file.write(str(datetime.datetime.today())+"\n")
    file.write("------date-----------\n")

#데이터입력
yesno_data = torchaudio.datasets.YESNO('./', download=True)

# print(24)
n = 3
waveform, sample_rate, labels = yesno_data[n]
# print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))
# Waveform: tensor([[ 3.0518e-05,  6.1035e-05,  3.0518e-05,  ..., -1.8311e-04,
#           4.2725e-04,  6.7139e-04]])
# Sample rate: 8000
# Labels: [0, 0, 1, 0, 0, 0, 1, 0]

data_loader = torch.utils.data.DataLoader(yesno_data,batch_size = 1,shuffle =True)

# for data in data_loader:
#     print("Data:",data)
#     print("Waveform: {}\nSample rate: {}\nLabels: {}".format(data[0], data[1], data[2]))
#     break

data = list(data_loader)[0]


# plt 사용
#######################################
import matplotlib.pyplot as plt

print(data[0][0].numpy)

plt.figure()
plt.plot(waveform.t().numpy())
plt.savefig(location_now+"save.png")

plt.show()#<- 서버에서 실행하기 때문에 상단의 savefig 처럼 처리함.
#######################################

#모델구성
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 첫번째 2D 컨볼루션 계층
        # 1개의 입력 채널(이미지)을 받아들이고, 사각 커널 사이즈가 3인 32개의 컨볼루션 특징들을 출력합니다.
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 두번째 2D 컨볼루션 계층
        # 32개의 입력 게층을 받아들이고, 사각 커널 사이즈가 3인 64개의 컨볼루션 특징을 출력합니다.
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        # 인접한 픽셀들은 입력 확률에 따라 모두 0 값을 가지거나 혹은 모두 유효한 값이 되도록 만듭니다.
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        # 첫번째 fully connected layer
        self.fc1 = nn.Linear(9216, 128)
        # 10개의 라벨을 출력하는 두번째 fully connected layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self,x):#순전파 함수
        # 데이터가 conv1을 지나갑니다.
        x = self.conv1(x)
        # x를 ReLU 활성함수(rectified-linear activation function)에 대입합니다.
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # x에 대해서 max pooling을 실행합니다.
        x = F.max_pool2d(x, 2)
        # 데이터가 dropout1을 지나갑니다.
        x = self.dropout1(x)
        # start_dim=1으로 x를 압축합니다.
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # 데이터가 fc1을 지나갑니다.
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x,dim=1)
        return output


#another -> 샘플 예측
random_data = torch.rand((1,1,28,28))

my_nn = Net()

result = my_nn(random_data)
print(data)

#훈련
#평가 및 예측