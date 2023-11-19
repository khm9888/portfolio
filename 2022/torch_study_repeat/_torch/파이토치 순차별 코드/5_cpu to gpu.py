# PYTORCH에서 추론(INFERENCE)을 위해 모델 저장하기 & 불러오기

import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#
        x = self.pool(F.relu(self.conv2(x)))#
        x = x.view(-1, 16 * 5 * 5)
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

# 3. GPU에서 저장하고 CPU에서 불러오기
# GPU로 학습된 모델을 CPU에서 불러올 때는 torch.load() 함수의 map_location 인자에 ``torch.device(‘cpu’)``를 전달합니다.

# 저장하고자 하는 경로를 지정합니다.
PATH = "model.pt"

# 저장하기
torch.save(net.state_dict(), PATH)

# 불러오기
device = torch.device('cpu')
model = Net()
model.load_state_dict(torch.load(PATH, map_location=device))


# 4. GPU에서 저장하고 GPU에서 불러오기
# GPU에서 학습하고 저장된 모델을 GPU에서 불러올 때는, 초기화된 모델에 ``model.to(torch.device(‘cuda’))``을 호출하여 CUDA에 최적화된 모델로 변환해주세요.

# 그리고 모든 입력에 .to(torch.device('cuda')) 함수를 호출해야 모델에 데이터를 제공할 수 있습니다.

# 저장하기
torch.save(net.state_dict(), PATH)

# 불러오기
device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)