# PYTORCH에서 변화도를 0으로 만들기

# 1. 데이터를 불러오기 위해 필요한 모든 라이브러리 import 하기
# 이 레시피에서는 데이터셋에 접근하기 위해 torch 와 torchvision 을 사용합니다.

import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# 2. 데이터셋 불러오고 정규화하기

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 3. 신경망 구축하기
# 컨볼루션 신경망을 정의하겠습니다. 자세한 내용은 신경망 정의하기 레시피를 참조해주세요.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. 손실 함수과 옵티마이저 정의하기
# 분류를 위한 Cross-Entropy 손실 함수와 모멘텀을 설정한 SGD 옵티마이저를 사용합니다.

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 5. 신경망을 학습시키는 동안 변화도를 0으로 만들기
# 이제부터 흥미로운 부분을 살펴보려고 합니다. 여기서 할 일은 데이터 이터레이터를 순회하면서, 신경망에 입력을 주고 최적화하는 것입니다.

# 데이터의 엔터티 각각의 변화도를 0으로 만들어주는 것에 유의하십시오. 신경망을 학습시킬 때 불필요한 정보를 추적하지 않도록 하기 위함입니다.

for epoch in range(2):  # 전체 데이터셋을 여러번 반복하기

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 입력 받기: 데이터는 [inputs, labels] 형태의 리스트
        inputs, labels = data

        # 파라미터 변화도를 0으로 만들기
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        running_loss += loss.item()
        if i % 2000 == 1999:    # 미니배치 2000개 마다 출력
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')