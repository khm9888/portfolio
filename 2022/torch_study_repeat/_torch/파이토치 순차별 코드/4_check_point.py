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

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# print(net.forward())


# # 모델의 state_dict 출력
# print("Model's state_dict:")
# for param_tensor in net.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# print()

# # 옵티마이저의 state_dict 출력
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

#4. state_dict 을 통해 모델을 저장하고 불러오기

# 경로 지정
PATH = "state_dict_model.pt"

# 저장하기
torch.save(net.state_dict(), PATH)

# 불러오기
model = Net()
model.load_state_dict(torch.load(PATH))
v=model.eval()


#5. 전체 모델을 저장하고 불러오기

# 경로 지정
PATH = "entire_model.pt"

# 저장하기
torch.save(net, PATH)

# 불러오기
model = torch.load(PATH)
model.eval()

# print(58,v)