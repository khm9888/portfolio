from collections import Counter

#데이터 처리
#모델 생성
#모델 실행
#데이터 시각화
# (추정)

########3nd##########################


#데이터 처리
# x,y 값에 대한 선언

#모델 생성
#회귀 모델에 대한 define, 초기화 함수, 실행 함수, 

#모델 실행
# 1.모델 선언, optimzier//criterion 에 대한 선언
# 2.파라미터(가중치, 편향에 대한 함수)
# 랜덤 시드 설정
#  parameters 가져오는 함수. item() 적용할 것.
# 3.그래프 시각화 코드 작성
#  제목 / 파라미터 가져와서 x1에 대해서 쓰고, 
#  x 값을 활용하여 y 선언,
#  x,y 값을 통해 시각화 구현.(산점도도 구현)
# 4. epoch 설정하여, 학습진행
#  optimizer.zero_grad() - 초기화
#  loss 계산.
#  역전파
#  optimizer 업데이트.
#  결과 출력

# 데이터 시각화
# 시각화 선언.

#########3nd-end#########################


###########2nd############################
## 모듈 선언

#데이터 처리
# x,y 값에 대한 선언

#모델 생성
#회귀 모델에 대한 define, 초기화 함수, 실행 함수, 

#모델 실행
# 1.모델 선언, optimzier//criterion 에 대한 선언
# 2.파라미터(가중치, 편향에 대한 함수)
# 랜덤 시드 설정
#  parameters 가져오는 함수. item() 적용할 것.
# 3.그래프 시각화 코드 작성
#  제목 / 파라미터 가져와서 x1에 대해서 쓰고, 
#  x 값을 활용하여 y 선언,
#  x,y 값을 통해 시각화 구현.(산점도도 구현)
# 4. epoch 설정하여, 학습진행
#  optimizer.zero_grad() - 초기화
#  loss 계산.
#  역전파
#  optimizer 업데이트.
#  결과 출력

# 데이터 시각화
# 시각화 선언.

## 모듈 선언
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

#데이터 처리
# x,y 값에 대한 선언

x = torch.randn(100,1)*10
y = x+torch.randn(100,1)*3

#모델 생성
#회귀 모델에 대한 define, 초기화 함수, 실행 함수, 
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(1,1)
        
    def forward(self,x):
        return self.linear(x)
        # pred = self.linear(x)
        # return pred


#모델 실행
# 1.모델 선언, optimzier//criterion 에 대한 선언
model = LinearRegressionModel()

optimizer = optim.SGD(model.parameters(),lr=0.001)
optimizer = optim.SGD(model.parameters(),lr=0.0001)#보다 나았음.
criterion = nn.MSELoss()
# 2.파라미터(가중치, 편향에 대한 함수)
# 랜덤 시드 설정
torch.manual_seed(111)
# torch.manual_seed(111)

def get_params():
    w,b = model.parameters()
    return w[0][0].item(),b[0].item() #

# 3.그래프 시각화 코드 작성

def plt_show(title):
#  제목 / 파라미터 가져와서 x1에 대해서 쓰고, 
    plt.title(title)
#  x 값을 활용하여 y 선언,
    x1 = np.array(range(-30,31))
    w1,b1 = get_params()
    y1 = w1*x1+b1
    plt.plot(x1,y1,"r")#<- plot에 대해 차후 공부
#  x,y 값을 통해 시각화 구현.(산점도도 구현)
    plt.scatter(x,y)
    plt.show()

plt_show("initial_show")
# 4. epoch 설정하여, 학습진행
epochs = 100
losses =[]

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred,y)
    losses.append(loss)
    loss.backward() #역전파를 왜 나중에 하는가..? losses에 추가하기 전에 해야하는 건 아닌가?
    # 위에 대한 내 생각, loss 는 계산하고, 현재 값은 반영해야하니까.(학습)
    optimizer.step()
#  optimizer.zero_grad() - 초기화
#  loss 계산.
#  역전파
#  optimizer 업데이트.
#  결과 출력
    if epoch%10 ==0:
        print(f"epoch : {epoch} loss : {loss}")

# 데이터 시각화 - epoch 대비, loss
plt.plot(range(epochs),losses)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
# 시각화 선언.


###########2nd-end#########################

###########1st#########################
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
#데이터 처리
x= torch.randn(100,1)*10

y = x+3*torch.randn(100,1)

#모델 생성

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(1,1)
        

    def forward(self,x):
        pred = self.linear(x)
        return pred
        
        
torch.manual_seed(111)

model = LinearRegressionModel()
print(model)

# print(model.parameters())
# print(list(model.parameters()))

# tensor([[0.4311]], requires_grad=True), Parameter containing:
# tensor([0.8280], requires_grad=True)]

w,b = model.parameters()

def get_params():
    return w[0][0].item(),b[0].item()
    
def plot_fit(title):
    plt.title = title
    w1, b1 = get_params()
    x1 = np.array(range(-30,31))
    
    y1 = w1*x1+b1
    print(y1)
    plt.plot(x1,y1,"r")
    plt.scatter(x,y)
    plt.show()
    
plot_fit("initial_Model")
#모델 실행

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)


epochs = 100
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred=model(x)
    loss = criterion(y_pred,y)
    losses.append(loss)
    loss.backward()
    
    optimizer.step()
#결과 출력
    if epoch % 10 ==0:
        print(f"epoch : {epoch} loss : {loss}")

print(0)
print(losses[0])

#데이터 시각화

plt.plot(range(epochs),losses)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()
###########1st-end#########################