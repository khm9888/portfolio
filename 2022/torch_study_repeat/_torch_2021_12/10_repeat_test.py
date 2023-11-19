


########3nd##########################
import torch
from torch import random 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


#데이터 처리
# x,y 값에 대한 선언
x = torch.randn(100,1)*3
y = x + 3*torch.randn(100,1)

plt.plot(x.numpy(),y.numpy(), "o")
plt.ylabel("y")
plt.xlabel("x")
plt.grid()
plt.show()

#모델 생성
#회귀 모델에 대한 define, 초기화 함수, 실행 함수, 

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(1,1)
        
    def forward(self,x):
        pred = self.linear(x)
        return pred
    


#모델 실행
# 1.모델 선언, optimzier//criterion 에 대한 선언
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)
# 2.파라미터(가중치, 편향에 대한 함수)
# 랜덤 시드 설정
random.manual_seed(222)
#  parameters 가져오는 함수. item() 적용할 것.
w,b = model.parameters()
def get_params():
    return w[0][0].item(),b[0].item()
# 3.그래프 시각화 코드 작성
#  제목 / 파라미터 가져와서 x1에 대해서 쓰고, 
#  x 값을 활용하여 y 선언,
#  x,y 값을 통해 시각화 구현.(산점도도 구현)
def plot_title(title):
    plt.title(title)
    w1,b1 = get_params()
    # x1 = np.array(range(-30,30))
    x1 = np.array([-30,30])
    y1 = x1*w1+b1
    plt.plot(x1,y1,"r")
    plt.scatter(x,y)
    plt.show()
# 4. epoch 설정하여, 학습진행

plot_title("initial")

epochs = 100
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pre =model(x)
    loss = criterion(y_pre,y)
    losses.append(loss)
    loss.backward()
    optimizer.step()
    if epoch %10 ==0:
        print(f"Epoch : {epoch} loss : {loss:.4f}")
#  optimizer.zero_grad() - 초기화
#  loss 계산.
#  역전파
#  optimizer 업데이트.
#  결과 출력

# 데이터 시각화
plot_title("advanced")
# 시각화 선언.
plt.plot(range(epochs),losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

#########3nd-end#########################