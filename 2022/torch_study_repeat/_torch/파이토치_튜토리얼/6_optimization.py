import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()


#############################################################################

# 하이퍼파라매터(Hyperparameter)

# 하이퍼파라매터(Hyperparameter)는 모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수입니다. 
# 서로 다른 하이퍼파라매터 값은 모델 학습과 수렴율(convergence rate)에 영향을 미칠 수 있습니다. 
# (하이퍼파라매터 튜닝(tuning)에 대해 더 알아보기)

# 학습 시에는 다음과 같은 하이퍼파라매터를 정의합니다:
# 에폭(epoch) 수 - 데이터셋을 반복하는 횟수
# 배치 크기(batch size) - 각 에폭에서 한 번에 모델에 전달할 데이터 샘플의 수
# 학습율(learning rate) - 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 
# 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생할 수 있습니다.

learning_rate = 1e-3
batch_size = 64
epochs = 5

#############################################################################

# 최적화 단계(Optimization Loop)

# 하이퍼파라매터를 설정한 뒤에는 최적화 단계를 통해 모델을 학습하고 최적화할 수 있습니다. 최적화 단계의 각 반복(iteration)을 에폭이라고 부릅니다.

# 하나의 에폭은 다음 두 부분으로 구성됩니다:
# 학습 단계(train loop) - 학습용 데이터셋을 반복(iterate)하고 최적의 매개변수로 수렴합니다.
# 검증/테스트 단계(validation/test loop) - 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복(iterate)합니다.

#############################################################################

# 손실 함수(loss function)

# 학습용 데이터를 제공하면, 학습되지 않은 신경망은 정답을 제공하지 않을 확률이 높습니다. 손실 함수(loss function)는 획득한 결과와 실제 값 사이의 틀린 정도(degree of dissimilarity)를 측정하며, 학습 중에 이 값을 최소화하려고 합니다. 주어진 데이터 샘플을 입력으로 계산한 예측과 정답(label)을 비교하여 손실(loss)을 계산합니다.
# 일반적인 손실함수에는 회귀 문제(regression task)에 사용하는 nn.MSELoss(평균 제곱 오차(MSE; Mean Square Error))나 분류(classification)에 사용하는 nn.NLLLoss (음의 로그 우도(Negative Log Likelihood)), 그리고 nn.LogSoftmax와 nn.NLLLoss를 합친 nn.CrossEntropyLoss 등이 있습니다.
# 모델의 출력 로짓(logit)을 nn.CrossEntropyLoss에 전달하여 로짓(logit)을 정규화하고 예측 오류를 계산합니다.

# 손실 함수를 초기화합니다.
loss_fn = nn.CrossEntropyLoss()

#############################################################################

# 옵티마이저(Optimizer)

# 최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정입니다. 
# 최적화 알고리즘은 이 과정이 수행되는 방식(여기에서는 확률적 경사하강법(SGD; Stochastic Gradient Descent))을 정의합니다. 
# 모든 최적화 절차(logic)는 optimizer 객체에 캡슐화(encapsulate)됩니다. 
# 여기서는 SGD 옵티마이저를 사용하고 있으며, PyTorch에는 ADAM이나 RMSProp과 같은 다른 종류의 모델과 데이터에서 
# 더 잘 동작하는 다양한 옵티마이저가 있습니다.

# 학습하려는 모델의 매개변수와 학습율(learning rate) 하이퍼파라매터를 등록하여 옵티마이저를 초기화합니다.

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#############################################################################

# 전체 구현

# 최적화 코드를 반복하여 수행하는 train_loop와 
# 테스트 데이터로 모델의 성능을 측정하는 test_loop를 정의하였습니다.


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# Epoch 1
# -------------------------------
# loss: 2.300364  [    0/60000]
# loss: 2.295813  [ 6400/60000]
# loss: 2.291142  [12800/60000]
# loss: 2.295828  [19200/60000]
# loss: 2.265892  [25600/60000]
# loss: 2.267982  [32000/60000]
# loss: 2.267560  [38400/60000]
# loss: 2.251968  [44800/60000]
# loss: 2.263330  [51200/60000]
# loss: 2.266193  [57600/60000]
# Test Error:
#  Accuracy: 39.0%, Avg loss: 0.035231

# Epoch 2
# -------------------------------
# loss: 2.243153  [    0/60000]
# loss: 2.232785  [ 6400/60000]
# loss: 2.225377  [12800/60000]
# loss: 2.255007  [19200/60000]
# loss: 2.170355  [25600/60000]
# loss: 2.190519  [32000/60000]
# loss: 2.198742  [38400/60000]
# loss: 2.161859  [44800/60000]
# loss: 2.206608  [51200/60000]
# loss: 2.211178  [57600/60000]
# Test Error:
#  Accuracy: 37.7%, Avg loss: 0.033950

# Epoch 3
# -------------------------------
# loss: 2.167587  [    0/60000]
# loss: 2.137901  [ 6400/60000]
# loss: 2.126740  [12800/60000]
# loss: 2.195847  [19200/60000]
# loss: 2.020861  [25600/60000]
# loss: 2.075463  [32000/60000]
# loss: 2.101159  [38400/60000]
# loss: 2.035974  [44800/60000]
# loss: 2.136377  [51200/60000]
# loss: 2.146552  [57600/60000]
# Test Error:
#  Accuracy: 37.6%, Avg loss: 0.032313

# Epoch 4
# -------------------------------
# loss: 2.073075  [    0/60000]
# loss: 2.020180  [ 6400/60000]
# loss: 2.011133  [12800/60000]
# loss: 2.126059  [19200/60000]
# loss: 1.862594  [25600/60000]
# loss: 1.963003  [32000/60000]
# loss: 2.007977  [38400/60000]
# loss: 1.924553  [44800/60000]
# loss: 2.069266  [51200/60000]
# loss: 2.097358  [57600/60000]
# Test Error:
#  Accuracy: 40.3%, Avg loss: 0.030882

# Epoch 5
# -------------------------------
# loss: 1.988646  [    0/60000]
# loss: 1.923417  [ 6400/60000]
# loss: 1.916111  [12800/60000]
# loss: 2.062968  [19200/60000]
# loss: 1.743387  [25600/60000]
# loss: 1.880766  [32000/60000]
# loss: 1.933582  [38400/60000]
# loss: 1.844118  [44800/60000]
# loss: 2.004896  [51200/60000]
# loss: 2.054718  [57600/60000]
# Test Error:
#  Accuracy: 41.6%, Avg loss: 0.029716

# Epoch 6
# -------------------------------
# loss: 1.915899  [    0/60000]
# loss: 1.848167  [ 6400/60000]
# loss: 1.839683  [12800/60000]
# loss: 2.008823  [19200/60000]
# loss: 1.653246  [25600/60000]
# loss: 1.816632  [32000/60000]
# loss: 1.873032  [38400/60000]
# loss: 1.784300  [44800/60000]
# loss: 1.947052  [51200/60000]
# loss: 2.016952  [57600/60000]
# Test Error:
#  Accuracy: 42.3%, Avg loss: 0.028760

# Epoch 7
# -------------------------------
# loss: 1.854357  [    0/60000]
# loss: 1.788710  [ 6400/60000]
# loss: 1.777642  [12800/60000]
# loss: 1.963411  [19200/60000]
# loss: 1.585778  [25600/60000]
# loss: 1.767275  [32000/60000]
# loss: 1.823988  [38400/60000]
# loss: 1.740292  [44800/60000]
# loss: 1.898942  [51200/60000]
# loss: 1.987282  [57600/60000]
# Test Error:
#  Accuracy: 42.8%, Avg loss: 0.028005

# Epoch 8
# -------------------------------
# loss: 1.804497  [    0/60000]
# loss: 1.743471  [ 6400/60000]
# loss: 1.729417  [12800/60000]
# loss: 1.928848  [19200/60000]
# loss: 1.536891  [25600/60000]
# loss: 1.731442  [32000/60000]
# loss: 1.785927  [38400/60000]
# loss: 1.706183  [44800/60000]
# loss: 1.861235  [51200/60000]
# loss: 1.963670  [57600/60000]
# Test Error:
#  Accuracy: 43.6%, Avg loss: 0.027431

# Epoch 9
# -------------------------------
# loss: 1.765532  [    0/60000]
# loss: 1.708738  [ 6400/60000]
# loss: 1.687073  [12800/60000]
# loss: 1.881281  [19200/60000]
# loss: 1.480083  [25600/60000]
# loss: 1.696122  [32000/60000]
# loss: 1.722303  [38400/60000]
# loss: 1.652256  [44800/60000]
# loss: 1.803653  [51200/60000]
# loss: 1.897154  [57600/60000]
# Test Error:
#  Accuracy: 52.4%, Avg loss: 0.026307

# Epoch 10
# -------------------------------
# loss: 1.728517  [    0/60000]
# loss: 1.672117  [ 6400/60000]
# loss: 1.635958  [12800/60000]
# loss: 1.805463  [19200/60000]
# loss: 1.405441  [25600/60000]
# loss: 1.659478  [32000/60000]
# loss: 1.651421  [38400/60000]
# loss: 1.596837  [44800/60000]
# loss: 1.752495  [51200/60000]
# loss: 1.837687  [57600/60000]
# Test Error:
#  Accuracy: 53.1%, Avg loss: 0.025345

# Done!