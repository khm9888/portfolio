import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

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
    

model = NeuralNetwork().to(device)
print(model)

# NeuralNetwork(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#     (5): ReLU()
#   )
# )

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

#Predicted class: tensor([4], device='cuda:0')

input_image = torch.rand(3,28,28)
print(input_image.size())

# torch.Size([3, 28, 28])

#######################################################################################################

# nn.Flatten
# nn.Flatten 계층을 초기화하여 각 28x28의 
# 2D 이미지를 784 픽셀 값을 갖는 연속된 배열로 변환합니다. (dim=0의 미니배치 차원은 유지됩니다.)

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# torch.Size([3, 784])

#######################################################################################################

# nn.Linear
# 선형 계층 은 저장된 가중치(weight)와 편향(bias)을 
# 사용하여 입력에 선형 변환(linear transformation)을 적용하는 모듈입니다.

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# torch.Size([3, 20])

#######################################################################################################

# nn.ReLU
# 비선형 활성화(activation)는 모델의 입력과 출력 사이에 복잡한 관계(mapping)를 만듭니다. 
# 비선형 활성화는 선형 변환 후에 적용되어 비선형성(nonlinearity) 을 도입하고, 신경망이 다양한 현상을 학습할 수 있도록 돕습니다.

# 이 모델에서는 nn.ReLU 를 선형 계층들 사이에 사용하지만, 
# 모델을 만들 때는 비선형성을 가진 다른 활성화를 도입할 수도 있습니다.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# Before ReLU: tensor([[ 0.4410, -0.0452, -0.2027,  0.1747, -0.2381, -0.4092, -0.2615,  0.0050,
#          -0.5300, -0.0780, -0.3229,  0.0871, -0.8442, -0.4860,  0.2393,  0.0142,
#          -0.2075,  0.1637,  0.6947,  0.6777],
#         [-0.1662,  0.0146, -0.0442,  0.4260, -0.3107, -0.2911, -0.2069,  0.4032,
#          -0.3424, -0.0808, -0.5424, -0.0741, -0.7567, -0.1303,  0.1675, -0.0421,
#          -0.0127,  0.0189,  0.2049,  0.6353],
#         [ 0.0528, -0.0792,  0.1829,  0.0642, -0.3702, -0.7138, -0.1625,  0.1354,
#          -0.5442,  0.1852, -0.5636, -0.2036, -0.4781, -0.4126,  0.4269,  0.1393,
#          -0.1686,  0.1084,  0.5630,  0.3421]], grad_fn=<AddmmBackward>)


# After ReLU: tensor([[0.4410, 0.0000, 0.0000, 0.1747, 0.0000, 0.0000, 0.0000, 0.0050, 0.0000,
#          0.0000, 0.0000, 0.0871, 0.0000, 0.0000, 0.2393, 0.0142, 0.0000, 0.1637,
#          0.6947, 0.6777],
#         [0.0000, 0.0146, 0.0000, 0.4260, 0.0000, 0.0000, 0.0000, 0.4032, 0.0000,
#          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1675, 0.0000, 0.0000, 0.0189,
#          0.2049, 0.6353],
#         [0.0528, 0.0000, 0.1829, 0.0642, 0.0000, 0.0000, 0.0000, 0.1354, 0.0000,
#          0.1852, 0.0000, 0.0000, 0.0000, 0.0000, 0.4269, 0.1393, 0.0000, 0.1084,
#          0.5630, 0.3421]], grad_fn=<ReluBackward0>)

#######################################################################################################

# nn.Sequential
# nn.Sequential 은 순서를 갖는 모듈의 컨테이너입니다. 데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달됩니다. 
# 순차 컨테이너(sequential container)를 사용하여 아래의 seq_modules 와 같은 신경망을 빠르게 만들 수 있습니다.

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

#######################################################################################################

# nn.Softmax
# 신경망의 마지막 선형 계층은 nn.Softmax 모듈에 전달될 ([-infty, infty] 범위의 원시 값(raw value)인) logits 를 반환합니다. 
# logits는 모델의 각 분류(class)에 대한 예측 확률을 나타내도록 [0, 1] 범위로 비례하여 조정(scale)됩니다. 
# dim 매개변수는 값의 합이 1이 되는 차원을 나타냅니다.

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

#######################################################################################################

# 모델 매개변수

# 신경망 내부의 많은 계층들은 매개변수화(parameterize) 됩니다. 
# 즉, 학습 중에 최적화되는 가중치와 편향과 연관지어집니다. 
# nn.Module 을 상속하면 모델 객체 내부의 모든 필드들이 자동으로 추적(track)되며, 
# 모델의 parameters() 및 named_parameters() 메소드로 모든 매개변수에 접근할 수 있게 됩니다.

print("Model structure: ", model, "\n\n")

# Model structure:  NeuralNetwork(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (linear_relu_stack): Sequential(
#     (0): Linear(in_features=784, out_features=512, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=512, out_features=512, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=512, out_features=10, bias=True)
#     (5): ReLU()
#   )
# )

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    
# Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-2.2684e-02, -1.1845e-02, -2.7930e-02,  ..., -4.1258e-03,
#          -4.4581e-03, -2.2919e-02],
#         [ 1.7550e-02,  7.0121e-05,  1.7727e-02,  ..., -1.5328e-02,
#          -3.1287e-02,  2.3250e-02]], device='cuda:0', grad_fn=<SliceBackward>)

# Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0057,  0.0113], device='cuda:0', grad_fn=<SliceBackward>)

# Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0088, -0.0062,  0.0156,  ..., -0.0414, -0.0116, -0.0196],
#         [ 0.0016, -0.0084,  0.0422,  ...,  0.0396, -0.0124, -0.0332]],
#        device='cuda:0', grad_fn=<SliceBackward>)

# Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0040, -0.0172], device='cuda:0', grad_fn=<SliceBackward>)

# Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0103, -0.0394, -0.0402,  ..., -0.0204,  0.0190, -0.0406],
#         [ 0.0012,  0.0098,  0.0006,  ..., -0.0170,  0.0290, -0.0116]],
#        device='cuda:0', grad_fn=<SliceBackward>)

# Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([ 0.0424, -0.0321], device='cuda:0', grad_fn=<SliceBackward>)

#######################################################################################################