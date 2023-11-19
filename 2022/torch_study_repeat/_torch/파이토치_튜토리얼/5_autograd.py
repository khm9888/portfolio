#https://tutorials.pytorch.kr/beginner/basics/autogradqs_tutorial.html

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('Gradient function for z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

# Gradient function for z = <AddBackward0 object at 0x7fefdce30e50>
# Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward object at 0x7fefdce30e50>

loss.backward()
print(w.grad)
print(b.grad)

# tensor([[0.2058, 0.0085, 0.1949],
#         [0.2058, 0.0085, 0.1949],
#         [0.2058, 0.0085, 0.1949],
#         [0.2058, 0.0085, 0.1949],
#         [0.2058, 0.0085, 0.1949]])
# tensor([0.2058, 0.0085, 0.1949])


##############################################################################

# 변화도 추적 멈추기

# 기본적으로, requires_grad=True인 모든 텐서들은 연산 기록을 추적하고 변화도 계산을 지원합니다. 그러나 모델을 학습한 뒤 입력 데이터를 단순히 적용하기만 하는 경우와 같이 순전파 연산만 필요한 경우에는, 이러한 추적이나 지원이 필요없을 수 있습니다. 
# 연산 코드를 torch.no_grad() 블록으로 둘러싸서 연산 추적을 멈출 수 있습니다:


z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# True
# False

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# False


####################################################################################


# 선택적으로 읽기(Optional Reading): 텐서 변화도와 야코비안 곱 (Jacobian Product)
# 대부분의 경우, 스칼라 손실 함수를 가지고 일부 매개변수와 관련한 변화도를 계산해야 합니다. 
# 그러나 출력 함수가 임의의 텐서인 경우가 있습니다. 
# 이럴 때, PyTorch는 실제 변화도가 아닌 야코비안 곱(Jacobian product)을 계산합니다.

# x⃗ =⟨x1,…,xn⟩이고, y⃗ =⟨y1,…,ym⟩일 때 벡터 함수 y⃗ =f(x⃗ )에서 x⃗ 에 대한 y⃗  의 변화도는 
# 야코비안 행렬(Jacobian matrix)로 주어집니다:

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)


# First call
#  tensor([[4., 2., 2., 2., 2.],
#         [2., 4., 2., 2., 2.],
#         [2., 2., 4., 2., 2.],
#         [2., 2., 2., 4., 2.],
#         [2., 2., 2., 2., 4.]])

# Second call
#  tensor([[8., 4., 4., 4., 4.],
#         [4., 8., 4., 4., 4.],
#         [4., 4., 8., 4., 4.],
#         [4., 4., 4., 8., 4.],
#         [4., 4., 4., 4., 8.]])

# Call after zeroing gradients
#  tensor([[4., 2., 2., 2., 2.],
#         [2., 4., 2., 2., 2.],
#         [2., 2., 4., 2., 2.],
#         [2., 2., 2., 4., 2.],
#         [2., 2., 2., 2., 4.]])