# 02. 자동 미분(Autograd)

import torch

w = torch.tensor(2.0,requires_grad = True)

y = w**2
z = 2*y +5

z.backward()

print(f"수식을 w로 미분한 값 : {w.grad}")