import numpy as np


t = np.array([float(i+j) for i in range(0,10,3) for j in range(1,4)])

t = np.reshape(t,(-1,3))

print(t)

import torch

t = torch.FloatTensor(t)

print(t)

print("t.dim()",t.dim())
print("t.size()",t.size())


#class

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self,num):
        self.result+=num
        return self.result

c1 = Calculator()

c1.add(3)

print(c1.result)