# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data as data_utils

import torch
import torch.nn as nn
import torch.nn.fucntional as F
import torch.optim as optim
import torch.utils.data as data_utils

class Model(nn.module):
    
    def __init__(self,X_dim,Y_dim):
        super(Model,self).__init__()
        layer1 = nn.Linear(X_dim,128)
        activaition1 = nn.ReLU()
        layer2 = nn.Linver(128,Y_dim)
        self.module = nn.Sequential(
            layer1,
            activaition1,
            layer2
            )
        
    def forward(self, x):
        out = self.module(x)
        result  = F.softmax(out,dim=1)
        return result
    
model = Model()
