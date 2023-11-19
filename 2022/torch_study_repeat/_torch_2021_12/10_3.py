# %% [markdown]
# # 파이토치 사전훈련 모델 사용
# - `torchvision`의 `models`를 활용하여 전이학습
# 
# - https://pytorch.org/docs/stable/torchvision/models.html
# 
# - 코드 출처: https://tutorials.pytorch.kr/beginner/transfer_learning_tutorial.html

# %% [markdown]
# ## modules import

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-white")

import time
import os
import copy

# %% [markdown]
# ## GPU 설정

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
print(device)

# %% [markdown]
# ## 데이터 로드 및 확인

# %%
!wget https://download.pytorch.org/tutorial/hymenoptera_data.zip
!unzip hymenoptera_data -d .

# %% [markdown]
# ## 전처리 설정

# %%
data_transforms = {"train":transforms.Compose([transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485,0.456,0.406),[0.229,0,224,0.225])]),
                   "val": transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485,0.456,0.406),[0.229,0,224,0.225])])
                       }

# %%
data_dir = "./hymenoptera_data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x),
                                          data_transforms[x]) 
                  for x in ["train","val"]}

# %%


# %%


# %% [markdown]
# ## 사전훈련된 모델 로드

# %%


# %% [markdown]
# - ImageNet 의 데이터를 훈련한 모델
#   - `num_class`가 1000

# %%


# %% [markdown]
# - 

# %% [markdown]
# - 데이터의 클래스 수에 맞게 `out_features` 변경

# %%


# %% [markdown]
# ## 손실함수와 옵티마이저

# %%


# %% [markdown]
# -  7 에폭마다 0.1씩 학습율 감소

# %%


# %% [markdown]
# ## 모델 학습

# %% [markdown]
# - 학습 함수 정의

# %%


# %%


# %% [markdown]
# ## 결과 시각화

# %% [markdown]
# - 시각화 함수 정의

# %%


# %%


# %%


# %% [markdown]
# ## 고정된 특징 추출기로써의 합성곱 신경망
# - 마지막 계층을 제외한 신경망의 모든 부분을 고정
# 
# - `requires_grad == False` 로 설정하여 매개변수를 고정하여 backward() 중에 경사도가 계산되지 않도록 한다.
# 
# 

# %%


# %%


# %%



