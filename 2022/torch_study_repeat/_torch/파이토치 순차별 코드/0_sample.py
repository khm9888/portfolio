import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import torchaudio

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


#데이터입력
#모델구성
# +@ 기록 하기
location_now = "/home/con/tutorial_torch/"

with open(f"{location_now}/record.txt","a") as file:

    file.write("------date-----------\n")
    file.write(str(datetime.datetime.today())+"\n")
    file.write("------date-----------\n")
#훈련
#평가 및 예측