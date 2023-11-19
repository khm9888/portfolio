# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict


plt.ion()   # 대화형 모드

# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)
print(torch.__version__)
# input()

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 1e-3
training_epochs = 20
batch_size = 200

import datetime

today = datetime.date.today()
m= today.month
d= today.day
today= f"{m:02d}{d}"

p=f"/home/con/bone_age/{today}"
if not os.path.isdir(p):
    os.mkdir(p)
# print(today)
with open(f"/home/con/bone_age/{today}/record.txt","a") as file:

    file.write("-------date--------"+"\n")
    file.write(str(today)+"\n")
    file.write("-------------------"+"\n")

class Bornage_Dataset(Dataset):
    def __init__(self, data_dir, transform, data_type='train'):
        # path to images
        path2data = data_dir

        csv_filename = data_type + '.csv'# train.csv
        # print(csv_filename)
        csv_dir="/home/con/data/boneage/csv"
        path2csvLabels = os.path.join(csv_dir, csv_filename)#data_dir ='data/boneage' ,csv_filename = train.csv
        # print("path2csvLabels",path2csvLabels)# path2csvLabels data/boneage/train.csv

        labels_df = pd.read_csv(path2csvLabels,index_col='id')

        ##############################################
        #재정의
        idx = labels_df.index
        self.id= list(labels_df.index)
        
        
        filenames=list(map(lambda x:x+"_pad_seg.png", (map(str,idx))))
    
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]#경로까지 추가하여 모두 각각 저장함.

        ##############################################


        self.labels = [labels_df.loc[int(filename[:-len("_pad_seg.png")])].values[0]//12 for filename in filenames]#str으로 찾는 건 맞으나, 
        # self.labels = [labels_df.loc[int(filename[:4])].values[2] for filename in filenames]#str으로 찾는 건 맞으나, 
        # except:
        # print(filenames)
        # 파일명으로 loc함수로 행을 찾고, 그 값의 0 번째니까.
        #0 - bone_age
        #1 - gender
        #2 - bone_age_year

        self.transform = transform
        self.idx = idx
        #data_transformer - 이미지를 수정함.사이즈 등


    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        # print(self.full_filenames[idx])
        image = Image.open(self.full_filenames[idx])
        # print(type(self))
        image = self.transform(image)
        return image, self.labels[idx],self.id[idx]

data_transformer = transforms.Compose([
        # transforms.RandomResizedCrop(512),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor()
        transforms.Resize((224,224)),
        transforms.Grayscale(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ])



data_dir ='data/segmented_hand_al'
dataset = Bornage_Dataset(data_dir, data_transformer, 'train_male')

# v=dataset[0]
# print(v)
# print(v[0].shape)#torch.Size([1, 28, 28])
# input()

dataset_len = len(dataset)
train_size , val_size = dataset_len//10*8,dataset_len//10
test_size= dataset_len - train_size - val_size
print("train_size , val_size, test_size")
print(train_size , val_size, test_size)
# trn_dataset, val_dataset,test_dataset = torch.utils.data.random_split(dataset, [10000, 1311,1300])
trn_dataset, val_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=4)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=4)

dataloaders = {"train":trn_loader,"val":val_loader}


dataset_sizes ={"train":train_size,"val": val_size}
########################################################################

def train_model(model, criterion, optimizer, scheduler, num_epochs=training_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in tqdm(range(num_epochs)):
        
        with open(f"/home/con/bone_age/{today}/record.txt","a") as file:
            file.write(f"epoch:{epoch}"+"\n")
        
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        save_data=defaultdict(np.array)
        save_data = {"id":np.array([]),"true":np.array([]),"predict":np.array([])}

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0
            running_revise_corrects = 0

            # 데이터를 반복
            for inputs, labels, ids in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                ids = ids.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                # print("pred[0:10]"+"\n")
                # print(torch.argmax(preds, 1)[0:10]+"\n")
                # print("labels.data[0:10]"+"\n")
                # print(labels.data[0:10]+"\n")
                preds_array=preds.cpu().numpy()
                labels_array=labels.data.cpu().numpy()
                ids_array=ids.data.cpu().numpy()
                if phase == 'val':
                    save_data["id"]=np.append(save_data["id"],ids_array)
                    save_data["true"]=np.append(save_data["true"],labels_array)
                    save_data["predict"]=np.append(save_data["predict"],preds_array)
                for i in range(len(preds)):
                    if labels_array[i]-1 <=preds_array[i] <= labels_array[i]+1:
                        running_revise_corrects+=1
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_revise_acc = running_revise_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(f'{phase} Loss: {epoch_loss:.4f} revise_acc: {epoch_revise_acc:.4f}')

            with open(f"/home/con/bone_age/{today}/record.txt","a") as file:
                file.write('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)+"\n")
                file.write(f'{phase} Loss: {epoch_loss:.4f} revise_acc: {epoch_revise_acc:.4f}'+'\n')

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                with open(f"/home/con/bone_age/{today}/record.txt","a") as file:
                    file.write(f"epoch:{epoch}"+"\n")
                    file.write("outputs"+"\n")
                    file.write("torch.argmax(outputs, 1)"+"\n")
                    file.write(f"{torch.argmax(outputs, 1)}"+"\n")
                    file.write("labels.data"+"\n")
                    file.write(f"{labels.data}"+"\n")
                data_f = pd.DataFrame(save_data)
                data_f.set_index("id")
                data_f=data_f.astype(int)
                PATH_csv = f'/home/con/bone_age/csv/{today}_epoch_{epoch}.csv'
                data_f.to_csv(PATH_csv)
                PATH = f'/home/con/bone_age/checkpoint/bone_age_cnn_{today}_epoch_{epoch}.pth'
                torch.save(model.state_dict(), PATH)
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open(f"/home/con/bone_age/{today}/record.txt","a") as file:
        file.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+"\n")
        file.write('Best val Acc: {:4f}'.format(best_acc)+"\n")
    

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

model_ft = models.resnet50(pretrained=True)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# 7 에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)

PATH = f'/home/con/bone_age/checkpoint/bone_age_cnn_{today}_total_epoch_{training_epochs}.pth'
print('test')
torch.save(model_ft.state_dict(), PATH)