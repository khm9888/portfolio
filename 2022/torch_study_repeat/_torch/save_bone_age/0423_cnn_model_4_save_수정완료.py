import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchsummary import summary
from tqdm import tqdm 

import os

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
training_epochs = 10
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
        #data_transformer - 이미지를 수정함.사이즈 등


    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        # print(self.full_filenames[idx])
        image = Image.open(self.full_filenames[idx])
        # image /=255
        # print(type(self))
        image = self.transform(image)
        return image, self.labels[idx]

data_transformer = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ])

data_dir ='data/segmented_hand_al'
dataset = Bornage_Dataset(data_dir, data_transformer, 'train')

# input()

dataset_len = len(dataset)
train_size , val_size = dataset_len//10*8,dataset_len//10
test_size= dataset_len - train_size - val_size
trn_dataset, val_dataset,test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        # L1 ImgIn shape=(?, 256, 256, 1)#(1514, 2044)
        #    Conv     -> (?, 256, 256, 32)
        #    Pool     -> (?, 128, 128, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 128, 128, 32)
        #    Conv      ->(?, 128, 128, 64)
        #    Pool      ->(?, 64, 64, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L3 ImgIn shape=(?, 64, 64, 64)
        #    Conv      ->(?, 64, 64, 128)
        #    Pool      ->(?, 33, 33, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        # self.layer3_2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(400, 800, kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # self.fc1 = torch.nn.Linear(33 * 33 * 128, 625, bias=True)
        self.fc1 = torch.nn.Linear(33 * 33 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # self.fc2 = torch.nn.Linear(625, 20, bias=True)
        self.fc2 = torch.nn.Linear(625, 20, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        # # L5 Final FC 625 inputs -> 10 outputs
        # self.fc3 = torch.nn.Linear(625, 20, bias=True)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        out = self.layer1(x)
        # print("out",out.shape)
        out = self.layer2(out)
        # print("out",out.shape)
        out = self.layer3(out)
        # print("out",out.shape)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        # print("out",out.shape)
        out = self.layer4(out)
        out = self.fc2(out)
        # out = self.fc3(out)
        return out

cnn = CNN().to(device)
print(cnn)
# input()

criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

total_batch = len(trn_loader)

print('배치사이즈 : {}'.format(batch_size))
print('총 배치의 수 : {}'.format(total_batch))

# input()

for epoch in range(training_epochs):
    avg_cost = 0
    # total_cost = 0
    for (X, Y) in tqdm(trn_loader): # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = cnn(X)
        cost = criterion(hypothesis, Y)
        correct_hypothesis = torch.argmax(hypothesis, 1) == Y
        accuracy = correct_hypothesis.float().mean()
        # print(torch.argmax(hypothesis,1))
        # print(Y)
        # print(correct_hypothesis)
        # print("---")
        # print(f"ACC:{accuracy}")
        # print("---")

        # input()
        cost.backward()
        optimizer.step()

        # total_cost += cost
        avg_cost += cost / total_batch #batch size로 나눠야 하지 않나..
        # avg_cost += cost / batch_size*(i+1) #batch size로 나눠야 하지 않나..
        # print("total_cost:",total_cost)
        print("avg_cost:",avg_cost)
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    if epoch%5==0 and epoch>0:
        PATH = f'/home/con/bone_age/checkpoint/bone_age_cnn_{today}_{epoch}.pth'
        torch.save(cnn.state_dict(), PATH)

PATH = f'/home/con/bone_age/checkpoint/bone_age_cnn_{today}_{training_epochs}.pth'

torch.save(cnn.state_dict(), PATH)

cnn = CNN().to(device)
cnn.load_state_dict(torch.load(PATH))
    # 학습을 진행하지 않을 것이므로 torch.no_grad()
correct = 0
total = 0
with torch.no_grad():
    for j, val in tqdm(enumerate(test_loader)):
        X_test,Y_test=val
        # X_test = test_dataset.test_data.view(len(test_dataset), 1, 512, 512).float().to(device)
        # Y_test = test_dataset.labels.to(device)

        X_test=X_test.to(device)
        Y_test=Y_test.to(device)
        prediction = cnn(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        correct+=accuracy.item()
        with open(f"/home/con/bone_age/{today}/record.txt","a") as file:

            file.write("-------date--------"+"\n")
            file.write(f"true:{Y}\n")
            file.write(f"hypothesis:{torch.argmax(prediction,1)}\n")
            # file.write(correct_hypothesis)
            file.write(f"accuracy:{accuracy}")
            file.write("-------------------"+"\n")

        print(j,'Accuracy:', correct/(j+1)*100,"%")
    print('Accuracy:', correct/len(test_loader),"%")

#mnist
#Accuracy: 0.9815999865531921