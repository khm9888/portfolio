import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init

import torch.utils.data.DataLodaer as DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

batch_size = 4

trainset = datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

testloader = DataLoader(testset, batch_size = batch_size, shuffle=False, num_workers=2)

classes =list()

classes.append("plane")
classes.append("car")
classes.append("bird")
classes.append("cat")
classes.append("deer")
classes.append("dog")
classes.append("frog")
classes.append("horse")
classes.append("ship")
classes.append("truck")

classes = tuple(classes)
