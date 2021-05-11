import os
import torch
import torchvision

def dataset(root):
    train_path = root + '/train/'
    test_path = root + '/test/'
    val_path = root + '/val/'
    
     
    transform_train = transforms.Compose([
        transforms.Resize((30,30),),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    transform_test = transforms.Compose([
    transforms.Resize((30,30),),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    
    trainset = torchvision.datasets.ImageFolder(root= train_path , transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=80, shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.ImageFolder(root= train_path , transform=transform_test)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=False, num_workers=4)
    
    valset = torchvision.datasets.ImageFolder(root= val_path , transform=transform_train)
    valloader = torch.utils.data.DataLoader(trainset, batch_size=30, shuffle=True, num_workers=4)
    
    return trainloader, testloader, valloader
    
           
  