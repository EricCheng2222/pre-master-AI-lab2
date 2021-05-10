import torch
import torchvision
import torchvision.transforms as transforms
import ssl
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
ssl._create_default_https_context = ssl._create_unverified_context

transform_set = [
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.RandomAffine(degrees=(-15,15), translate=(0, 0.8), scale=(0.5, 1.0), shear=(0,0), fillcolor=(0,0,0)),
     transforms.RandomGrayscale(p=0.5),
     transforms.ColorJitter((0,3), (0,3), (0, 3), (0.0, 0.5))
]


transform = transforms.Compose(
    [
     transforms.Resize(224),
     transforms.RandomChoice(transform_set),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])
# Parameters
params = {'batch_size': 64,
          'num_workers': 8}

test_params = {'batch_size': 64,
               'shuffle' : True,
               'num_workers': 8}


class_count  = get_count_in_folder("../food11re/food11re/skewed_training")
temp_weight = list(map(lambda x: [1.0/x]*x, class_count))
class_weight = [i for sublist in temp_weight for i in sublist]
print(len(class_weight))
string_weight = [str(num) for num in class_weight]
print_string = "\n".join(string_weight)
f = open("log", "w+")
f.write(print_string)
f.close()
my_sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weight, len(class_weight))


#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
trainset = torchvision.datasets.ImageFolder(root="../food11re/food11re/skewed_training", transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, **params, sampler=my_sampler)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                       download=True, transform=transform)

testset = torchvision.datasets.ImageFolder(root="../food11re/food11re/evaluation", transform=transform)
testloader = torch.utils.data.DataLoader(testset, **test_params)






import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)


def main(argv, argc):
    import os.path
    from os import path
    EPOCH = 25
    PATH = './cifar_net.pth'
    if True:
        print("Training")
        num_feats = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_feats, 11)
        resetnet18 = resnet18.to(device)
        if(argc>=2 and argv[1]=="--usepth"):
            print("use old network")
            resnet18.load_state_dict(torch.load(PATH))
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(resnet18.parameters(), lr=0.0005, momentum=0.95)
        for epoch in range(EPOCH):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = resnet18(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    
                # print statistics
                running_loss += loss.item()
                if i % EPOCH == (EPOCH-1):    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / EPOCH))
                    running_loss = 0.0
        print('Finished Training')
        torch.save(resnet18.state_dict(), PATH)
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device) 
    # print images
    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(1)))
    resnet_my = models.resnet18(pretrained=False, num_classes=11)
    resnet_my = resnet_my.to(device)
    resnet_my.load_state_dict(torch.load(PATH))
    outputs = resnet_my(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(1)))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device) 
            outputs = resnet_my(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    

if __name__ == "__main__":
    import sys
    main(sys.argv, len(sys.argv))
