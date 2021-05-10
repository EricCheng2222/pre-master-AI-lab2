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
     transforms.Resize(144),
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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.bn1   = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU(0.01)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(128, 64, 3)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.01)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 64, 3)
        self.bn3   = nn.BatchNorm2d(64)
        self.relu3 = nn.LeakyReLU(0.01)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 64, 3)
        self.bn4   = nn.BatchNorm2d(64)
        self.relu4 = nn.LeakyReLU(0.01)
        self.pool4 = nn.MaxPool2d(2)

        #self.conv5 = nn.Conv2d(64, 25, 3)
        #self.bn5   = nn.BatchNorm2d(25)
        #self.relu5 = nn.LeakyReLU(0.01)
        #self.pool5 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(3136, 625)
        self.relu3 = nn.LeakyReLU(0.01)
        self.fc2 = nn.Linear(625, 625)
        self.relu4 = nn.LeakyReLU(0.01)
        self.fc3 = nn.Linear(625, 11)
        self.relu5 = nn.LeakyReLU(0.01)


    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.relu3(y)
        y = self.pool3(y)
        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu4(y)
        y = self.pool4(y)
        #y = self.conv5(y)
        #y = self.bn5(y)
        #y = self.relu5(y)
        #y = self.pool5(y)
        #print(y.size())
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y 

def main(argv, argc):
    import os.path
    from os import path
    EPOCH = 50
    PATH = './cifar_net.pth'
    if False:
        print("Training")
        net = Net()
        net.to(device)
        if(argc>=2 and argv[1]=="--usepth"):
            print("use old network")
            net.load_state_dict(torch.load(PATH))
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
        for epoch in range(EPOCH):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = net(inputs)
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
        torch.save(net.state_dict(), PATH)
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(1)))
    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(1)))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    

if __name__ == "__main__":
    import sys
    main(sys.argv, len(sys.argv))
