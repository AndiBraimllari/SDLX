import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import Cityscapes
import numpy as np
from matplotlib import pyplot as plt
from skimage import color
from skimage import io
from ttictoc import tic, toc
from scipy import misc
from UNet import UNet


# ========== definitions start here ==========
def get_training_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader


def train_net(my_net, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(my_net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # inputImage = torch.from_numpy(color.rgb2gray(inputs[0]))

            # print(torch.reshape(inputs[0][0], (1, 1, 32, 32)).shape)

            mySlice = color.rgb2gray(io.imread('rsz_572pls.jpg')).astype(np.float32)

            mySlice = torch.from_numpy(mySlice)
            mySlice = torch.reshape(mySlice, (1, 1, mySlice.shape[0], mySlice.shape[1]))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = my_net(mySlice)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


# ========== playground start here ==========
def demo():
    unet = UNet()

    train_net(unet, get_training_data())


demo()

# # TODO dataset for training Cityscapes?
# dataset = Cityscapes('./data/cityscapes', split='train', mode='fine', target_type='semantic')
# img, smnt = dataset[0]

# ========== deprecated stuff ==========
# PATH = './cifar_net.pth'
# torch.save(unet.state_dict(), PATH)
#
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# net = UNet()
# net.load_state_dict(torch.load(PATH))
#
# outputs = net(images)
#
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# ========== PyTorch demo code ==========

# model = torch.hub.load('pytorch/vision:v0.9.0', 'fcn_resnet101', pretrained=True)
# # print(model.eval(color.rgb2gray(io.imread('slice_511.jpg'))))
#
# # Download an example image from the pytorch website
# import urllib
#
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try:
#     urllib.URLopener().retrieve(url, filename)
# except:
#     urllib.request.urlretrieve(url, filename)
#
# filename = 'slice_511.jpg'
#
# # sample execution (requires torchvision)
# from PIL import Image
# from torchvision import transforms
#
# input_image = Image.open(filename)
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
#
# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     print('here boi')
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')
#
# with torch.no_grad():
#     output = model(input_batch)['out'][0]
# output_predictions = output.argmax(0)
#
# # create a color pallette, selecting a color for each class
# palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
# colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
# colors = (colors % 255).numpy().astype("uint8")
#
# # plot the semantic segmentation predictions of 21 classes in each color
# r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
# r.putpalette(colors)
#
# import matplotlib.pyplot as plt
# plt.imshow(r)
# plt.show()
