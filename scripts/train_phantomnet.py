import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from matplotlib import pyplot as plt

from shearlet_dl.PhantomNet import PhantomNet


class ShearletedSlicesDataset(Dataset):
    """Class representing a dataset of shearleted images. Items are fetched on the assumption that the files
    contained in the X/Y_dir are of the format NUMBER.npy for NUMBER from 0 to length - 1."""

    def __init__(self, X_dir, Y_dir):
        self.X_dir = X_dir
        self.Y_dir = Y_dir

    def __len__(self):  # TODO
        return len(os.listdir(self.X_dir))

    def __getitem__(self, idx):  # ideally this is loaded per-request and not all objects should be kept in memory
        X = np.load(self.X_dir + '/' + str(idx) + '.npy')
        Y = np.load(self.Y_dir + '/' + str(idx) + '.npy')

        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        return X, Y


def train_phantomnet(shearlets_dir, multip_gpus=True, num_epochs=50, batch_size=32, learning_rate=5e-5,
                     weight_decay=1e-5, criterion=nn.MSELoss()):
    sample_shape = np.load(shearlets_dir + '/' + os.listdir(shearlets_dir)[0])
    oversampling_factor = sample_shape.shape[0]

    shearleted_dataset = ShearletedSlicesDataset(shearlets_dir, shearlets_dir)  # TODO auto-encoder for now
    dataloader = DataLoader(shearleted_dataset, batch_size=batch_size, shuffle=True)

    pnModel = PhantomNet(oversampling_factor)

    if multip_gpus:
        pnModel = nn.DataParallel(pnModel)

    pnModel = pnModel.cuda()

    optimizer = torch.optim.Adam(pnModel.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_hst = []
    for epoch in range(num_epochs):
        loss = None
        for X, Y in dataloader:
            X = Variable(X).cuda()
            Y = Variable(Y).cuda()

            output = pnModel(X)
            loss = criterion(output, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_hst.append(loss.data.cpu().detach().numpy())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))

    print(loss_hst)
    plt.plot(loss_hst)
    plt.show()

    # below here, some predictions are made

    # make reshape input dynamic
    test_sample = torch.reshape(shearleted_dataset.__getitem__(0),
                                (1, oversampling_factor, sample_shape[0], sample_shape[1])).cuda()
    # test_sample = torch.reshape(shearleted_dataset.__getitem__(0), (1, 61, 512, 512))  # .cuda()

    pred = pnModel(test_sample)

    plt.imshow(test_sample[0][8].cpu().detach().numpy())
    plt.title('test sample 8')
    plt.show()

    plt.imshow(pred[0][8].cpu().detach().numpy())
    plt.title('pred 8')
    plt.show()

    plt.imshow(test_sample[0][6].cpu().detach().numpy())
    plt.title('test sample 6')
    plt.show()

    plt.imshow(pred[0][6].cpu().detach().numpy())
    plt.title('pred 6')
    plt.show()
