import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path
import sys

# add the parent of the parent of this file to the system path
sys.path.append(str(Path(os.path.abspath(Path(__file__).parent)).parent))

from shearlet_dl.PhantomNet import PhantomNet


class ShearletedSlicesDataset(Dataset):
    """Class representing a dataset of shearleted images. Items are fetched on the assumption that the files
    contained in the X/Y_dir are of the format NUMBER.npy, for NUMBER from 0 to length - 1."""

    def __init__(self, X_dir, Y_dir):
        self.X_dir = X_dir
        self.Y_dir = Y_dir

    def __len__(self):  # TODO
        return len(os.listdir(self.X_dir))

    def __getitem__(self, idx):
        X = np.load(self.X_dir + '/' + str(idx) + '.npy')
        Y = np.load(self.Y_dir + '/' + str(idx) + '.npy')

        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        return X, Y


def train_phantomnet(X_dir, Y_dir, multip_gpus=True, num_epochs=50, batch_size=32, learning_rate=5e-5,
                     weight_decay=1e-5, criterion=nn.MSELoss(), visualize_loss=True, visualize_pred=True,
                     save_model=True):
    sample_shape = np.load(X_dir + '/' + os.listdir(X_dir)[0]).shape
    oversampling_factor = sample_shape[0]

    shearleted_dataset = ShearletedSlicesDataset(X_dir, Y_dir)
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

    if visualize_loss:
        plt.plot(loss_hst)
        plt.show()

    # make reshape input dynamic
    test_sample = torch.reshape(shearleted_dataset.__getitem__(0)[1],
                                (1, oversampling_factor, sample_shape[1], sample_shape[2])).cuda()

    pred = pnModel(test_sample)

    if save_model:
        model_name = pnModel.__class__.__name__ + '_epochs_' + str(num_epochs) + '_lr_' + str(
            learning_rate) + '_datetime_' + str(datetime.now()) + '_model'
        model_name = model_name.replace(' ', '_')
        print('Model name to be saved is: ' + model_name)
        torch.save(pnModel.state_dict(), model_name)

    if visualize_pred:
        i = 0  # randomly choose this
        j = 8  # randomly choose this

        plt.imshow(test_sample[i][j].cpu().detach().numpy())
        plt.title('test sample ' + str(i) + ' at shearlet coeff. ' + str(j))
        plt.show()

        plt.imshow(pred[i][j].cpu().detach().numpy())
        plt.title('pred. ' + str(i) + ' at shearlet coeff. ' + str(j))
        plt.show()
