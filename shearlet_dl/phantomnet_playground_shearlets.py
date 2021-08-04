import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from matplotlib import pyplot as plt

from shearlet_dl.PhantomNet import PhantomNet

# this script uses the PhantomNet model as an auto-encoder to train shearlet coefficients (e.g. given a slice (512, 512)
# the shearlet transform will generate an object (the shearlet coefficients) of shape (512, 512, 61). The input and
# output of this model are of the same shape

sh_dir = '../sh_low_dosage_jpgs'

num_epochs = 50
batch_size = 1
learning_rate = 5e-5


class ShearletedSlicesDataset(Dataset):
    """Shearleted images."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        paths = [root_dir + '/' + sh_slice for sh_slice in os.listdir(sh_dir)]
        self.shearleted_slices = []
        i = 0
        for sh_slice in paths:
            # if i == 10:  # take 10 for now, current setup can't handle more
            #     break
            self.shearleted_slices.append(np.load(sh_slice))
            i += 1

        self.shearleted_slices = np.array(self.shearleted_slices)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.shearleted_slices)

    def __getitem__(self, idx):  # ideally this is loaded per-request and not all objects should be kept in memory
        sample = self.shearleted_slices[idx]

        sample = torch.Tensor(sample)
        return sample


# note that, torchvision.transforms.ToTensor converts a PIL Image or numpy.ndarray (H x W x C) to a torch.FloatTensor of
# shape (C x H x W).
shearleted_dataset = ShearletedSlicesDataset(sh_dir)
dataloader = DataLoader(shearleted_dataset, batch_size=batch_size, shuffle=True)

# take this from data
oversamplingFactor = 61

model = PhantomNet(oversamplingFactor)  # .cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_hst = []
for epoch in range(num_epochs):
    for data in dataloader:
        img = data
        img = Variable(img).type(torch.FloatTensor)  # .cuda()

        output = model(img)
        loss = criterion(output, img)

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
test_sample = torch.reshape(shearleted_dataset.__getitem__(0), (1, 61, 512, 512))  # .cuda()

pred = model(test_sample)

plt.imshow(test_sample[0][8].cpu().detach().numpy())
plt.title('test sample 8')
plt.show()

plt.imshow(test_sample[0][6].cpu().detach().numpy())
plt.title('test sample 6')
plt.show()

plt.imshow(pred[0][8].cpu().detach().numpy())
plt.title('pred 8')
plt.show()

plt.imshow(pred[0][6].cpu().detach().numpy())
plt.title('pred 6')
plt.show()
