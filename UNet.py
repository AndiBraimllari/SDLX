import torch.nn as nn
import torch.nn.functional as F


# TODO pooling layers do not have parameters, double check
# NB there are 4 crops
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, (3, 3))
        self.conv2 = nn.Conv2d(64, 64, (3, 3))

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.conv4 = nn.Conv2d(128, 128, (3, 3))

        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(128, 256, (3, 3))
        self.conv6 = nn.Conv2d(256, 256, (3, 3))

        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(256, 512, (3, 3))
        self.conv8 = nn.Conv2d(512, 512, (3, 3))

        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv9 = nn.Conv2d(512, 1024, (3, 3))
        self.conv10 = nn.Conv2d(1024, 1024, (3, 3))

        self.transpConv1 = nn.ConvTranspose2d(1024, 512, (2, 2))

        self.conv11 = nn.Conv2d(1024, 512, (3, 3))
        self.conv12 = nn.Conv2d(512, 512, (3, 3))

        self.transpConv2 = nn.ConvTranspose2d(512, 256, (2, 2))

        self.conv13 = nn.Conv2d(512, 256, (3, 3))
        self.conv14 = nn.Conv2d(256, 128, (3, 3))

        self.transpConv3 = nn.ConvTranspose2d(256, 128, (2, 2))

        self.conv15 = nn.Conv2d(256, 128, (3, 3))
        self.conv16 = nn.Conv2d(128, 128, (3, 3))

        self.transpConv4 = nn.ConvTranspose2d(128, 64, (2, 2))

        self.conv17 = nn.Conv2d(128, 64, (3, 3))
        self.conv18 = nn.Conv2d(64, 64, (3, 3))

        self.conv1x1 = nn.Conv2d(64, 2, (1, 1))  # TODO does this need to be 2, or can it be 1? what do we need

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # TODO first crop here

        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # TODO second crop here

        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        # TODO third crop here

        x = self.pool3(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        # TODO fourth crop here

        x = self.pool4(x)

        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        x = self.transpConv1(x)

        x = F.relu(self.conv11(x))  # TODO x conc fourth crop
        x = F.relu(self.conv12(x))

        x = self.transpConv2(x)

        x = F.relu(self.conv13(x))  # TODO x conc third crop
        x = F.relu(self.conv14(x))

        x = self.transpConv3(x)

        x = F.relu(self.conv15(x))  # TODO x conc second crop
        x = F.relu(self.conv16(x))

        x = self.transpConv4(x)

        x = F.relu(self.conv17(x))  # TODO x conc first crop
        x = F.relu(self.conv18(x))

        x = self.conv1x1(x)

        return x
