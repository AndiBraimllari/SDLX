import torch
import torch.nn as nn
import torch.nn.functional as F


# this file contains the PhantomNet architecture

# note that not every parameter here is required to specify as some are the default, but I added them for readability

class TDB4(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=(3, 3), padding='same',
                               stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=1 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=2 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=3 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))

    def forward(self, x):
        x0 = x
        x1 = F.tanh(self.conv1(x0))
        x2 = F.tanh(self.conv2(torch.cat((x0, x1), 1)))
        x3 = F.tanh(self.conv3(torch.cat((x0, x1, x2), 1)))
        x4 = F.tanh(self.conv4(torch.cat((x0, x1, x2, x3), 1)))

        return torch.cat((x1, x2, x3, x4), 1)


class TDB8(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=(3, 3), padding='same',
                               stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=1 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=2 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=3 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=4 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=5 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=6 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))
        self.conv8 = nn.Conv2d(in_channels=7 * growth_rate + in_channels, out_channels=growth_rate, kernel_size=(3, 3),
                               padding='same', stride=(1, 1))

    def forward(self, x):
        x0 = x
        x1 = F.tanh(self.conv1(x0))
        x2 = F.tanh(self.conv2(torch.cat((x0, x1), 1)))
        x3 = F.tanh(self.conv3(torch.cat((x0, x1, x2), 1)))
        x4 = F.tanh(self.conv4(torch.cat((x0, x1, x2, x3), 1)))
        x5 = F.tanh(self.conv5(torch.cat((x0, x1, x2, x3, x4), 1)))
        x6 = F.tanh(self.conv6(torch.cat((x0, x1, x2, x3, x4, x5), 1)))
        x7 = F.tanh(self.conv7(torch.cat((x0, x1, x2, x3, x4, x5, x6), 1)))
        x8 = F.tanh(self.conv8(torch.cat((x0, x1, x2, x3, x4, x5, x6, x7), 1)))

        return torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 1)


class TD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same',
                              stride=(1, 1))
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = F.tanh(self.conv(x))
        x = self.maxPool(x)
        return x


class TU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same',
                              stride=(1, 1))

        # TODO swapped from transposed conv. to upsample + conv because PyTorch doesn't allow SAME padding for
        #  trans. conv. and I don't want to add code calculating it. AFAIK it also avoid checkerboard artifacts.
        #  Nonetheless trans. conv. is worth trying if SAME padding is supported (TF might offer it).
        # self.transpConv = nn.ConvTranspose2d(in_channels=in_channels, , kernel_size=(3, 3),
        #                                      padding='same', stride=(2, 2))

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class PhantomNet(nn.Module):
    """
    FCN model based on the U-Net architecture.

    Reference: https://arxiv.org/pdf/1811.04602.pdf
    """

    def __init__(self, num_of_subbands):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of_subbands, out_channels=64, padding='same', kernel_size=(3, 3))  # o=64
        self.tdb1 = TDB4(in_channels=64, growth_rate=32)  # out_channels == 128

        self.td1 = TD(in_channels=128, out_channels=128)
        self.tdb2 = TDB4(in_channels=128, growth_rate=64)  # out_channels == 256

        self.td2 = TD(in_channels=256, out_channels=256)
        self.tdb3 = TDB4(in_channels=256, growth_rate=128)  # out_channels === 512

        self.td3 = TD(in_channels=512, out_channels=512)
        self.tdb4 = TDB8(in_channels=512, growth_rate=64)  # out_channels == 512
        self.tu1 = TU(in_channels=512, out_channels=512)

        self.tdb5 = TDB4(in_channels=512 + 512, growth_rate=64)  # out_channels == 256
        self.tu2 = TU(in_channels=256, out_channels=256)

        self.tdb6 = TDB4(in_channels=256 + 256, growth_rate=32)  # out_channels == 128
        self.tu3 = TU(in_channels=128, out_channels=128)

        self.tdb7 = TDB4(in_channels=128 + 128, growth_rate=16)  # out_channels == 64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=num_of_subbands, kernel_size=(3, 3), padding='same',
                               stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        tdb1 = self.tdb1(x)

        x = self.td1(tdb1)
        tdb2 = self.tdb2(x)

        x = self.td2(tdb2)
        tdb3 = self.tdb3(x)

        x = self.td3(tdb3)
        x = self.tdb4(x)
        x = self.tu1(x)

        x = torch.cat((tdb3, x), 1)
        x = self.tdb5(x)
        x = self.tu2(x)

        x = torch.cat((tdb2, x), 1)
        x = self.tdb6(x)
        x = self.tu3(x)

        x = torch.cat((tdb1, x), 1)
        x = self.tdb7(x)
        x = self.conv2(x)

        return x
