import torch
import torch.nn as nn
import torch.nn.functional as F


class TDB4(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=1 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=2 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=3 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))

    def forward(self, x):
        x0 = x
        x1 = F.tanh(self.conv1(x0))
        x2 = F.tanh(self.conv2(torch.cat((x0, x1))))
        x3 = F.tanh(self.conv3(torch.cat((x0, x1, x2))))
        x4 = F.tanh(self.conv4(torch.cat((x0, x1, x2, x3))))

        return torch.cat((x1, x2, x3, x4))  # TODO difference w/ DB is that x0 is not included here?


class TDB8(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=1 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=2 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=3 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=4 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=5 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=6 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))
        self.conv8 = nn.Conv2d(in_channels=7 * growth_rate, out_channels=growth_rate, kernel_size=(3, 3), stride=(1, 1))

    def forward(self, x):
        x0 = x
        x1 = F.tanh(self.conv1(x0))
        x2 = F.tanh(self.conv2(torch.cat((x0, x1))))
        x3 = F.tanh(self.conv3(torch.cat((x0, x1, x2))))
        x4 = F.tanh(self.conv4(torch.cat((x0, x1, x2, x3))))
        x5 = F.tanh(self.conv4(torch.cat((x0, x1, x2, x3, x4))))
        x6 = F.tanh(self.conv4(torch.cat((x0, x1, x2, x3, x4, x5))))
        x7 = F.tanh(self.conv4(torch.cat((x0, x1, x2, x3, x4, x5, x6))))
        x8 = F.tanh(self.conv4(torch.cat((x0, x1, x2, x3, x4, x5, x6, x7))))

        return torch.cat((x1, x2, x3, x4, x5, x6, x7, x8))  # TODO difference w/ DB is that x0 is not included here?


class TD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # TODO double check this

    def forward(self, x):
        x = F.tanh(self.conv(x))
        x = self.maxPool(x)
        return x


class TU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transpConv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                             stride=(2, 2))

    def forward(self, x):
        return self.transpConv(x)


# TODO 3 questions here,
#  what are in/out_channels for intermediate layers in TBD
#  is concatenating different from UNet?
#  what about padding?


class ModifiedUNet(nn.Module):
    def __init__(self, num_of_subbands):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_of_subbands, out_channels=64, kernel_size=(3, 3))
        self.tdb1 = TDB4(in_channels=64, growth_rate=16)  # out_channels == 128

        self.td1 = TD(in_channels=128, out_channels=128)
        self.tdb2 = TDB4(in_channels=128, growth_rate=32)  # out_channels == 256

        self.td2 = TD(in_channels=256, out_channels=256)
        self.tdb3 = TDB4(in_channels=256, growth_rate=64)  # out_channels === 512

        self.td3 = TD(in_channels=512, out_channels=512)
        self.tdb4 = TDB8(in_channels=512, growth_rate=128)  # out_channels == 512
        self.tu1 = TU(in_channels=512, out_channels=512)

        self.tdb5 = TDB4(in_channels=?, growth_rate = 64)  # out_channels == 256
        self.tu2 = TU(in_channels=256, out_channels=256)

        self.tdb6 = TDB4(in_channels=?, growth_rate = 32)  # out_channels == 128
        self.tu3 = TU(in_channels=128, out_channels=128)

        self.tdb7 = TDB4(in_channels=?, growth_rate = 16)  # out_channels == 64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=num_of_subbands, kernel_size=(3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.tdb1(x)
        xFirstConcat = x

        x = self.td1(x)
        x = self.tdb2(x)
        xSecondConcat = x

        x = self.tdb2(x)
        x = self.tdb3(x)
        xThirdConcat = x

        x = self.td3(x)
        x = self.tdb4(x)
        x = self.tu1(x)

        x = torch.cat((xThirdConcat, x), 1)  # TODO apply concat
        x = self.tdb5(x)
        x = self.tu2(x)

        x = torch.cat((xSecondConcat, x), 1)  # TODO apply concat
        x = self.tdb6(x)
        x = self.tu3(x)

        x = torch.cat((xFirstConcat, x), 1)  # TODO apply concat
        x = self.tdb7(x)
        x = self.conv2(x)

        print('in modified unet final x shape is', x.shape)

        return x
