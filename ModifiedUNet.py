import torch
import torch.nn as nn
import torch.nn.functional as F


class TDB4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))

    def forward(self, x):
        x0 = x
        x1 = F.tanh(self.conv1(x0))
        x2 = F.tanh(self.conv2(torch.cat((x0, x1))))
        x3 = F.tanh(self.conv3(torch.cat((x0, x1, x2))))
        x4 = F.tanh(self.conv4(torch.cat((x0, x1, x2, x3))))

        return torch.cat((x0, x1, x2, x3, x4))  # TODO remove x0, difference w/ DB?


class TDB8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv7 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))
        self.conv8 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1))

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

        return torch.cat((x0, x1, x2, x3, x4, x5, x6, x7, x8))  # TODO remove x0, difference w/ DB?


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


# TODO 2 questions here,
#  what are in/out_channels for intermediate layers in TBD
#  is concatenating different from UNet?


class ModifiedUNet(nn.Module):
    def __init__(self, num_of_subbands):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3))  # TODO chng in_channels to 3 if rgb
        self.tdb1 = TDB4(64, 128)

        self.td1 = TD(128, 128)
        self.tdb2 = TDB4(128, 256)

        self.td2 = TD(256, 256)
        self.tdb3 = TDB4(256, 512)

        self.td3 = TD(512, 512)
        self.tdb4 = TDB8(512, 512)
        self.tu1 = TU(512, 512)

        self.tdb5 = TDB4(A, B)
        self.tu2 = TU(A, B)

        self.tdb6 = TDB4(A, B)
        self.tu3 = TU(A, B)

        self.tdb7 = TDB4(A, 64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=num_of_subbands, kernel_size=(3, 3))

    def forward(self, x):
        print('in modified unet final x shape is', x.shape)

        return x
