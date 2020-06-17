import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_input_channels, n_output_channels, mode='nearest', residual=False, kernel_regularisation=False, num_internal_channels=8, ndim=2):
        super(UNet, self).__init__()
        self.n_channels = n_input_channels
        self.n_classes = n_output_channels
        self.residual = residual
        self.kernel_regularisation = kernel_regularisation
        self.kernel = None

        n = num_internal_channels

        self.inc = DoubleConv(n_input_channels, n, ndim=ndim)
        self.down1 = Down(n, n * 2, ndim=ndim)
        self.down2 = Down(n * 2, n * 4, ndim=ndim)
        self.down3 = Down(n * 4, n * 8, ndim=ndim)
        factor = 2 if mode == 'bilinear' or mode == 'nearest' else 1
        self.down4 = Down(n * 8, n * 16 // factor, ndim=ndim)
        self.up1 = Up(n * 16, n * 8 // factor, mode, ndim=ndim)
        self.up2 = Up(n * 8, n * 4 // factor, mode, ndim=ndim)
        self.up3 = Up(n * 4, n * 2 // factor, mode, ndim=ndim)
        self.up4 = Up(n * 2, n, mode, ndim=ndim)
        self.outc = OutConv(n, n_output_channels, ndim=ndim)

    def forward(self, x0):

        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x + x0 if self.residual else x

    def post_optimisation(self):
        b = 1e-5
        if self.kernel_regularisation:
            with torch.no_grad():
                weights = self.outc.conv._parameters['weight']
                num_channels = weights.shape[1]
                if self.kernel is None:
                    kernel = numpy.array([[b, b, b], [b, 1, b], [b, b, b]])
                    kernel = kernel[numpy.newaxis, numpy.newaxis, ...].astype(
                        numpy.float32
                    )
                    self.kernel = torch.from_numpy(kernel).to(weights.device)
                    self.kernel /= self.kernel.sum()
                    self.kernel = self.kernel.expand(num_channels, 1, -1, -1)

                weights = F.conv2d(weights, self.kernel, groups=num_channels, padding=1)
                self.outc.conv._parameters['weight'] = weights



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, ndim=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if ndim==2:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        elif ndim==3:
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=5, padding=2),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(),
                nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, ndim=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2) if ndim == 2 else (nn.MaxPool3d(2) if ndim == 3 else None),
            DoubleConv(in_channels, out_channels, ndim=ndim)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mode='bilinear', ndim=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if mode == 'bilinear' or mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True if mode == 'bilinear' else None)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, ndim=ndim)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, ndim=ndim)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, ndim=2):
        super(OutConv, self).__init__()
        if ndim==2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)
        elif ndim==3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        return self.conv(x)
