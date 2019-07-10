import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlockInstanceNorm(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockInstanceNorm, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class ResidualBlockNoNorm(nn.Module):
    """Residual Block with no normalization.
    In AppendixA.2, Our discriminator is a Patch GAN discriminator [21].
        It utilizes the Leaky ReLU nonlinearity and employs no normalization.
    """
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockNoNorm, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False))
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.lrelu(x + self.main(x))


class ResidualBlockNoNormUpsample(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockNoNormUpsample, self).__init__()
        self.learned_shortcut = (dim_in != dim_out)
        dim_middle = min(dim_in, dim_out)

        # create conv layers
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_middle, kernel_size=3, padding=1),
            nn.Conv2d(dim_middle, dim_out, kernel_size=3, padding=1)
        )

        self.conv_s = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)


    def forward(self, x):
        x_s = self.conv_s(x)
        dx = self.main(x)

        out = x_s + dx

        return out


class ContentEncoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=2):
        super(ContentEncoder, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)) # Official is k3,p1,s2
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(3):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlockInstanceNorm(dim_in=curr_dim, dim_out=curr_dim))

        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x) # -> [1, 512, 16, 16]



class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(Generator, self).__init__()

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
        In AppendixA.2, Our discriminator is a Patch GAN discriminator [21].
        It utilizes the Leaky ReLU nonlinearity and employs no normalization."""
    def __init__(self, conv_dim=64, c_dim=256):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)) # Official is maybe 312.
        layers.append(nn.LeakyReLU(inplace=True))

        curr_dim = conv_dim
        
        # for _ in range(5):
        # # Official paper is wrong?
        # # I can not understand too, https://github.com/NVlabs/FUNIT/issues/3 
        #     curr_dim = curr_dim * 2
        #     if curr_dim > 1024:
        #         curr_dim = 1024
        #     layers.append(nn.AvgPool2d(2, 2))
        #     for _ in range(2):
        #         layers.append(ResidualBlockNoNorm(dim_in=curr_dim, dim_out=curr_dim))
        # layers.append(nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=1, padding=1))

        for _ in range(4):
            layers.append(nn.AvgPool2d(2, 2))
            layers.append(ResidualBlockNoNormUpsample(dim_in=curr_dim, dim_out=curr_dim*2))
            layers.append(ResidualBlockNoNormUpsample(dim_in=curr_dim*2, dim_out=curr_dim*2))
            curr_dim = curr_dim * 2

        layers.append(ResidualBlockNoNorm(dim_in=curr_dim, dim_out=curr_dim))
        layers.append(ResidualBlockNoNorm(dim_in=curr_dim, dim_out=curr_dim))


        layers.append(nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=1, padding=1))

        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x) # -> [1, 10, 4, 4]


if __name__ == "__main__":
    z = torch.randn(1,3,128,128).cuda()

    # # Content Encoder
    # CE = ContentEncoder(conv_dim=64, repeat_num=2).cuda()
    # x = CE(z)
    # print(x.size()) # -> [1, 512, 16, 16]

    # Discriminator
    D = Discriminator(conv_dim=64, c_dim=10).cuda()
    out = D(z)
    print(out.size())