import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time



class AdainLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = linear
    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm2d(nn.Module):
    """AdaptiveInstanceNorm2d"""
    def __init__(self, in_channel=0, style_dim=0):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = AdainLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3) # -> [4, 1024, 1, 1]
        
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


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


class ResidualBlockAdaIn(nn.Module):
    """Residual Block with Adaptive Instance normalization.
        For generator upsampling.
    """
    def __init__(self, dim_in=512, dim_out=512):
        super(ResidualBlockAdaIn, self).__init__()
        dim_middle = min(dim_in, dim_out)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(dim_in, dim_middle, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain1 = AdaptiveInstanceNorm2d(in_channel=512, style_dim=1024)
        self.conv2 = nn.ConvTranspose2d(dim_middle, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain2 = AdaptiveInstanceNorm2d(in_channel=512, style_dim=1024)

    def forward(self, x, style):
        dx = self.conv1(x)
        dx = self.adain1(dx, style)
        dx = self.relu(dx)
        dx = self.conv2(dx)
        dx = self.adain2(dx, style)

        out = self.relu(dx + x)

        return out



class ResidualBlockNoNorm(nn.Module):
    """Residual Block with no normalization.
        In AppendixA.2, Our discriminator is a Patch GAN discriminator [21].
        It utilizes the Leaky ReLU nonlinearity and employs no normalization.
    """

    def __init__(self, dim_in, dim_out):
        super(ResidualBlockNoNorm, self).__init__()
        self.learned_shortcut = (dim_in != dim_out)
        dim_middle = min(dim_in, dim_out)

        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_middle, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim_middle, dim_out, kernel_size=3, padding=1)
        )

        self.conv_s = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)
        self.lrelu = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        x_s = self.conv_s(x)
        dx = self.main(x)

        return self.lrelu(x_s + dx)


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



class ContentDecoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=2):
        super(ContentDecoder, self).__init__()
        curr_dim = conv_dim * 8

        self.adain_res1 = ResidualBlockAdaIn(dim_in=curr_dim, dim_out=curr_dim)
        self.adain_res2 = ResidualBlockAdaIn(dim_in=curr_dim, dim_out=curr_dim)

        layers = []
        for _ in range(3):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.conv = nn.Sequential(*layers)

    def forward(self, x, style):
        out = self.adain_res1(x, style)
        out = self.adain_res2(out, style)

        return self.conv(out)



class StyleEncoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=5):
        super(StyleEncoder, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)) # Official is k3,p1,s2
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(repeat_num - 1):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        layers.append(nn.AdaptiveMaxPool2d(1))
        self.down_sample = nn.Sequential(*layers)

        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        nn.init.xavier_uniform_(self.linear1.weight.data)
        nn.init.xavier_uniform_(self.linear2.weight.data)
        nn.init.xavier_uniform_(self.linear3.weight.data)

    def forward(self, styles):
        donwn_sampled = []
        for style in styles:
            x = self.down_sample(style.cuda()) # -> [4, 1024, 1, 1]
            x = torch.squeeze(x, -1)
            x = torch.squeeze(x, -1)
            donwn_sampled.append(x)
        mean = sum(donwn_sampled) / len(donwn_sampled) # -> [b, 1024]
        mean = self.linear3(self.linear2(self.linear1(mean)))
        return mean # -> [4, 512]



class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64):
        super(Generator, self).__init__()
        self.CE = ContentEncoder(conv_dim=conv_dim, repeat_num=2)
        self.SE = StyleEncoder(conv_dim=conv_dim, repeat_num=5)
        self.CD = ContentDecoder(conv_dim=conv_dim, repeat_num=2)

    def forward(self, x, styles):
        style = self.SE(styles) # -> [4, 1024]
        x = self.CE(x) # -> [4, 512, 16, 16]

        return self.CD(x, style)



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
            layers.append(ResidualBlockNoNorm(dim_in=curr_dim, dim_out=curr_dim*2))
            layers.append(ResidualBlockNoNorm(dim_in=curr_dim*2, dim_out=curr_dim*2))
            curr_dim = curr_dim * 2

        layers.append(ResidualBlockNoNorm(dim_in=curr_dim, dim_out=curr_dim))
        layers.append(ResidualBlockNoNorm(dim_in=curr_dim, dim_out=curr_dim))


        layers.append(nn.Conv2d(curr_dim, c_dim, kernel_size=3, stride=1, padding=1))

        self.main = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.main(x) # -> [1, 10, 4, 4]


if __name__ == "__main__":
    z = torch.randn(4,3,128,128).cuda()

    # # Content Encoder
    # CE = ContentEncoder(conv_dim=64, repeat_num=2).cuda()
    # x = CE(z)
    # print(x.size()) # -> [1, 512, 16, 16]

    # Style Encoder
    # start = time() # -> 0.2872
    styles = [torch.randn(4,3,128,128) for _ in range(10)]
    # SE = StyleEncoder(conv_dim=64, repeat_num=5).cuda()
    # x = SE(styles)
    # print(x.size())
    # print(time() - start)


    # # Discriminator
    # D = Discriminator(conv_dim=64, c_dim=10).cuda()
    # out = D(z)
    # print(out.size())

    # Generator
    G = Generator(conv_dim=64).cuda()
    y = G(z, styles)
    print(y.size())