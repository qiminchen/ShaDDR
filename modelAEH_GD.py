import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable


# cell = 4
# input 256
# output 120 (128-4-4)
# receptive field = 18

# r_l-1 = s_l * r_l + (k_l - s_l)
#             0  18
# conv 4x4 s1 4  15
# conv 3x3 s2 6  7
# conv 3x3 s1 10 5
# conv 3x3 s1 14 3
# conv 3x3 s1 18 1
# conv 1x1 s1 1  1
class discriminator_rf36(nn.Module):
    def __init__(self, d_dim, z_dim, d_in=1):
        super(discriminator_rf36, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim

        # receptive field 36, input 512, output 120 (works okay)
        self.conv_1 = nn.Conv3d(d_in,            self.d_dim,      4, stride=1, padding=0, bias=True)  # 509, 4
        self.conv_2 = nn.Conv3d(self.d_dim,      self.d_dim * 2,  3, stride=2, padding=0, bias=True)  # 254, 4+(3-1)
        self.conv_3 = nn.Conv3d(self.d_dim * 2,  self.d_dim * 4,  4, stride=2, padding=0, bias=True)  # 126, 6+(4-1)*2
        self.conv_4 = nn.Conv3d(self.d_dim * 4,  self.d_dim * 8,  3, stride=1, padding=0, bias=True)  # 124, 12+(3-1)*4
        self.conv_5 = nn.Conv3d(self.d_dim * 8,  self.d_dim * 16, 3, stride=1, padding=0, bias=True)  # 122, 20+(3-1)*4
        self.conv_6 = nn.Conv3d(self.d_dim * 16, self.d_dim * 32, 3, stride=1, padding=0, bias=True)  # 120, 28+(3-1)*4
        self.conv_7 = nn.Conv3d(self.d_dim * 32, self.z_dim,      1, stride=1, padding=0, bias=True)  # 120, 36

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = torch.sigmoid(out)

        return out


class discriminator_rf18(nn.Module):
    def __init__(self, d_dim, z_dim, d_in=1):
        super(discriminator_rf18, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim

        # receptive field 18, input 128, output 56, input 256, output 120
        self.conv_1 = nn.Conv3d(d_in,            self.d_dim,      4, stride=1, padding=0, bias=True)  # 4
        self.conv_2 = nn.Conv3d(self.d_dim,      self.d_dim * 2,  3, stride=2, padding=0, bias=True)  # 4+(3-1)
        self.conv_3 = nn.Conv3d(self.d_dim * 2,  self.d_dim * 4,  3, stride=1, padding=0, bias=True)  # 6+(3-1)*2
        self.conv_4 = nn.Conv3d(self.d_dim * 4,  self.d_dim * 8,  3, stride=1, padding=0, bias=True)  # 10+(3-1)*2
        self.conv_5 = nn.Conv3d(self.d_dim * 8,  self.d_dim * 16, 3, stride=1, padding=0, bias=True)  # 14+(3-1)*2
        self.conv_6 = nn.Conv3d(self.d_dim * 16, self.z_dim,      1, stride=1, padding=0, bias=True)  # 18

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = torch.sigmoid(out)

        return out


class discriminator2d(nn.Module):
    def __init__(self, d_dim, z_dim, d_in=1, rf=18):
        super(discriminator2d, self).__init__()
        self.d_dim = d_dim
        self.z_dim = z_dim
        self.rf = rf

        if self.rf == 11:
            # receptive field 11, output 502
            self.conv_1 = nn.Conv2d(d_in,            self.d_dim,      3, stride=1, padding=0, bias=True)  # 3
            self.conv_2 = nn.Conv2d(self.d_dim,      self.d_dim * 2,  3, stride=1, padding=0, bias=True)  # 3+(3-1)
            self.conv_3 = nn.Conv2d(self.d_dim * 2,  self.d_dim * 4,  3, stride=1, padding=0, bias=True)  # 5+(3-1)
            self.conv_4 = nn.Conv2d(self.d_dim * 4,  self.d_dim * 8,  3, stride=1, padding=0, bias=True)  # 7+(3-1)
            self.conv_5 = nn.Conv2d(self.d_dim * 8,  self.d_dim * 16, 3, stride=1, padding=0, bias=True)  # 9+(3-1)
            self.conv_6 = nn.Conv2d(self.d_dim * 16, self.z_dim,      1, stride=1, padding=0, bias=True)  # 11
        elif self.rf == 18:
            # receptive field 18, output 248
            self.conv_1 = nn.Conv2d(d_in,            self.d_dim,      4, stride=1, padding=0, bias=True)  # 4
            self.conv_2 = nn.Conv2d(self.d_dim,      self.d_dim * 2,  3, stride=2, padding=0, bias=True)  # 4+(3-1)
            self.conv_3 = nn.Conv2d(self.d_dim * 2,  self.d_dim * 4,  3, stride=1, padding=0, bias=True)  # 6+(3-1)*2
            self.conv_4 = nn.Conv2d(self.d_dim * 4,  self.d_dim * 8,  3, stride=1, padding=0, bias=True)  # 10+(3-1)*2
            self.conv_5 = nn.Conv2d(self.d_dim * 8,  self.d_dim * 16, 3, stride=1, padding=0, bias=True)  # 14+(3-1)*2
            self.conv_6 = nn.Conv2d(self.d_dim * 16, self.z_dim,      1, stride=1, padding=0, bias=True)  # 18
        elif self.rf == 36:
            # receptive field 36, output 120
            self.conv_1 = nn.Conv2d(d_in,            self.d_dim,      4, stride=1, padding=0, bias=True)  # 541, 4
            self.conv_2 = nn.Conv2d(self.d_dim,      self.d_dim * 2,  3, stride=2, padding=0, bias=True)  # 270, 4+(3-1)
            self.conv_3 = nn.Conv2d(self.d_dim * 2,  self.d_dim * 4,  4, stride=2, padding=0, bias=True)  # 134, 6+(4-1)*2
            self.conv_4 = nn.Conv2d(self.d_dim * 4,  self.d_dim * 8,  3, stride=1, padding=0, bias=True)  # 132, 12+(3-1)*4
            self.conv_5 = nn.Conv2d(self.d_dim * 8,  self.d_dim * 16, 3, stride=1, padding=0, bias=True)  # 130, 20+(3-1)*4
            self.conv_6 = nn.Conv2d(self.d_dim * 16, self.d_dim * 32, 3, stride=1, padding=0, bias=True)  # 128, 28+(3-1)*4
            self.conv_7 = nn.Conv2d(self.d_dim * 32, self.z_dim,      1, stride=1, padding=0, bias=True)  # 36

    def forward(self, voxels, is_training=False):
        out = voxels

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        if self.rf == 11 or self.rf == 18:
            out = self.conv_6(out)
            out = torch.sigmoid(out)

            return out
        elif self.rf == 36:
            out = self.conv_6(out)
            out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

            out = self.conv_7(out)
            out = torch.sigmoid(out)

            return out


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# 64 -> 512 dual branch
class backend_generator(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(backend_generator, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_0 = nn.Conv3d(1 + self.z_dim,              self.g_dim,      5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim + self.z_dim,     self.g_dim * 2,  5, stride=1, dilation=2, padding=4, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim * 2 + self.z_dim, self.g_dim * 4,  5, stride=1, dilation=2, padding=4, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim * 4 + self.z_dim, self.g_dim * 8,  5, stride=1, dilation=1, padding=2, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim * 8 + self.z_dim, self.g_dim * 8,  5, stride=1, dilation=1, padding=2, bias=True)

    def forward(self, voxels, z, is_training=False):
        out = voxels

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        return out


class geometry_generator(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(geometry_generator, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_5 = nn.ConvTranspose3d(self.g_dim * 8 + self.z_dim,  self.g_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim * 4 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)

        self.conv_7 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim * 2 + self.z_dim,           self.g_dim * 2, 3, stride=1, padding=1, bias=True)
        self.conv_8_out = nn.Conv3d(self.g_dim * 2,                    1,              3, stride=1, padding=1, bias=True)

        self.conv_9 = nn.ConvTranspose3d(self.g_dim * 2 + self.z_dim,  self.g_dim * 1, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim * 1 + self.z_dim,          self.g_dim * 1, 3, stride=1, padding=1, bias=True)
        self.conv_10_out = nn.Conv3d(self.g_dim * 1,                   1,              3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask_512 = F.interpolate(mask_, scale_factor=8, mode='nearest')
        mask_256 = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_256 = self.conv_8_out(out)
        out_256 = torch.max(torch.min(out_256, out_256 * 0.002 + 0.998), out_256 * 0.002)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_10(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_512 = self.conv_10_out(out)
        out_512 = torch.max(torch.min(out_512, out_512 * 0.002 + 0.998), out_512 * 0.002)

        out_512 = out_512 * mask_512
        out_256 = out_256 * mask_256

        return out_512, out_256


class texture_generator(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(texture_generator, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_5 = nn.ConvTranspose3d(self.g_dim * 8 + self.z_dim,  self.g_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim * 4 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim * 2 + self.z_dim,           self.g_dim * 2, 3, stride=1, padding=1, bias=True)
        self.conv_9 = nn.ConvTranspose3d(self.g_dim * 2 + self.z_dim,  self.g_dim * 1, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim * 1,                       3,              3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, is_training=False):
        out = voxels

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_10(out)
        out = torch.max(torch.min(out, out * 0.002 + 0.998), out * 0.002)

        return out


class generator_dual(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(generator_dual, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        geometry_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.geometry_codes = nn.Parameter(geometry_codes)
        nn.init.constant_(self.geometry_codes, 0.0)

        texture_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.texture_codes = nn.Parameter(texture_codes)
        nn.init.constant_(self.texture_codes, 0.0)

        self.backend_generator = backend_generator(self.g_dim, self.prob_dim, self.z_dim)
        self.backend_texture_generator = backend_generator(self.g_dim, self.prob_dim, self.z_dim)
        self.geometry_generator = geometry_generator(self.g_dim, self.prob_dim, self.z_dim)
        self.texture_generator = texture_generator(self.g_dim, self.prob_dim, self.z_dim)

    def forward(self, voxels, geometry_z, texture_z, mask_, is_geometry_training=True):
        out = voxels

        # backend
        if is_geometry_training:
            out = self.backend_generator(out, geometry_z)
            out_512, out_256 = self.geometry_generator(out, geometry_z, mask_)

            return out_512, out_256
        else:
            with torch.no_grad():
                out_geometry = self.backend_generator(out, geometry_z.detach())
                out_geometry, _ = self.geometry_generator(out_geometry, geometry_z.detach(), mask_)

            out_texture = self.backend_texture_generator(out, geometry_z.detach())
            out_texture = self.texture_generator(out_texture, texture_z)

            return out_geometry, out_texture


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# 32 -> 256 dual branch
class backend_generator_halfsize_x8(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(backend_generator_halfsize_x8, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_0 = nn.Conv3d(1 + self.z_dim,              self.g_dim,      3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim + self.z_dim,     self.g_dim * 2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim * 2 + self.z_dim, self.g_dim * 4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim * 4 + self.z_dim, self.g_dim * 8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim * 8 + self.z_dim, self.g_dim * 16, 3, stride=1, dilation=1, padding=1, bias=True)

    def forward(self, voxels, z, is_training=False):
        out = voxels

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        return out


class geometry_generator_halfsize_x8(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(geometry_generator_halfsize_x8, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_5 = nn.ConvTranspose3d(self.g_dim * 16 + self.z_dim, self.g_dim * 8, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim * 8 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)

        self.conv_7 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim * 2 + self.z_dim,           self.g_dim * 2, 3, stride=1, padding=1, bias=True)
        self.conv_8_out = nn.Conv3d(self.g_dim * 2,                    1,              3, stride=1, padding=1, bias=True)

        self.conv_9 = nn.ConvTranspose3d(self.g_dim * 2 + self.z_dim,  self.g_dim * 1, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim * 1 + self.z_dim,          self.g_dim * 1, 3, stride=1, padding=1, bias=True)
        self.conv_10_out = nn.Conv3d(self.g_dim * 1,                   1,              3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask_256 = F.interpolate(mask_, scale_factor=8, mode='nearest')
        mask_128 = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_128 = self.conv_8_out(out)
        out_128 = torch.max(torch.min(out_128, out_128 * 0.002 + 0.998), out_128 * 0.002)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_10(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_256 = self.conv_10_out(out)
        out_256 = torch.max(torch.min(out_256, out_256 * 0.002 + 0.998), out_256 * 0.002)

        out_256 = out_256 * mask_256
        out_128 = out_128 * mask_128

        return out_256, out_128


class texture_generator_halfsize_x8(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(texture_generator_halfsize_x8, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_5 = nn.ConvTranspose3d(self.g_dim * 16 + self.z_dim, self.g_dim * 8, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim * 8 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim * 2 + self.z_dim,           self.g_dim * 2, 3, stride=1, padding=1, bias=True)
        self.conv_9 = nn.ConvTranspose3d(self.g_dim * 2 + self.z_dim,  self.g_dim * 1, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim * 1,                       3,              3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, is_training=False):
        out = voxels

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_10(out)
        out = torch.max(torch.min(out, out * 0.002 + 0.998), out * 0.002)
        # out = torch.sigmoid(out)

        return out


# 32 -> 256 dual branch
class generator_dual_halfsize_x8(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(generator_dual_halfsize_x8, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        geometry_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.geometry_codes = nn.Parameter(geometry_codes)
        nn.init.constant_(self.geometry_codes, 0.0)

        texture_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.texture_codes = nn.Parameter(texture_codes)
        nn.init.constant_(self.texture_codes, 0.0)

        self.backend_generator = backend_generator_halfsize_x8(self.g_dim, self.prob_dim, self.z_dim)
        self.backend_texture_generator = backend_generator_halfsize_x8(self.g_dim, self.prob_dim, self.z_dim)
        self.geometry_generator = geometry_generator_halfsize_x8(self.g_dim, self.prob_dim, self.z_dim)
        self.texture_generator = texture_generator_halfsize_x8(self.g_dim, self.prob_dim, self.z_dim)

    def forward(self, voxels, geometry_z, texture_z, mask_, is_geometry_training=True):
        out = voxels

        # backbone
        if is_geometry_training:
            out = self.backend_generator(out, geometry_z)
            out_256, out_128 = self.geometry_generator(out, geometry_z, mask_)

            return out_256, out_128
        else:
            with torch.no_grad():
                out_geometry = self.backend_generator(out, geometry_z)
                out_geometry, _ = self.geometry_generator(out_geometry, geometry_z, mask_)

            out_texture = self.backend_texture_generator(out, geometry_z.detach())
            out_texture = self.texture_generator(out_texture, texture_z)

            return out_geometry, out_texture


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# 32 -> 256 dual branch small
class backend_generator_halfsize_x8_small(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(backend_generator_halfsize_x8_small, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_0 = nn.Conv3d(1 + self.z_dim,              self.g_dim,      3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim + self.z_dim,     self.g_dim * 2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim * 2 + self.z_dim, self.g_dim * 4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim * 4 + self.z_dim, self.g_dim * 8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim * 8 + self.z_dim, self.g_dim * 8,  3, stride=1, dilation=1, padding=1, bias=True)

    def forward(self, voxels, z, is_training=False):
        out = voxels

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        return out


class geometry_generator_halfsize_x8_small(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(geometry_generator_halfsize_x8_small, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_5 = nn.ConvTranspose3d(self.g_dim * 8 + self.z_dim,  self.g_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim * 4 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)

        self.conv_7 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim * 2 + self.z_dim,           self.g_dim * 2, 3, stride=1, padding=1, bias=True)
        self.conv_8_out = nn.Conv3d(self.g_dim * 2,                    1,              3, stride=1, padding=1, bias=True)

        self.conv_9 = nn.ConvTranspose3d(self.g_dim * 2 + self.z_dim,  self.g_dim * 1, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim * 1 + self.z_dim,          self.g_dim * 1, 3, stride=1, padding=1, bias=True)
        self.conv_10_out = nn.Conv3d(self.g_dim * 1,                   1,              3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask_256 = F.interpolate(mask_, scale_factor=8, mode='nearest')
        mask_128 = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_128 = self.conv_8_out(out)
        out_128 = torch.max(torch.min(out_128, out_128 * 0.002 + 0.998), out_128 * 0.002)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_10(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_256 = self.conv_10_out(out)
        out_256 = torch.max(torch.min(out_256, out_256 * 0.002 + 0.998), out_256 * 0.002)

        out_256 = out_256 * mask_256
        out_128 = out_128 * mask_128

        return out_256, out_128


class texture_generator_halfsize_x8_small(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(texture_generator_halfsize_x8_small, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_5 = nn.ConvTranspose3d(self.g_dim * 8 + self.z_dim,  self.g_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim * 4 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim * 2 + self.z_dim,           self.g_dim * 2, 3, stride=1, padding=1, bias=True)
        self.conv_9 = nn.ConvTranspose3d(self.g_dim * 2 + self.z_dim,  self.g_dim * 1, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim * 1,                       3,              3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, is_training=False):
        out = voxels

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_10(out)
        out = torch.max(torch.min(out, out * 0.002 + 0.998), out * 0.002)
        # out = torch.sigmoid(out)

        return out


# 32 -> 256 dual branch small
class generator_dual_halfsize_x8_small(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(generator_dual_halfsize_x8_small, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        geometry_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.geometry_codes = nn.Parameter(geometry_codes)
        nn.init.constant_(self.geometry_codes, 0.0)

        texture_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.texture_codes = nn.Parameter(texture_codes)
        nn.init.constant_(self.texture_codes, 0.0)

        self.backend_generator = backend_generator_halfsize_x8_small(self.g_dim, self.prob_dim, self.z_dim)
        self.backend_texture_generator = backend_generator_halfsize_x8_small(self.g_dim, self.prob_dim, self.z_dim)
        self.geometry_generator = geometry_generator_halfsize_x8_small(self.g_dim, self.prob_dim, self.z_dim)
        self.texture_generator = texture_generator_halfsize_x8_small(self.g_dim, self.prob_dim, self.z_dim)

    def forward(self, voxels, geometry_z, texture_z, mask_, is_geometry_training=True):
        out = voxels

        # backbone
        if is_geometry_training:
            out = self.backend_generator(out, geometry_z)
            out_256, out_128 = self.geometry_generator(out, geometry_z, mask_)

            return out_256, out_128
        else:
            with torch.no_grad():
                out_geometry = self.backend_generator(out, geometry_z)
                out_geometry, _ = self.geometry_generator(out_geometry, geometry_z, mask_)

            out_texture = self.backend_texture_generator(out, geometry_z.detach())
            out_texture = self.texture_generator(out_texture, texture_z)

            return out_geometry, out_texture


class geometry_generator_halfsize_x8_small_plant(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(geometry_generator_halfsize_x8_small_plant, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_5 = nn.ConvTranspose3d(self.g_dim * 8,  self.g_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim * 4,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)

        self.conv_7 = nn.ConvTranspose3d(self.g_dim * 4,  self.g_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim * 2,           self.g_dim * 2, 3, stride=1, padding=1, bias=True)
        self.conv_8_out = nn.Conv3d(self.g_dim * 2,       1,              3, stride=1, padding=1, bias=True)

        self.conv_9 = nn.ConvTranspose3d(self.g_dim * 2,  self.g_dim * 1, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim * 1,          1,              3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask_256 = F.interpolate(mask_, scale_factor=8, mode='nearest')
        mask_128 = F.interpolate(mask_, scale_factor=4, mode='nearest')

        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_128 = self.conv_8_out(out)
        out_128 = torch.max(torch.min(out_128, out_128 * 0.002 + 0.998), out_128 * 0.002)

        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_256 = self.conv_10(out)
        out_256 = torch.max(torch.min(out_256, out_256 * 0.002 + 0.998), out_256 * 0.002)

        out_256 = out_256 * mask_256
        out_128 = out_128 * mask_128

        return out_256, out_128


class generator_dual_halfsize_x8_small_plant(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(generator_dual_halfsize_x8_small_plant, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        geometry_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.geometry_codes = nn.Parameter(geometry_codes)
        nn.init.constant_(self.geometry_codes, 0.0)

        texture_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.texture_codes = nn.Parameter(texture_codes)
        nn.init.constant_(self.texture_codes, 0.0)

        self.backend_generator = backend_generator_halfsize_x8_small(self.g_dim, self.prob_dim, self.z_dim)
        self.backend_texture_generator = backend_generator_halfsize_x8_small(self.g_dim, self.prob_dim, self.z_dim)
        self.geometry_generator = geometry_generator_halfsize_x8_small_plant(self.g_dim, self.prob_dim, self.z_dim)
        self.texture_generator = texture_generator_halfsize_x8_small(self.g_dim, self.prob_dim, self.z_dim)

    def forward(self, voxels, geometry_z, texture_z, mask_, is_geometry_training=True):
        out = voxels

        # backbone
        if is_geometry_training:
            out = self.backend_generator(out, geometry_z)
            out_256, out_128 = self.geometry_generator(out, geometry_z, mask_)

            return out_256, out_128
        else:
            with torch.no_grad():
                out_geometry = self.backend_generator(out, geometry_z)
                out_geometry, _ = self.geometry_generator(out_geometry, geometry_z, mask_)

            out_texture = self.backend_texture_generator(out, geometry_z.detach())
            out_texture = self.texture_generator(out_texture, texture_z)

            return out_geometry, out_texture


# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# 16 -> 256 dual branch small - for demo
class backend_generator_halfsize_x16_small(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(backend_generator_halfsize_x16_small, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_0 = nn.Conv3d(1 + self.z_dim,              self.g_dim,      3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_1 = nn.Conv3d(self.g_dim + self.z_dim,     self.g_dim * 2,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_2 = nn.Conv3d(self.g_dim * 2 + self.z_dim, self.g_dim * 4,  3, stride=1, dilation=2, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.g_dim * 4 + self.z_dim, self.g_dim * 8,  3, stride=1, dilation=1, padding=1, bias=True)
        self.conv_4 = nn.Conv3d(self.g_dim * 8 + self.z_dim, self.g_dim * 8,  3, stride=1, dilation=1, padding=1, bias=True)

    def forward(self, voxels, z, is_training=False):
        out = voxels

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        return out


class geometry_generator_halfsize_x16_small(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(geometry_generator_halfsize_x16_small, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_5 = nn.ConvTranspose3d(self.g_dim * 8 + self.z_dim,  self.g_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim * 4 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)

        self.conv_7 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim * 4 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)

        self.conv_9 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim * 2 + self.z_dim,          self.g_dim * 2, 3, stride=1, padding=1, bias=True)
        self.conv_10_out = nn.Conv3d(self.g_dim * 2,                   1,              3, stride=1, padding=1, bias=True)

        self.conv_11 = nn.ConvTranspose3d(self.g_dim * 2 + self.z_dim, self.g_dim * 1, 4, stride=2, padding=1, bias=True)
        self.conv_12 = nn.Conv3d(self.g_dim * 1 + self.z_dim,          self.g_dim * 1, 3, stride=1, padding=1, bias=True)
        self.conv_12_out = nn.Conv3d(self.g_dim * 1,                   1,              3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, mask_, is_training=False):
        out = voxels
        mask_256 = F.interpolate(mask_, scale_factor=8, mode='nearest')
        mask_128 = F.interpolate(mask_, scale_factor=4, mode='nearest')

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_10(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_128 = self.conv_10_out(out)
        out_128 = torch.max(torch.min(out_128, out_128 * 0.002 + 0.998), out_128 * 0.002)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_11(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_12(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out_256 = self.conv_12_out(out)
        out_256 = torch.max(torch.min(out_256, out_256 * 0.002 + 0.998), out_256 * 0.002)

        out_256 = out_256 * mask_256
        out_128 = out_128 * mask_128

        return out_256, out_128


class texture_generator_halfsize_x16_small(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(texture_generator_halfsize_x16_small, self).__init__()

        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        self.conv_5 = nn.ConvTranspose3d(self.g_dim * 8 + self.z_dim,  self.g_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_6 = nn.Conv3d(self.g_dim * 4 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)
        self.conv_7 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 4, 4, stride=2, padding=1, bias=True)
        self.conv_8 = nn.Conv3d(self.g_dim * 4 + self.z_dim,           self.g_dim * 4, 3, stride=1, padding=1, bias=True)
        self.conv_9 = nn.ConvTranspose3d(self.g_dim * 4 + self.z_dim,  self.g_dim * 2, 4, stride=2, padding=1, bias=True)
        self.conv_10 = nn.Conv3d(self.g_dim * 2 + self.z_dim,          self.g_dim * 2, 3, stride=1, padding=1, bias=True)
        self.conv_11 = nn.ConvTranspose3d(self.g_dim * 2 + self.z_dim, self.g_dim * 1, 4, stride=2, padding=1, bias=True)
        self.conv_12 = nn.Conv3d(self.g_dim * 1,                       3,              3, stride=1, padding=1, bias=True)

    def forward(self, voxels, z, is_training=False):
        out = voxels

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_5(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_6(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_7(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_8(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_9(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_10(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        _, _, dimx, dimy, dimz = out.size()
        zs = z.repeat(1, 1, dimx, dimy, dimz)
        out = torch.cat([out, zs], dim=1)
        out = self.conv_11(out)
        out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

        out = self.conv_12(out)
        out = torch.max(torch.min(out, out * 0.002 + 0.998), out * 0.002)

        return out


# 16 -> 256 dual branch small - for demo
class generator_dual_halfsize_x16_small(nn.Module):
    def __init__(self, g_dim, prob_dim, z_dim):
        super(generator_dual_halfsize_x16_small, self).__init__()
        self.g_dim = g_dim
        self.prob_dim = prob_dim
        self.z_dim = z_dim

        geometry_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.geometry_codes = nn.Parameter(geometry_codes)
        nn.init.constant_(self.geometry_codes, 0.0)

        texture_codes = torch.zeros((self.prob_dim, self.z_dim))
        self.texture_codes = nn.Parameter(texture_codes)
        nn.init.constant_(self.texture_codes, 0.0)

        self.backend_generator = backend_generator_halfsize_x16_small(self.g_dim, self.prob_dim, self.z_dim)
        self.backend_texture_generator = backend_generator_halfsize_x16_small(self.g_dim, self.prob_dim, self.z_dim)
        self.geometry_generator = geometry_generator_halfsize_x16_small(self.g_dim, self.prob_dim, self.z_dim)
        self.texture_generator = texture_generator_halfsize_x16_small(self.g_dim, self.prob_dim, self.z_dim)

    def forward(self, voxels, geometry_z, texture_z, mask_, is_geometry_training=True):
        out = voxels

        # backbone
        if is_geometry_training:
            out = self.backend_generator(out, geometry_z)
            out_256, out_128 = self.geometry_generator(out, geometry_z, mask_)

            return out_256, out_128
        else:
            with torch.no_grad():
                out_geometry = self.backend_generator(out, geometry_z)
                out_geometry, _ = self.geometry_generator(out_geometry, geometry_z, mask_)

            out_texture = self.backend_texture_generator(out, geometry_z.detach())
            out_texture = self.texture_generator(out_texture, texture_z)

            return out_geometry, out_texture


if __name__ == '__main__':
    model = backend_generator_halfsize_x8_small(g_dim=32, prob_dim=8, z_dim=8).cuda()
    # model = discriminator2d(d_dim=32, z_dim=17, d_in=4).cuda()
    print("Number of parameters: {:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
