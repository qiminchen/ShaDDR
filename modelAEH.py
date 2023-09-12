import os
import time
import math
import random
import h5py
import numpy as np
import cv2
import mcubes
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage.filters import gaussian_filter
from sklearn.manifold import TSNE
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from utils import *
from modelAEH_GD import *


class IM_AE(object):
    def __init__(self, config):
        # mask_margin is linear dependent to real_size
        self.real_size = config.output_size
        self.mask_margin = 16

        self.g_dim = 32
        self.d_dim = 32
        self.z_dim = 8
        self.param_alpha = config.alpha
        self.param_beta = config.beta

        self.input_size = config.input_size
        self.output_size = config.output_size

        self.train_geo = config.train_geo
        self.train_tex = config.train_tex

        if self.input_size == 64 and self.output_size == 512:
            self.upsample_rate = 8
        elif self.input_size == 32 and self.output_size == 256:
            self.upsample_rate = 8
        elif self.input_size == 16 and self.output_size == 256:
            self.upsample_rate = 16
        else:
            print("ERROR: invalid input/output size!")
            exit(-1)

        self.asymmetry = config.asymmetry

        self.save_epoch = 1

        self.sampling_threshold = 0.4

        self.render_view_id = 0
        if self.asymmetry:
            self.render_view_id = 6  # render side view for motorbike
        self.voxel_renderer = voxel_renderer(self.real_size)

        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.category = config.data_dir.split('/')[-2]

        self.data_style = config.data_style
        self.data_content = config.data_content

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # load data
        print("preprocessing - start")

        self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)

        if os.path.exists("splits/" + self.data_style + ".txt"):

            # load style data
            fin = open("splits/" + self.data_style + ".txt")
            self.styleset_names = [name.strip() for name in fin.readlines()]
            fin.close()
            self.styleset_len = len(self.styleset_names)
            self.voxel_style_lg = []  # geometry voxel for discriminator input &&& GT geometry voxel to compute recon loss
            self.voxel_style_sm = []  # geometry voxel for discriminator input &&& GT geometry voxel to compute recon loss
            self.Gmask_style = []   # mask for generator of geometry
            self.Dmask_style_lg = []  # mask for geometry discriminator output
            self.Dmask_style_sm = []  # mask for geometry discriminator output
            self.input_style = []  # coarse input for generator to compute recon loss
            self.pos_style = []    # for recovering voxel
            self.render_style = []  # rendered images for texture

            if config.train:
                for i in range(self.styleset_len):
                    print("preprocessing style - " + str(i + 1) + "/" + str(self.styleset_len) + " " + self.styleset_names[i])
                    voxel_path = os.path.join(self.data_dir, self.styleset_names[i] + "/model_depth_fusion.binvox")
                    color_path = os.path.join(self.data_dir, self.styleset_names[i] + "/voxel_color.hdf5")
                    if self.output_size == 128:
                        tmp_raw = get_vox_from_binvox_1over2(voxel_path).astype(np.uint8)
                    elif self.output_size == 256:
                        tmp_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
                    elif self.output_size == 512:
                        tmp_raw = get_vox_from_binvox_512(voxel_path).astype(np.uint8)
                    else:
                        raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")
                    xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
                    tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

                    self.voxel_style_lg.append(gaussian_filter(tmp.astype(np.float32), sigma=1))
                    tmp_sm = F.max_pool3d(torch.from_numpy(tmp.astype(np.float32)).unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2, padding=0).numpy()[0, 0]
                    self.voxel_style_sm.append(gaussian_filter(tmp_sm.astype(np.float32), sigma=1))
                    tmp_Dmask_lg, tmp_Dmask_sm = self.get_style_voxel_Dmask(tmp)
                    self.Dmask_style_lg.append(tmp_Dmask_lg)
                    self.Dmask_style_sm.append(tmp_Dmask_sm)

                    tmp_input, _, _, tmp_Gmask = self.get_voxel_input_Dmask_Gmask(tmp)
                    self.input_style.append(tmp_input)
                    self.Gmask_style.append(tmp_Gmask)
                    self.pos_style.append([xmin, xmax, ymin, ymax, zmin, zmax])

                    back, front, top, left, right = self.get_rendered_views(color_path, xmin, xmax, ymin, ymax, zmin, zmax)
                    self.render_style.append([back, front, top, left, right])

                    img_y = i // 4
                    img_x = (i % 4) * 2 + 1
                    if img_y < 4:
                        xmin, xmax, ymin, ymax, zmin, zmax = self.pos_style[-1]
                        tmpvox = self.recover_voxel(self.voxel_style_lg[-1], xmin, xmax, ymin, ymax, zmin, zmax)
                        self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                                      img_x * self.real_size:(img_x + 1) * self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold,
                                                                                                                            self.render_view_id)
                    img_y = i // 4
                    img_x = (i % 4) * 2
                    if img_y < 4:
                        tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                        xmin, xmax, ymin, ymax, zmin, zmax = self.pos_style[-1]
                        tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                        self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                        img_x * self.real_size:(img_x + 1) * self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold,
                                                                                                              self.render_view_id)
        else:
            raise FileNotFoundError("Cannot load style set txt: " + "splits/" + self.data_style + ".txt")
        if config.train:
            cv2.imwrite(config.sample_dir + "/a_style_0.png", self.imgout_0)

        self.imgout_0 = np.full([self.real_size * 4, self.real_size * 4 * 2], 255, np.uint8)

        if os.path.exists("splits/" + self.data_content + ".txt"):

            # load content data
            fin = open("splits/" + self.data_content + ".txt")
            self.dataset_names = [name.strip() for name in fin.readlines()]
            fin.close()
            self.dataset_len = len(self.dataset_names)
            self.Gmask_content = []  # mask for generator of geometry
            self.Dmask_content_lg = []  # mask for geometry discriminator output
            self.Dmask_content_sm = []  # mask for geometry discriminator output
            self.input_content = []  # coarse voxel input for generator
            self.pos_content = []  # for recovering voxel

            if config.train:
                for i in range(self.dataset_len):
                    print("preprocessing content - " + str(i + 1) + "/" + str(self.dataset_len))
                    voxel_path = os.path.join(self.data_dir, self.dataset_names[i] + "/model_depth_fusion.binvox")
                    if self.output_size == 128:
                        tmp_raw = get_vox_from_binvox_1over2(voxel_path).astype(np.uint8)
                    elif self.output_size == 256:
                        tmp_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
                    elif self.output_size == 512:
                        tmp_raw = get_vox_from_binvox_512(voxel_path).astype(np.uint8)
                    else:
                        raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")
                    xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
                    tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

                    tmp_input, tmp_Dmask_lg, tmp_Dmask_sm, tmp_Gmask = self.get_voxel_input_Dmask_Gmask(tmp)
                    self.input_content.append(tmp_input)
                    self.Dmask_content_lg.append(tmp_Dmask_lg)
                    self.Dmask_content_sm.append(tmp_Dmask_sm)
                    self.Gmask_content.append(tmp_Gmask)
                    self.pos_content.append([xmin, xmax, ymin, ymax, zmin, zmax])

                    img_y = i // 4
                    img_x = (i % 4) * 2
                    if img_y < 4:
                        tmp_mask_exact = self.get_voxel_mask_exact(tmp)
                        xmin, xmax, ymin, ymax, zmin, zmax = self.pos_content[i]
                        tmpvox = self.recover_voxel(tmp_mask_exact, xmin, xmax, ymin, ymax, zmin, zmax)
                        self.imgout_0[img_y * self.real_size:(img_y + 1) * self.real_size,
                                      img_x * self.real_size:(img_x + 1) * self.real_size] = self.voxel_renderer.render_img(tmpvox, self.sampling_threshold,
                                                                                                                            self.render_view_id)
        else:
            raise FileNotFoundError("Cannot load dataset txt: " + "splits/" + self.data_content + ".txt")

        if config.train:
            cv2.imwrite(config.sample_dir + "/a_content_0.png", self.imgout_0)

        # build model - generator
        if self.input_size == 64 and self.output_size == 512:
            self.generator = generator_dual(self.g_dim, self.styleset_len, self.z_dim)
        elif self.input_size == 32 and self.output_size == 256 and self.category == "03001627":
            self.generator = generator_dual_halfsize_x8_small(self.g_dim, self.styleset_len, self.z_dim)
        elif self.input_size == 32 and self.output_size == 256 and self.category == "00000000":
            self.generator = generator_dual_halfsize_x8(self.g_dim, self.styleset_len, self.z_dim)
        elif self.input_size == 32 and self.output_size == 256 and self.category == "03593526_03991062":
            self.generator = generator_dual_halfsize_x8_small_plant(self.g_dim, self.styleset_len, self.z_dim)
        elif self.input_size == 16 and self.output_size == 256:
            self.generator = generator_dual_halfsize_x16_small(self.g_dim, self.styleset_len, self.z_dim)
        self.generator.to(self.device)
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
        print("Generator number of parameters: {:,}".format(sum(p.numel() for p in self.generator.parameters() if p.requires_grad)))

        if self.train_geo:
            if self.input_size == 32 and self.output_size == 256:
                self.geometry_discriminator_rfl = discriminator_rf18(self.d_dim // 1, self.styleset_len + 1, d_in=1)
                self.geometry_discriminator_rfs = discriminator_rf18(self.d_dim // 1, self.styleset_len + 1, d_in=1)
            elif self.input_size == 64 and self.output_size == 512:
                self.geometry_discriminator_rfl = discriminator_rf36(self.d_dim // 2, self.styleset_len + 1, d_in=1)
                self.geometry_discriminator_rfs = discriminator_rf18(self.d_dim // 1, self.styleset_len + 1, d_in=1)
            elif self.input_size == 16 and self.output_size == 256:
                self.geometry_discriminator_rfl = discriminator_rf18(self.d_dim // 1, self.styleset_len + 1, d_in=1)
                self.geometry_discriminator_rfs = discriminator_rf18(self.d_dim // 1, self.styleset_len + 1, d_in=1)
            self.geometry_discriminator_rfl.to(self.device)
            self.geometry_discriminator_rfs.to(self.device)
            self.optimizer_d_geometry_rfl = torch.optim.Adam(self.geometry_discriminator_rfl.parameters(), lr=0.0001)
            self.optimizer_d_geometry_rfs = torch.optim.Adam(self.geometry_discriminator_rfs.parameters(), lr=0.0001)
            print("Geometry D rfl number of parameters: {:,}".format(sum(p.numel() for p in self.geometry_discriminator_rfl.parameters() if p.requires_grad)))
            print("Geometry D rfs number of parameters: {:,}".format(sum(p.numel() for p in self.geometry_discriminator_rfs.parameters() if p.requires_grad)))

        elif self.train_tex:
            if self.category == "02958343" and self.output_size == 512:
                d_dim, receptive_field = self.d_dim // 2, 36
            elif self.category == "02691156":
                d_dim, receptive_field = self.d_dim // 2, 11
            else:
                d_dim, receptive_field = self.d_dim // 1, 18
            self.texture_discriminator_back = discriminator2d(d_dim, self.styleset_len + 1, d_in=4, rf=receptive_field)
            self.texture_discriminator_front = discriminator2d(d_dim, self.styleset_len + 1, d_in=4, rf=receptive_field)
            self.texture_discriminator_top = discriminator2d(d_dim, self.styleset_len + 1, d_in=4, rf=receptive_field)
            self.texture_discriminator_side = discriminator2d(d_dim, self.styleset_len + 1, d_in=4, rf=receptive_field)
            self.texture_discriminator_right = discriminator2d(d_dim, self.styleset_len + 1, d_in=4, rf=receptive_field)
            self.texture_discriminator_back.to(self.device)
            self.texture_discriminator_front.to(self.device)
            self.texture_discriminator_top.to(self.device)
            self.texture_discriminator_side.to(self.device)
            self.texture_discriminator_right.to(self.device)
            self.optimizer_d_texture = torch.optim.Adam(list(self.texture_discriminator_back.parameters()) +
                                                        list(self.texture_discriminator_front.parameters()) +
                                                        list(self.texture_discriminator_top.parameters()) +
                                                        list(self.texture_discriminator_side.parameters()) +
                                                        list(self.texture_discriminator_right.parameters()), lr=0.0001)
            print("Textures Discriminator number of parameters: {:,}".format(sum(p.numel() for p in self.texture_discriminator_side.parameters() if p.requires_grad)))

        # pytorch does not have a checkpoint manager
        # have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 40
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name = 'IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

        if config.train:
            self.print_args(config)

    def get_rendered_views(self, vox_path, xmin, xmax, ymin, ymax, zmin, zmax):
        data_dict = h5py.File(vox_path, 'r')
        voxel_texture = data_dict["voxel_color"][:]
        data_dict.close()

        # the color is BGR, change to RGB
        geometry = voxel_texture[:, :, :, -1]
        texture = voxel_texture[:, :, :, :3]
        texture = texture[:, :, :, [2, 1, 0]]

        if self.output_size == 256:
            geometry = F.max_pool3d(torch.from_numpy(geometry).unsqueeze(0).unsqueeze(0).float(), kernel_size=2, stride=2, padding=0).numpy()[0, 0]
            texture = F.interpolate(torch.from_numpy(texture).permute(3, 0, 1, 2).unsqueeze(0).float(),
                                    scale_factor=0.5, mode='trilinear').squeeze(0).permute(1, 2, 3, 0).numpy()
            assert (texture >= 0).all()

        # crop voxel color same as geometry
        geometry_crop = self.crop_voxel(geometry, xmin, xmax, ymin, ymax, zmin, zmax)
        texture_crop = self.crop_color_voxel(texture, xmin, xmax, ymin, ymax, zmin, zmax)
        geometry_crop = torch.from_numpy(geometry_crop).to(self.device).unsqueeze(0).unsqueeze(0).float()
        texture_crop = torch.from_numpy(texture_crop).to(self.device).permute(3, 0, 1, 2).contiguous().unsqueeze(0).float() / 255.0

        back_texture, front_texture, top_texture, left_texture, right_texture = self.rendering(geometry_crop, texture_crop)

        # each (512, 512, 3+1)
        back_texture = back_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
        front_texture = front_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
        top_texture = top_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
        left_texture = left_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
        right_texture = right_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()

        return back_texture, front_texture, top_texture, left_texture, right_texture

    def rendering(self, geometry_tensor, texture_tensor):
        # back
        _, _, dim_x, dim_y, dim_z = geometry_tensor.size()
        back_mask, back_depth = torch.max(geometry_tensor[0, 0], 0)
        texture = torch.cat([torch.gather(texture_tensor[0, 0], 0, back_depth.unsqueeze(0)),
                             torch.gather(texture_tensor[0, 1], 0, back_depth.unsqueeze(0)),
                             torch.gather(texture_tensor[0, 2], 0, back_depth.unsqueeze(0))], 0)
        back_texture = texture * back_mask.unsqueeze(0)  # (3, 512, 512)

        # front
        front_mask, front_depth = torch.max(geometry_tensor[0, 0].flip(0), 0)
        texture = torch.cat([torch.gather(texture_tensor[0, 0], 0, dim_x - 1 - front_depth.unsqueeze(0)),
                             torch.gather(texture_tensor[0, 1], 0, dim_x - 1 - front_depth.unsqueeze(0)),
                             torch.gather(texture_tensor[0, 2], 0, dim_x - 1 - front_depth.unsqueeze(0))], 0)
        front_texture = texture * front_mask.unsqueeze(0)  # (3, 512, 512)

        # top
        top_mask, top_depth = torch.max(geometry_tensor[0, 0].flip(1), 1)
        texture = torch.cat([torch.gather(texture_tensor[0, 0], 1, dim_y - 1 - top_depth.unsqueeze(1)),
                             torch.gather(texture_tensor[0, 1], 1, dim_y - 1 - top_depth.unsqueeze(1)),
                             torch.gather(texture_tensor[0, 2], 1, dim_y - 1 - top_depth.unsqueeze(1))], 1)
        top_texture = texture.permute(1, 0, 2) * top_mask.unsqueeze(0)  # (512, 3, 512) -> (3, 512, 512)

        # side - left
        left_mask, left_depth = torch.max(geometry_tensor[0, 0].flip(2), 2)
        texture = torch.cat([torch.gather(texture_tensor[0, 0], 2, dim_z - 1 - left_depth.unsqueeze(2)),
                             torch.gather(texture_tensor[0, 1], 2, dim_z - 1 - left_depth.unsqueeze(2)),
                             torch.gather(texture_tensor[0, 2], 2, dim_z - 1 - left_depth.unsqueeze(2))], 2)
        left_texture = texture.permute(2, 0, 1) * left_mask.unsqueeze(0)  # (512, 512, 3) -> (3, 512, 512)
        # left_mask = self.fill_geometry_mask(left_mask, fill_x=True)

        # side - right, [only needed when asymmetry]
        right_mask, right_depth = torch.max(geometry_tensor[0, 0], 2)
        texture = torch.cat([torch.gather(texture_tensor[0, 0], 2, right_depth.unsqueeze(2)),
                             torch.gather(texture_tensor[0, 1], 2, right_depth.unsqueeze(2)),
                             torch.gather(texture_tensor[0, 2], 2, right_depth.unsqueeze(2))], 2)
        right_texture = texture.permute(2, 0, 1) * right_mask.unsqueeze(0)  # (512, 512, 3) -> (3, 512, 512)

        # each (1, 3+1, 512, 512)
        back_texture = torch.cat((back_texture, back_mask.unsqueeze(0)), dim=0).unsqueeze(0)
        front_texture = torch.cat((front_texture, front_mask.unsqueeze(0)), dim=0).unsqueeze(0)
        top_texture = torch.cat((top_texture, top_mask.unsqueeze(0)), dim=0).unsqueeze(0)
        left_texture = torch.cat((left_texture, left_mask.unsqueeze(0)), dim=0).unsqueeze(0)
        right_texture = torch.cat((right_texture, right_mask.unsqueeze(0)), dim=0).unsqueeze(0)

        return back_texture, front_texture, top_texture, left_texture, right_texture

    def get_image_Dmask_from_rendered_view(self, rendered_view):
        if self.upsample_rate == 8 and self.input_size == 32 and self.output_size == 256:
            # 256 -maxpoolk8s8- 32 -crop- 30 -upsample- 120
            # output: 120
            crop_margin = 1
            scale_factor = 2
            upsample_rate = self.upsample_rate
        elif self.upsample_rate == 8 and self.input_size == 64 and self.output_size == 512:
            # 512 -maxpoolk16s16- 64 -crop- 60 -upsample- 120
            # output: 120
            crop_margin = 2
            scale_factor = 4
            upsample_rate = self.upsample_rate
        elif self.upsample_rate == 16 and self.input_size == 16 and self.output_size == 256:
            # 512 -maxpoolk16s16- 64 -crop- 60 -upsample- 120
            # output: 120
            crop_margin = 1
            scale_factor = 2
            upsample_rate = self.upsample_rate // 2
        else:
            raise NotImplementedError("Upsample rate " + str(self.upsample_rate) + " not supported")

        # rendered_view is already a tensor (1, 1, 512, 512)
        smallmaskx_tensor = F.max_pool2d(rendered_view, kernel_size=upsample_rate, stride=upsample_rate, padding=0)
        smallmask_tensor = smallmaskx_tensor[:, :, crop_margin:-crop_margin, crop_margin:-crop_margin]
        smallmask_tensor = F.interpolate(smallmask_tensor, scale_factor=upsample_rate // scale_factor, mode='nearest')

        return smallmask_tensor

    def get_style_voxel_Dmask(self, vox):
        if self.upsample_rate == 8 and self.input_size == 32 and self.output_size == 256:
            # 256 -maxpoolk8s8- 32 -crop- 30 -upsample- 120
            # output: 56, 120
            crop_margin_1 = 1
            crop_margin_2 = 2
            scale_factor_1 = 2
            scale_factor_2 = 4
            upsample_rate = self.upsample_rate
        elif self.upsample_rate == 8 and self.input_size == 64 and self.output_size == 512:
            # 512 -maxpoolk16s16- 32 -crop- 30 -upsample- 120
            # output: 120, 120
            crop_margin_1 = 2
            crop_margin_2 = 2
            scale_factor_1 = 4
            scale_factor_2 = 4
            upsample_rate = self.upsample_rate
        elif self.upsample_rate == 16 and self.input_size == 16 and self.output_size == 256:
            # demo
            # 256 -maxpoolk8s8- 32 -crop- 30 -upsample- 120
            # output: 56, 120
            crop_margin_1 = 1
            crop_margin_2 = 2
            scale_factor_1 = 2
            scale_factor_2 = 4
            upsample_rate = self.upsample_rate // 2
        else:
            raise NotImplementedError("Upsample rate " + str(self.upsample_rate) + " not supported")

        # Dmask contains the whole voxel (surface + inside)
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()

        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=upsample_rate, stride=upsample_rate, padding=0)
        smallmask_tensor_rfl = smallmaskx_tensor[:, :, crop_margin_1:-crop_margin_1, crop_margin_1:-crop_margin_1, crop_margin_1:-crop_margin_1]
        if self.input_size == 32 and self.output_size == 256:
            smallmask_tensor_rfl = F.max_pool3d(smallmask_tensor_rfl, kernel_size=3, stride=1, padding=1)
        elif self.input_size == 16 and self.output_size == 256:
            smallmask_tensor_rfl = F.max_pool3d(smallmask_tensor_rfl, kernel_size=5, stride=1, padding=2)
        smallmask_tensor_rfl = F.interpolate(smallmask_tensor_rfl, scale_factor=upsample_rate // scale_factor_1, mode='nearest')

        smallmask_tensor_rfs = smallmaskx_tensor[:, :, crop_margin_2:-crop_margin_2, crop_margin_2:-crop_margin_2, crop_margin_2:-crop_margin_2]
        smallmask_tensor_rfs = F.interpolate(smallmask_tensor_rfs, scale_factor=upsample_rate // scale_factor_2, mode='nearest')

        smallmask_rfl = smallmask_tensor_rfl.detach().cpu().numpy()[0, 0]
        smallmask_rfs = smallmask_tensor_rfs.detach().cpu().numpy()[0, 0]
        smallmask_rfl = np.round(smallmask_rfl).astype(np.uint8)
        smallmask_rfs = np.round(smallmask_rfs).astype(np.uint8)

        return smallmask_rfl, smallmask_rfs

    def get_voxel_input_Dmask_Gmask(self, vox):
        if self.upsample_rate == 8 and self.input_size == 32 and self.output_size == 256:
            # 512 -maxpoolk8s8- 64 -crop- 60 -upsample- 120
            # output: 56, 120
            crop_margin_1 = 1
            crop_margin_2 = 2
            scale_factor_1 = 2
            scale_factor_2 = 4
            upsample_rate = self.upsample_rate
        elif self.upsample_rate == 8 and self.input_size == 64 and self.output_size == 512:
            # 512 -maxpoolk16s16- 32 -crop- 30 -upsample- 120
            # output: 120, 120
            crop_margin_1 = 2
            crop_margin_2 = 2
            scale_factor_1 = 4
            scale_factor_2 = 4
            upsample_rate = self.upsample_rate
        elif self.upsample_rate == 16 and self.input_size == 16 and self.output_size == 256:
            # demo
            # 256 -maxpoolk8s8- 32 -crop- 30 -upsample- 120
            # output: 56, 120
            crop_margin_1 = 1
            crop_margin_2 = 2
            scale_factor_1 = 2
            scale_factor_2 = 4
            upsample_rate = self.upsample_rate // 2
        else:
            raise NotImplementedError("Upsample rate " + str(self.upsample_rate) + " not supported")

        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        # input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=upsample_rate, stride=upsample_rate, padding=0)

        # Dmask contains the whole voxel (surface + inside)
        smallmask_tensor_rfl = smallmaskx_tensor[:, :, crop_margin_1:-crop_margin_1, crop_margin_1:-crop_margin_1, crop_margin_1:-crop_margin_1]
        if self.input_size == 32 and self.output_size == 256:
            smallmask_tensor_rfl = F.max_pool3d(smallmask_tensor_rfl, kernel_size=3, stride=1, padding=1)
        elif self.input_size == 16 and self.output_size == 256:
            smallmask_tensor_rfl = F.max_pool3d(smallmask_tensor_rfl, kernel_size=5, stride=1, padding=2)
        smallmask_tensor_rfl = F.interpolate(smallmask_tensor_rfl, scale_factor=upsample_rate // scale_factor_1, mode='nearest')

        smallmask_tensor_rfs = smallmaskx_tensor[:, :, crop_margin_2:-crop_margin_2, crop_margin_2:-crop_margin_2, crop_margin_2:-crop_margin_2]
        smallmask_tensor_rfs = F.interpolate(smallmask_tensor_rfs, scale_factor=upsample_rate // scale_factor_2, mode='nearest')

        # Gmask
        # expand 1
        if self.upsample_rate == 8:
            mask_tensor = smallmaskx_tensor
        elif self.upsample_rate == 16:
            mask_tensor = smallmaskx_tensor
        else:
            raise NotImplementedError("Upsample rate " + str(self.upsample_rate) + " not supported")
        mask_tensor = F.max_pool3d(mask_tensor, kernel_size=3, stride=1, padding=1)

        # to numpy
        if self.upsample_rate == 16 and self.input_size == 16 and self.output_size == 256:
            smallmaskx_tensor = F.max_pool3d(smallmaskx_tensor, kernel_size=2, stride=2, padding=0)
        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0, 0]
        smallmask_rfl = smallmask_tensor_rfl.detach().cpu().numpy()[0, 0]
        smallmask_rfs = smallmask_tensor_rfs.detach().cpu().numpy()[0, 0]
        mask = mask_tensor.detach().cpu().numpy()[0, 0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        smallmask_rfl = np.round(smallmask_rfl).astype(np.uint8)
        smallmask_rfs = np.round(smallmask_rfs).astype(np.uint8)
        mask = np.round(mask).astype(np.uint8)

        return smallmaskx, smallmask_rfl, smallmask_rfs, mask

    def get_voxel_bbox(self, vox):
        # minimap
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
        smallmaskx = smallmaskx_tensor.detach().cpu().numpy()[0, 0]
        smallmaskx = np.round(smallmaskx).astype(np.uint8)
        smallx, smally, smallz = smallmaskx.shape
        # x
        ray = np.max(smallmaskx, (1, 2))
        xmin = 0
        xmax = 0
        for i in range(smallx):
            if ray[i] > 0:
                if xmin == 0:
                    xmin = i
                xmax = i
        # y
        ray = np.max(smallmaskx, (0, 2))
        ymin = 0
        ymax = 0
        for i in range(smally):
            if ray[i] > 0:
                if ymin == 0:
                    ymin = i
                ymax = i
        # z
        ray = np.max(smallmaskx, (0, 1))
        if self.asymmetry:
            zmin = 0
            zmax = 0
            for i in range(smallz):
                if ray[i] > 0:
                    if zmin == 0:
                        zmin = i
                    zmax = i
        else:
            zmin = smallz // 2
            zmax = 0
            for i in range(zmin, smallz):
                if ray[i] > 0:
                    zmax = i

        return xmin, xmax + 1, ymin, ymax + 1, zmin, zmax + 1

    def get_voxel_mask_exact(self, vox):
        # 512 -maxpoolk8s8- 64 -upsample- 512
        vox_tensor = torch.from_numpy(vox).to(self.device).unsqueeze(0).unsqueeze(0).float()
        # input
        smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
        # mask
        smallmask_tensor = F.interpolate(smallmaskx_tensor, scale_factor=self.upsample_rate, mode='nearest')
        # to numpy
        smallmask = smallmask_tensor.detach().cpu().numpy()[0, 0]
        smallmask = np.round(smallmask).astype(np.uint8)

        return smallmask

    def crop_voxel(self, vox, xmin, xmax, ymin, ymax, zmin, zmax):
        xspan = xmax - xmin
        yspan = ymax - ymin
        zspan = zmax - zmin
        tmp = np.zeros([xspan * self.upsample_rate + self.mask_margin * 2, yspan * self.upsample_rate + self.mask_margin * 2,
                        zspan * self.upsample_rate + self.mask_margin * 2], np.uint8)
        if self.asymmetry:
            tmp[self.mask_margin:-self.mask_margin,
                self.mask_margin:-self.mask_margin,
                self.mask_margin:-self.mask_margin] = vox[xmin * self.upsample_rate:xmax * self.upsample_rate,
                                                          ymin * self.upsample_rate:ymax * self.upsample_rate,
                                                          zmin * self.upsample_rate:zmax * self.upsample_rate]
        else:
            # note z is special: only get half of the shape in z:  0     0.5-----1
            tmp[self.mask_margin:-self.mask_margin,
                self.mask_margin:-self.mask_margin,
                :-self.mask_margin] = vox[xmin * self.upsample_rate:xmax * self.upsample_rate,
                                          ymin * self.upsample_rate:ymax * self.upsample_rate,
                                          zmin * self.upsample_rate - self.mask_margin:zmax * self.upsample_rate]
        return tmp

    def crop_color_voxel(self, color_vox, xmin, xmax, ymin, ymax, zmin, zmax):
        xspan = xmax - xmin
        yspan = ymax - ymin
        zspan = zmax - zmin
        tmp = np.zeros([xspan * self.upsample_rate + self.mask_margin * 2, yspan * self.upsample_rate + self.mask_margin * 2,
                        zspan * self.upsample_rate + self.mask_margin * 2, 3], np.float32)
        if self.asymmetry:
            tmp[self.mask_margin:-self.mask_margin,
                self.mask_margin:-self.mask_margin,
                self.mask_margin:-self.mask_margin, :] = color_vox[xmin * self.upsample_rate:xmax * self.upsample_rate,
                                                                   ymin * self.upsample_rate:ymax * self.upsample_rate,
                                                                   zmin * self.upsample_rate:zmax * self.upsample_rate, :]
        else:
            # note z is special: only get half of the shape in z:  0     0.5-----1
            tmp[self.mask_margin:-self.mask_margin,
                self.mask_margin:-self.mask_margin,
                :-self.mask_margin, :] = color_vox[xmin * self.upsample_rate:xmax * self.upsample_rate,
                                                   ymin * self.upsample_rate:ymax * self.upsample_rate,
                                                   zmin * self.upsample_rate - self.mask_margin:zmax * self.upsample_rate, :]

        return tmp

    def recover_voxel(self, vox, xmin, xmax, ymin, ymax, zmin, zmax):
        tmpvox = np.zeros([self.real_size, self.real_size, self.real_size], np.float32)
        xmin_, ymin_, zmin_ = (0, 0, 0)
        xmax_, ymax_, zmax_ = vox.shape
        xmin = xmin * self.upsample_rate - self.mask_margin
        xmax = xmax * self.upsample_rate + self.mask_margin
        ymin = ymin * self.upsample_rate - self.mask_margin
        ymax = ymax * self.upsample_rate + self.mask_margin
        if self.asymmetry:
            zmin = zmin * self.upsample_rate - self.mask_margin
        else:
            zmin = zmin * self.upsample_rate
            zmin_ = self.mask_margin
        zmax = zmax * self.upsample_rate + self.mask_margin
        if xmin < 0:
            xmin_ = -xmin
            xmin = 0
        if xmax > self.real_size:
            xmax_ = xmax_ + self.real_size - xmax
            xmax = self.real_size
        if ymin < 0:
            ymin_ = -ymin
            ymin = 0
        if ymax > self.real_size:
            ymax_ = ymax_ + self.real_size - ymax
            ymax = self.real_size
        if zmin < 0:
            zmin_ = -zmin
            zmin = 0
        if zmax > self.real_size:
            zmax_ = zmax_ + self.real_size - zmax
            zmax = self.real_size
        if self.asymmetry:
            tmpvox[xmin:xmax, ymin:ymax, zmin:zmax] = vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_]
        else:
            tmpvox[xmin:xmax, ymin:ymax, zmin:zmax] = vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_]
            if zmin * 2 - zmax - 1 < 0:
                tmpvox[xmin:xmax, ymin:ymax, zmin - 1::-1] = vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_]
            else:
                tmpvox[xmin:xmax, ymin:ymax, zmin - 1:zmin * 2 - zmax - 1:-1] = vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_]

        return tmpvox

    def recover_color_voxel(self, color_vox, xmin, xmax, ymin, ymax, zmin, zmax):
        tmpvox = np.zeros([self.real_size, self.real_size, self.real_size, 3], np.float32)
        xmin_, ymin_, zmin_ = (0, 0, 0)
        xmax_, ymax_, zmax_, _ = color_vox.shape
        xmin = xmin * self.upsample_rate - self.mask_margin
        xmax = xmax * self.upsample_rate + self.mask_margin
        ymin = ymin * self.upsample_rate - self.mask_margin
        ymax = ymax * self.upsample_rate + self.mask_margin
        if self.asymmetry:
            zmin = zmin * self.upsample_rate - self.mask_margin
        else:
            zmin = zmin * self.upsample_rate
            zmin_ = self.mask_margin
        zmax = zmax * self.upsample_rate + self.mask_margin
        if xmin < 0:
            xmin_ = -xmin
            xmin = 0
        if xmax > self.real_size:
            xmax_ = xmax_ + self.real_size - xmax
            xmax = self.real_size
        if ymin < 0:
            ymin_ = -ymin
            ymin = 0
        if ymax > self.real_size:
            ymax_ = ymax_ + self.real_size - ymax
            ymax = self.real_size
        if zmin < 0:
            zmin_ = -zmin
            zmin = 0
        if zmax > self.real_size:
            zmax_ = zmax_ + self.real_size - zmax
            zmax = self.real_size
        if self.asymmetry:
            tmpvox[xmin:xmax, ymin:ymax, zmin:zmax, :] = color_vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_, :]
        else:
            tmpvox[xmin:xmax, ymin:ymax, zmin:zmax, :] = color_vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_, :]
            if zmin * 2 - zmax - 1 < 0:
                tmpvox[xmin:xmax, ymin:ymax, zmin - 1::-1, :] = color_vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_, :]
            else:
                tmpvox[xmin:xmax, ymin:ymax, zmin - 1:zmin * 2 - zmax - 1:-1, :] = color_vox[xmin_:xmax_, ymin_:ymax_, zmin_:zmax_, :]

        return tmpvox

    def recover_texture_image(self, img, xmin, xmax, ymin, ymax, asymmetry=False):
        tmpimg = np.zeros([self.real_size, self.real_size, img.shape[-1]], np.float32)
        xmin_, ymin_ = (0, 0)
        xmax_, ymax_ = img.shape[:2]
        xmin = xmin * self.upsample_rate - self.mask_margin
        xmax = xmax * self.upsample_rate + self.mask_margin
        if asymmetry:
            ymin = ymin * self.upsample_rate - self.mask_margin
        else:
            ymin = ymin * self.upsample_rate
            ymin_ = self.mask_margin
        ymax = ymax * self.upsample_rate + self.mask_margin
        if xmin < 0:
            xmin_ = -xmin
            xmin = 0
        if xmax > self.real_size:
            xmax_ = xmax_ + self.real_size - xmax
            xmax = self.real_size
        if ymin < 0:
            ymin_ = -ymin
            ymin = 0
        if ymax > self.real_size:
            ymax_ = ymax_ + self.real_size - ymax
            ymax = self.real_size
        if asymmetry:
            tmpimg[xmin:xmax, ymin:ymax] = img[xmin_:xmax_, ymin_:ymax_]
        else:
            tmpimg[xmin:xmax, ymin:ymax] = img[xmin_:xmax_, ymin_:ymax_]
            if ymin * 2 - ymax - 1 < 0:
                tmpimg[xmin:xmax, ymin - 1::-1] = img[xmin_:xmax_, ymin_:ymax_]
            else:
                tmpimg[xmin:xmax, ymin - 1:ymin * 2 - ymax - 1:-1] = img[xmin_:xmax_, ymin_:ymax_]

        return tmpimg

    @staticmethod
    def fill_geometry_mask(image, fill_x=False, fill_y=False, fill_xy=False):
        # loop-free, faster
        if fill_xy:
            left = image.flip(1).cumsum(1).flip(1)
            right = image.cumsum(1)
            tmpimg = ((left * right + image) > 0).float()
            left = tmpimg.flip(0).cumsum(0).flip(0)
            right = tmpimg.cumsum(0)
            tmpimg = ((left * right + tmpimg) > 0).float()
        elif fill_x:
            left = image.flip(1).cumsum(1).flip(1)
            right = image.cumsum(1)
            tmpimg = ((left * right + image) > 0).float()
        else:
            assert fill_y
            left = image.flip(0).cumsum(0).flip(0)
            right = image.cumsum(0)
            tmpimg = ((left * right + image) > 0).float()

        return tmpimg

    def load(self):
        # load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            checkpoint = torch.load(model_dir)
            self.generator.load_state_dict(checkpoint['generator'])
            print(" [{}] Load SUCCESS".format(model_dir))
            return True
        else:
            print(" [!] Load failed...")
            return False

    def load_pretrained_geometry(self):
        # load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            checkpoint = torch.load(model_dir)
            model_dict = self.generator.state_dict()
            matched_checkpoint_dict = {k: v for k, v in checkpoint['generator'].items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(matched_checkpoint_dict)
            self.generator.load_state_dict(model_dict)
            print(" [{}] Load SUCCESS".format(model_dir))
            return True
        else:
            print(" [!] Load failed...".format(checkpoint_txt))
            return False

    def save(self, epoch):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        save_dir = os.path.join(self.checkpoint_path, self.checkpoint_name + "-" + str(epoch) + ".pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer + 1) % self.max_to_keep

        # delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])

        # save checkpoint
        if self.train_geo:
            torch.save({
                'generator': self.generator.state_dict(),
                'geometry_discriminator_rfl': self.geometry_discriminator_rfl.state_dict(),
                'geometry_discriminator_rfs': self.geometry_discriminator_rfs.state_dict(),
            }, save_dir)
        elif self.train_tex:
            torch.save({
                'generator': self.generator.state_dict(),
                'texture_discriminator_back': self.texture_discriminator_back.state_dict(),
                'texture_discriminator_front': self.texture_discriminator_front.state_dict(),
                'texture_discriminator_top': self.texture_discriminator_top.state_dict(),
                'texture_discriminator_side': self.texture_discriminator_side.state_dict(),
                'texture_discriminator_right': self.texture_discriminator_right.state_dict(),
            }, save_dir)

        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir

        # write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer + self.max_to_keep - i) % self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer] + "\n")
        fout.close()

    @property
    def model_dir(self):
        return "{}_aemr".format(self.data_style)

    def print_args(self, config):
        print("=======================================================")
        print("[asymmetry]:       ", self.asymmetry)
        print("[style]:           ", self.data_style)
        print("[content]:         ", self.data_content)
        print("[alpha / beta]:    ", self.param_alpha, self.param_beta)
        print("[in / out size]:   ", self.input_size, self.output_size)
        print("[upsample rate]:   ", self.upsample_rate)
        print("[sample dir]:      ", config.sample_dir)
        print("[checkpoint path]: ", self.checkpoint_path)
        print("=======================================================")

    def train_geometry(self, config):

        print("Start training geometry generation...")

        # self.load()

        start_time = time.time()
        training_epoch = config.epoch

        batch_index_list = np.arange(self.dataset_len)
        iter_counter = 0

        for epoch in range(0, training_epoch):
            np.random.shuffle(batch_index_list)

            self.geometry_discriminator_rfl.train()
            self.geometry_discriminator_rfs.train()
            self.generator.train()

            # geometry_style_idx_for_test = 0

            for idx in range(self.dataset_len):
                # random a z vector for D training
                z_vector_geometry = np.zeros([self.styleset_len], np.float32)
                z_vector_geometry_idx = np.random.randint(self.styleset_len)
                z_vector_geometry[z_vector_geometry_idx] = 1
                z_geometry_tensor = torch.from_numpy(z_vector_geometry).to(self.device).view([1, -1])

                # ready a fake voxel
                dxb = batch_index_list[idx]
                Gmask_fake = torch.from_numpy(self.Gmask_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                Dmask_fake_lg = torch.from_numpy(self.Dmask_content_lg[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                Dmask_fake_sm = torch.from_numpy(self.Dmask_content_sm[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(self.input_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                z_geometry_code = torch.matmul(z_geometry_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                voxel_fake_lg, voxel_fake_sm = self.generator(input_fake, z_geometry_code, None, Gmask_fake, is_geometry_training=True)
                voxel_fake_lg = voxel_fake_lg.detach()
                voxel_fake_sm = voxel_fake_sm.detach()

                # D step
                d_steps = 1
                for d_step in range(d_steps):
                    qxp = z_vector_geometry_idx

                    self.geometry_discriminator_rfl.zero_grad()
                    self.geometry_discriminator_rfs.zero_grad()

                    voxel_style_lg = torch.from_numpy(self.voxel_style_lg[qxp]).to(self.device).unsqueeze(0).unsqueeze(0)
                    voxel_style_sm = torch.from_numpy(self.voxel_style_sm[qxp]).to(self.device).unsqueeze(0).unsqueeze(0)
                    Dmask_style_lg = torch.from_numpy(self.Dmask_style_lg[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    Dmask_style_sm = torch.from_numpy(self.Dmask_style_sm[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()

                    # 512/256
                    D_out = self.geometry_discriminator_rfl(voxel_style_lg, is_training=True)
                    loss_d_real_lg = (torch.sum((D_out[:, z_vector_geometry_idx:z_vector_geometry_idx + 1] - 1) ** 2 * Dmask_style_lg) +
                                       torch.sum((D_out[:, -1:] - 1) ** 2 * Dmask_style_lg)) / torch.sum(Dmask_style_lg)
                    loss_d_real_lg.backward()

                    D_out = self.geometry_discriminator_rfl(voxel_fake_lg, is_training=True)
                    loss_d_fake_lg = (torch.sum((D_out[:, z_vector_geometry_idx:z_vector_geometry_idx + 1]) ** 2 * Dmask_fake_lg) +
                                       torch.sum((D_out[:, -1:]) ** 2 * Dmask_fake_lg)) / torch.sum(Dmask_fake_lg)
                    loss_d_fake_lg.backward()

                    self.optimizer_d_geometry_rfl.step()
                    self.geometry_discriminator_rfl.zero_grad()

                    # 256/128
                    D_out = self.geometry_discriminator_rfs(voxel_style_sm, is_training=True)
                    loss_d_real_sm = (torch.sum((D_out[:, z_vector_geometry_idx:z_vector_geometry_idx + 1] - 1) ** 2 * Dmask_style_sm) +
                                       torch.sum((D_out[:, -1:] - 1) ** 2 * Dmask_style_sm)) / torch.sum(Dmask_style_sm)
                    loss_d_real_sm.backward()

                    D_out = self.geometry_discriminator_rfs(voxel_fake_sm, is_training=True)
                    loss_d_fake_sm = (torch.sum((D_out[:, z_vector_geometry_idx:z_vector_geometry_idx + 1]) ** 2 * Dmask_fake_sm) +
                                       torch.sum((D_out[:, -1:]) ** 2 * Dmask_fake_sm)) / torch.sum(Dmask_fake_sm)
                    loss_d_fake_sm.backward()

                    self.optimizer_d_geometry_rfs.step()
                    self.geometry_discriminator_rfs.zero_grad()

                # recon step
                # reconstruct style image
                r_steps = 4 if iter_counter < 5000 else 1
                iter_counter += 1
                for r_step in range(r_steps):
                    qxp = np.random.randint(self.styleset_len)

                    z_vector_geometry_recon = np.zeros([self.styleset_len], np.float32)
                    z_vector_geometry_recon[qxp] = 1
                    z_geometry_recon_tensor = torch.from_numpy(z_vector_geometry_recon).to(self.device).view([1, -1])

                    voxel_style_lg = torch.from_numpy(self.voxel_style_lg[qxp]).to(self.device).unsqueeze(0).unsqueeze(0)
                    voxel_style_sm = torch.from_numpy(self.voxel_style_sm[qxp]).to(self.device).unsqueeze(0).unsqueeze(0)
                    Gmask_style = torch.from_numpy(self.Gmask_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    input_style = torch.from_numpy(self.input_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()

                    self.generator.zero_grad()

                    z_geometry_recon_code = torch.matmul(z_geometry_recon_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                    voxel_fake_lg, voxel_fake_sm = self.generator(input_style, z_geometry_recon_code, None, Gmask_style, is_geometry_training=True)

                    loss_r_lg = torch.mean((voxel_style_lg - voxel_fake_lg) ** 2) * self.param_beta
                    loss_r_sm = torch.mean((voxel_style_sm - voxel_fake_sm) ** 2) * self.param_beta
                    loss_r = loss_r_lg + loss_r_sm
                    loss_r.backward()
                    self.optimizer_g.step()
                    self.generator.zero_grad()

                # G step
                g_steps = 1
                for step in range(g_steps):
                    self.generator.zero_grad()

                    z_geometry_code = torch.matmul(z_geometry_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                    voxel_fake_lg, voxel_fake_sm = self.generator(input_fake, z_geometry_code, None, Gmask_fake, is_geometry_training=True)

                    # 512/256
                    D_out_lg = self.geometry_discriminator_rfl(voxel_fake_lg, is_training=False)

                    loss_g_lg = (torch.sum((D_out_lg[:, z_vector_geometry_idx:z_vector_geometry_idx + 1] - 1) ** 2 * Dmask_fake_lg) * self.param_alpha +
                                  torch.sum((D_out_lg[:, -1:] - 1) ** 2 * Dmask_fake_lg)) / torch.sum(Dmask_fake_lg)
                    # 256/128
                    D_out_sm = self.geometry_discriminator_rfs(voxel_fake_sm, is_training=False)
                    loss_g_sm = (torch.sum((D_out_sm[:, z_vector_geometry_idx:z_vector_geometry_idx + 1] - 1) ** 2 * Dmask_fake_sm) * self.param_alpha +
                                  torch.sum((D_out_sm[:, -1:] - 1) ** 2 * Dmask_fake_sm)) / torch.sum(Dmask_fake_sm)

                    # param_gamma, param_delta = (0.5, 1) if epoch < 5 else (0.8, 1)
                    param_gamma, param_delta = 1, 0.1
                    loss_g = loss_g_lg * param_gamma + loss_g_sm * param_delta
                    loss_g.backward()
                    self.optimizer_g.step()
                    self.generator.zero_grad()

                if epoch % 1 == 0 and (idx + 1) % (self.dataset_len // 4) == 0:

                    geometry_voxel_fake = voxel_fake_lg[0, 0].detach().cpu().numpy()
                    xmin, xmax, ymin, ymax, zmin, zmax = self.pos_content[dxb]
                    geometry_voxel = self.recover_voxel(geometry_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                    vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                    vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                    write_ply_triangle(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_geometry_lg.ply", vertices, triangles)

                    # geometry_voxel_fake = F.interpolate(voxel_fake_sm, scale_factor=2, mode='nearest')[0, 0].detach().cpu().numpy()
                    # geometry_voxel = self.recover_voxel(geometry_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                    # vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                    # vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                    # write_ply_triangle(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_geometry_sm.ply", vertices, triangles)

                    del voxel_fake_lg
                    del voxel_fake_sm

            print("Epoch: [%d/%d] time: %.0f, d_real_lg: %.4f, d_fake_lg: %.4f, loss_r_lg: %.4f, loss_g_lg: %.4f | "
                  "d_real_sm: %.4f, d_fake_sm: %.4f, loss_r_sm: %.4f, loss_g_sm: %.4f " % (
                   epoch, training_epoch, time.time() - start_time, loss_d_real_lg.item(), loss_d_fake_lg.item(), loss_r_lg.item(), loss_g_lg.item(),
                   loss_d_real_sm.item(), loss_d_fake_sm.item(), loss_r_sm.item(), loss_g_sm.item()))

            if epoch % self.save_epoch == 0:
                self.save(epoch)

        # if finish, save
        self.save(epoch)

    def test_geometry(self, config):

        if not self.load_pretrained_geometry():
            exit(-1)

        max_num_of_contents = 10
        max_num_of_styles = 64

        fin = open("splits/" + self.data_content + ".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        print("testing {} contents with {} styles...".format(self.dataset_len, self.styleset_len))

        for i in range(self.dataset_len):

            print(i, self.dataset_names[i])

            voxel_path = os.path.join(self.data_dir, self.dataset_names[i] + "/model_depth_fusion.binvox")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(voxel_path).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
            elif self.output_size == 512:
                tmp_raw = get_vox_from_binvox_512(voxel_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            tmp_input, _, _, tmp_Gmask = self.get_voxel_input_Dmask_Gmask(tmp)
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Gmask_fake = torch.from_numpy(tmp_Gmask).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for j in range(self.styleset_len):

                save_dir = os.path.join(config.sample_dir, self.dataset_names[i], self.styleset_names[j])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                with torch.no_grad():
                    z_vector_geometry = np.zeros([self.styleset_len], np.float32)
                    z_vector_geometry[j] = 1
                    z_geometry_tensor = torch.from_numpy(z_vector_geometry).to(self.device).view([1, -1])
                    z_tensor_g = torch.matmul(z_geometry_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                    voxel_fake_lg, voxel_fake_sm = self.generator(input_fake, z_tensor_g, None, Gmask_fake, is_geometry_training=True)

                geometry_voxel_fake = voxel_fake_lg[0, 0].detach().cpu().numpy()
                geometry_voxel_fake = gaussian_filter(geometry_voxel_fake, sigma=1.0)
                geometry_voxel = self.recover_voxel(geometry_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                write_ply_triangle(save_dir + "/" + "geometry_lg.ply", vertices, triangles)

    def train_texture(self, config):

        print("Start training texture symmetry...")

        if not self.load_pretrained_geometry():
            exit(-1)

        start_time = time.time()
        training_epoch = config.epoch

        batch_index_list = np.arange(self.dataset_len)
        iter_counter = 0

        for epoch in range(0, training_epoch):
            np.random.shuffle(batch_index_list)

            self.texture_discriminator_back.train()
            self.texture_discriminator_front.train()
            self.texture_discriminator_top.train()
            self.texture_discriminator_side.train()
            self.generator.train()

            for idx in range(self.dataset_len):
                # random a z vector for D training
                z_vector_geometry = np.zeros([self.styleset_len], np.float32)
                z_vector_geometry_idx = np.random.randint(self.styleset_len)
                z_vector_geometry[z_vector_geometry_idx] = 1
                z_geometry_tensor = torch.from_numpy(z_vector_geometry).to(self.device).view([1, -1])

                z_vector_texture = np.zeros([self.styleset_len], np.float32)
                # z_vector_texture_idx = np.random.randint(self.styleset_len)
                z_vector_texture_idx = z_vector_geometry_idx
                z_vector_texture[z_vector_texture_idx] = 1
                z_texture_tensor = torch.from_numpy(z_vector_texture).to(self.device).view([1, -1])

                # ready a fake voxel
                dxb = batch_index_list[idx]
                Gmask_fake = torch.from_numpy(self.Gmask_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(self.input_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                z_geometry_code = torch.matmul(z_geometry_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                z_texture_code = torch.matmul(z_texture_tensor, self.generator.texture_codes).view([1, -1, 1, 1, 1])
                voxel_fake, texture_fake = self.generator(input_fake, z_geometry_code, z_texture_code, Gmask_fake, is_geometry_training=False)
                voxel_fake = voxel_fake.detach()
                texture_fake = texture_fake.detach()
                back_fake, front_fake, top_fake, side_fake, _ = self.rendering((voxel_fake > self.sampling_threshold).float(), texture_fake)
                Dmask_fake_back = self.get_image_Dmask_from_rendered_view(back_fake[:, 3:, :, :])
                Dmask_fake_front = self.get_image_Dmask_from_rendered_view(front_fake[:, 3:, :, :])
                Dmask_fake_top = self.get_image_Dmask_from_rendered_view(top_fake[:, 3:, :, :])
                Dmask_fake_side = self.get_image_Dmask_from_rendered_view(side_fake[:, 3:, :, :])

                # D step
                d_steps = 1
                for d_step in range(d_steps):
                    qxp = z_vector_texture_idx

                    self.texture_discriminator_back.zero_grad()
                    self.texture_discriminator_front.zero_grad()
                    self.texture_discriminator_top.zero_grad()
                    self.texture_discriminator_side.zero_grad()

                    back_view = torch.from_numpy(self.render_style[qxp][0]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    front_view = torch.from_numpy(self.render_style[qxp][1]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    top_view = torch.from_numpy(self.render_style[qxp][2]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    side_view = torch.from_numpy(self.render_style[qxp][3]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    Dmask_style_back = self.get_image_Dmask_from_rendered_view(back_view[:, 3:, :, :])
                    Dmask_style_front = self.get_image_Dmask_from_rendered_view(front_view[:, 3:, :, :])
                    Dmask_style_top = self.get_image_Dmask_from_rendered_view(top_view[:, 3:, :, :])
                    Dmask_style_side = self.get_image_Dmask_from_rendered_view(side_view[:, 3:, :, :])

                    # train real images
                    D_out_back = self.texture_discriminator_back(back_view, is_training=True)
                    D_out_front = self.texture_discriminator_front(front_view, is_training=True)
                    D_out_top = self.texture_discriminator_top(top_view, is_training=True)
                    D_out_side = self.texture_discriminator_side(side_view, is_training=True)

                    loss_d_real_back = (torch.sum((D_out_back[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_style_back) +
                                        torch.sum((D_out_back[:, -1:] - 1) ** 2 * Dmask_style_back)) / torch.sum(Dmask_style_back)
                    loss_d_real_front = (torch.sum((D_out_front[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_style_front) +
                                         torch.sum((D_out_front[:, -1:] - 1) ** 2 * Dmask_style_front)) / torch.sum(Dmask_style_front)
                    loss_d_real_top = (torch.sum((D_out_top[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_style_top) +
                                       torch.sum((D_out_top[:, -1:] - 1) ** 2 * Dmask_style_top)) / torch.sum(Dmask_style_top)
                    loss_d_real_side = (torch.sum((D_out_side[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_style_side) +
                                        torch.sum((D_out_side[:, -1:] - 1) ** 2 * Dmask_style_side)) / torch.sum(Dmask_style_side)

                    loss_d_real = loss_d_real_back + loss_d_real_front + loss_d_real_top + loss_d_real_side
                    loss_d_real.backward()

                    # train fake images
                    D_out_back = self.texture_discriminator_back(back_fake, is_training=True)
                    D_out_front = self.texture_discriminator_front(front_fake, is_training=True)
                    D_out_top = self.texture_discriminator_top(top_fake, is_training=True)
                    D_out_side = self.texture_discriminator_side(side_fake, is_training=True)

                    loss_d_fake_back = (torch.sum((D_out_back[:, z_vector_texture_idx:z_vector_texture_idx + 1]) ** 2 * Dmask_fake_back) +
                                        torch.sum((D_out_back[:, -1:]) ** 2 * Dmask_fake_back)) / torch.sum(Dmask_fake_back)
                    loss_d_fake_front = (torch.sum((D_out_front[:, z_vector_texture_idx:z_vector_texture_idx + 1]) ** 2 * Dmask_fake_front) +
                                         torch.sum((D_out_front[:, -1:]) ** 2 * Dmask_fake_front)) / torch.sum(Dmask_fake_front)
                    loss_d_fake_top = (torch.sum((D_out_top[:, z_vector_texture_idx:z_vector_texture_idx + 1]) ** 2 * Dmask_fake_top) +
                                       torch.sum((D_out_top[:, -1:]) ** 2 * Dmask_fake_top)) / torch.sum(Dmask_fake_top)
                    loss_d_fake_side = (torch.sum((D_out_side[:, z_vector_texture_idx:z_vector_texture_idx + 1]) ** 2 * Dmask_fake_side) +
                                        torch.sum((D_out_side[:, -1:]) ** 2 * Dmask_fake_side)) / torch.sum(Dmask_fake_side)

                    loss_d_fake = loss_d_fake_back + loss_d_fake_front + loss_d_fake_top + loss_d_fake_side
                    loss_d_fake.backward()

                    self.optimizer_d_texture.step()

                # recon step
                # reconstruct style image
                r_steps = 4 if iter_counter < 5000 else 1
                iter_counter += 1
                for r_step in range(r_steps):
                    qxp = np.random.randint(self.styleset_len)

                    z_vector_geometry_recon = np.zeros([self.styleset_len], np.float32)
                    z_vector_geometry_recon[qxp] = 1
                    z_geometry_recon_tensor = torch.from_numpy(z_vector_geometry_recon).to(self.device).view([1, -1])

                    z_vector_texture_recon = np.zeros([self.styleset_len], np.float32)
                    z_vector_texture_recon[qxp] = 1
                    z_texture_recon_tensor = torch.from_numpy(z_vector_texture_recon).to(self.device).view([1, -1])

                    Gmask_style = torch.from_numpy(self.Gmask_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    input_style = torch.from_numpy(self.input_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    back_view = torch.from_numpy(self.render_style[qxp][0]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    front_view = torch.from_numpy(self.render_style[qxp][1]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    top_view = torch.from_numpy(self.render_style[qxp][2]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    side_view = torch.from_numpy(self.render_style[qxp][3]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()

                    self.generator.zero_grad()

                    z_geometry_recon_code = torch.matmul(z_geometry_recon_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                    z_texture_recon_code = torch.matmul(z_texture_recon_tensor, self.generator.texture_codes).view([1, -1, 1, 1, 1])
                    voxel_fake, texture_fake = self.generator(input_style, z_geometry_recon_code, z_texture_recon_code, Gmask_style, is_geometry_training=False)

                    back_rendered, front_rendered, top_rendered, side_rendered, _ = self.rendering((voxel_fake > self.sampling_threshold).float(), texture_fake)
                    loss_r_back = torch.mean((back_view - back_rendered) ** 2) * self.param_beta
                    loss_r_front = torch.mean((front_view - front_rendered) ** 2) * self.param_beta
                    loss_r_top = torch.mean((top_view - top_rendered) ** 2) * self.param_beta
                    loss_r_side = torch.mean((side_view - side_rendered) ** 2) * self.param_beta

                    loss_r = loss_r_back + loss_r_front + loss_r_top + loss_r_side
                    loss_r.backward()

                    self.optimizer_g.step()

                # G step
                g_step = 1
                for step in range(g_step):
                    self.generator.zero_grad()

                    z_geometry_code = torch.matmul(z_geometry_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                    z_texture_code = torch.matmul(z_texture_tensor, self.generator.texture_codes).view([1, -1, 1, 1, 1])
                    voxel_fake, texture_fake = self.generator(input_fake, z_geometry_code, z_texture_code, Gmask_fake, is_geometry_training=False)

                    back_fake, front_fake, top_fake, side_fake, _ = self.rendering((voxel_fake > self.sampling_threshold).float(), texture_fake)

                    D_out_back = self.texture_discriminator_back(back_fake, is_training=False)
                    D_out_front = self.texture_discriminator_front(front_fake, is_training=False)
                    D_out_top = self.texture_discriminator_top(top_fake, is_training=False)
                    D_out_side = self.texture_discriminator_side(side_fake, is_training=False)

                    loss_g_back = (torch.sum((D_out_back[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_fake_back) * self.param_alpha +
                                   torch.sum((D_out_back[:, -1:] - 1) ** 2 * Dmask_fake_back)) / torch.sum(Dmask_fake_back)
                    loss_g_front = (torch.sum((D_out_front[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_fake_front) * self.param_alpha +
                                    torch.sum((D_out_front[:, -1:] - 1) ** 2 * Dmask_fake_front)) / torch.sum(Dmask_fake_front)
                    loss_g_top = (torch.sum((D_out_top[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_fake_top) * self.param_alpha +
                                  torch.sum((D_out_top[:, -1:] - 1) ** 2 * Dmask_fake_top)) / torch.sum(Dmask_fake_top)
                    loss_g_side = (torch.sum((D_out_side[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_fake_side) * self.param_alpha +
                                   torch.sum((D_out_side[:, -1:] - 1) ** 2 * Dmask_fake_side)) / torch.sum(Dmask_fake_side)

                    loss_g = loss_g_back + loss_g_front + loss_g_top + loss_g_side
                    loss_g.backward()

                    self.optimizer_g.step()

                # visualize during training
                if epoch % 1 == 0 and (idx + 1) % (self.dataset_len // 8) == 0:
                    # save rendered texture images
                    back_rendered_view = back_fake.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    front_rendered_view = front_fake.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    top_rendered_view = top_fake.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    side_rendered_view = side_fake.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    back_rendered_view = np.round(np.clip(back_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                    front_rendered_view = np.round(np.clip(front_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                    top_rendered_view = np.round(np.clip(top_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                    side_rendered_view = np.round(np.clip(side_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                    Image.fromarray(back_rendered_view).save(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_back.png")
                    Image.fromarray(front_rendered_view).save(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_front.png")
                    Image.fromarray(top_rendered_view).save(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_top.png")
                    Image.fromarray(side_rendered_view).save(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_side.png")

                    # save geometry with texture
                    geometry_voxel_fake = voxel_fake[0, 0].detach().cpu().numpy()
                    texture_voxel_fake = texture_fake[0, :].permute(1, 2, 3, 0).detach().cpu().numpy()
                    xmin, xmax, ymin, ymax, zmin, zmax = self.pos_content[dxb]
                    geometry_voxel = self.recover_voxel(geometry_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                    texture_voxel = self.recover_color_voxel(texture_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                    texture_voxel = np.round(np.clip(texture_voxel * 255, a_min=0, a_max=255)).astype(np.uint8)

                    vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                    vertices_1 = vertices.astype(np.int32)
                    vertices_2 = (vertices + 0.5).astype(np.int32)
                    vertices_3 = (vertices - 0.5).astype(np.int32)
                    vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                    vertices_colors = np.maximum(np.maximum(texture_voxel[vertices_1[:, 0], vertices_1[:, 1], vertices_1[:, 2]],
                                                            texture_voxel[vertices_2[:, 0], vertices_2[:, 1], vertices_2[:, 2]]),
                                                 texture_voxel[vertices_3[:, 0], vertices_1[:, 1], vertices_1[:, 2]])
                    vertices_colors = vertices_colors.astype(np.uint8)
                    write_ply_triangle_color(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_color.ply", vertices, vertices_colors, triangles)

            print("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f " % (
                epoch, training_epoch, time.time() - start_time, loss_d_real.item(), loss_d_fake.item(), loss_r.item(), loss_g.item()))

            if epoch % self.save_epoch == 0:
                self.save(epoch)

        # if finish, save
        self.save(epoch)

    def train_texture_asymmetry(self, config):

        print("Start training texture with 5 views and 5 discriminators...")

        if not self.load_pretrained_geometry():
            exit(-1)

        start_time = time.time()
        training_epoch = config.epoch

        batch_index_list = np.arange(self.dataset_len)
        iter_counter = 0

        for epoch in range(0, training_epoch):
            np.random.shuffle(batch_index_list)

            self.texture_discriminator_top.train()
            self.texture_discriminator_back.train()
            self.texture_discriminator_front.train()
            self.texture_discriminator_side.train()
            self.texture_discriminator_right.train()
            self.generator.train()

            for idx in range(self.dataset_len):
                # random a z vector for D training
                z_vector_geometry = np.zeros([self.styleset_len], np.float32)
                z_vector_geometry_idx = np.random.randint(self.styleset_len)
                z_vector_geometry[z_vector_geometry_idx] = 1
                z_geometry_tensor = torch.from_numpy(z_vector_geometry).to(self.device).view([1, -1])

                z_vector_texture = np.zeros([self.styleset_len], np.float32)
                z_vector_texture_idx = z_vector_geometry_idx
                z_vector_texture[z_vector_texture_idx] = 1
                z_texture_tensor = torch.from_numpy(z_vector_texture).to(self.device).view([1, -1])

                # ready a fake voxel
                dxb = batch_index_list[idx]
                Gmask_fake = torch.from_numpy(self.Gmask_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                input_fake = torch.from_numpy(self.input_content[dxb]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                z_geometry_code = torch.matmul(z_geometry_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                z_texture_code = torch.matmul(z_texture_tensor, self.generator.texture_codes).view([1, -1, 1, 1, 1])
                voxel_fake, texture_fake = self.generator(input_fake, z_geometry_code, z_texture_code, Gmask_fake, is_geometry_training=False)
                voxel_fake = voxel_fake.detach()
                texture_fake = texture_fake.detach()
                back_fake, front_fake, top_fake, left_fake, right_fake = self.rendering((voxel_fake > self.sampling_threshold).float(), texture_fake)
                Dmask_fake_back = self.get_image_Dmask_from_rendered_view(back_fake[:, 3:4, :, :])
                Dmask_fake_front = self.get_image_Dmask_from_rendered_view(front_fake[:, 3:4, :, :])
                Dmask_fake_top = self.get_image_Dmask_from_rendered_view(top_fake[:, 3:4, :, :])
                Dmask_fake_left = self.get_image_Dmask_from_rendered_view(left_fake[:, 3:4, :, :])
                Dmask_fake_right = self.get_image_Dmask_from_rendered_view(right_fake[:, 3:4, :, :])

                # D step
                d_steps = 1
                for d_step in range(d_steps):
                    qxp = z_vector_texture_idx

                    self.texture_discriminator_top.zero_grad()
                    self.texture_discriminator_back.zero_grad()
                    self.texture_discriminator_front.zero_grad()
                    self.texture_discriminator_side.zero_grad()
                    self.texture_discriminator_right.zero_grad()

                    back_view = torch.from_numpy(self.render_style[qxp][0]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    front_view = torch.from_numpy(self.render_style[qxp][1]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    top_view = torch.from_numpy(self.render_style[qxp][2]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    left_view = torch.from_numpy(self.render_style[qxp][3]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    right_view = torch.from_numpy(self.render_style[qxp][4]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    Dmask_style_back = self.get_image_Dmask_from_rendered_view(back_view[:, 3:4, :, :])
                    Dmask_style_front = self.get_image_Dmask_from_rendered_view(front_view[:, 3:4, :, :])
                    Dmask_style_top = self.get_image_Dmask_from_rendered_view(top_view[:, 3:4, :, :])
                    Dmask_style_left = self.get_image_Dmask_from_rendered_view(left_view[:, 3:4, :, :])
                    Dmask_style_right = self.get_image_Dmask_from_rendered_view(right_view[:, 3:4, :, :])

                    # train real images
                    D_out_top = self.texture_discriminator_top(top_view, is_training=True)
                    D_out_back = self.texture_discriminator_back(back_view, is_training=True)
                    D_out_front = self.texture_discriminator_front(front_view, is_training=True)
                    D_out_left = self.texture_discriminator_side(left_view, is_training=True)
                    D_out_right = self.texture_discriminator_right(right_view, is_training=True)

                    loss_d_real_top = (torch.sum((D_out_top[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_style_top) +
                                       torch.sum((D_out_top[:, -1:] - 1) ** 2 * Dmask_style_top)) / torch.sum(Dmask_style_top)
                    loss_d_real_back = (torch.sum((D_out_back[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_style_back) +
                                        torch.sum((D_out_back[:, -1:] - 1) ** 2 * Dmask_style_back)) / torch.sum(Dmask_style_back)
                    loss_d_real_front = (torch.sum((D_out_front[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_style_front) +
                                         torch.sum((D_out_front[:, -1:] - 1) ** 2 * Dmask_style_front)) / torch.sum(Dmask_style_front)
                    loss_d_real_left = (torch.sum((D_out_left[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_style_left) +
                                        torch.sum((D_out_left[:, -1:] - 1) ** 2 * Dmask_style_left)) / torch.sum(Dmask_style_left)
                    loss_d_real_right = (torch.sum((D_out_right[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_style_right) +
                                         torch.sum((D_out_right[:, -1:] - 1) ** 2 * Dmask_style_right)) / torch.sum(Dmask_style_right)

                    loss_d_real = loss_d_real_top + loss_d_real_back + loss_d_real_front + loss_d_real_left + loss_d_real_right
                    loss_d_real.backward()

                    # train fake images
                    D_out_top = self.texture_discriminator_top(top_fake, is_training=True)
                    D_out_back = self.texture_discriminator_back(back_fake, is_training=True)
                    D_out_front = self.texture_discriminator_front(front_fake, is_training=True)
                    D_out_left = self.texture_discriminator_side(left_fake, is_training=True)
                    D_out_right = self.texture_discriminator_right(right_fake, is_training=True)

                    loss_d_fake_top = (torch.sum((D_out_top[:, z_vector_texture_idx:z_vector_texture_idx + 1]) ** 2 * Dmask_fake_top) +
                                       torch.sum((D_out_top[:, -1:]) ** 2 * Dmask_fake_top)) / torch.sum(Dmask_fake_top)
                    loss_d_fake_back = (torch.sum((D_out_back[:, z_vector_texture_idx:z_vector_texture_idx + 1]) ** 2 * Dmask_fake_back) +
                                        torch.sum((D_out_back[:, -1:]) ** 2 * Dmask_fake_back)) / torch.sum(Dmask_fake_back)
                    loss_d_fake_front = (torch.sum((D_out_front[:, z_vector_texture_idx:z_vector_texture_idx + 1]) ** 2 * Dmask_fake_front) +
                                         torch.sum((D_out_front[:, -1:]) ** 2 * Dmask_fake_front)) / torch.sum(Dmask_fake_front)
                    loss_d_fake_left = (torch.sum((D_out_left[:, z_vector_texture_idx:z_vector_texture_idx + 1]) ** 2 * Dmask_fake_left) +
                                        torch.sum((D_out_left[:, -1:]) ** 2 * Dmask_fake_left)) / torch.sum(Dmask_fake_left)
                    loss_d_fake_right = (torch.sum((D_out_right[:, z_vector_texture_idx:z_vector_texture_idx + 1]) ** 2 * Dmask_fake_right) +
                                         torch.sum((D_out_right[:, -1:]) ** 2 * Dmask_fake_right)) / torch.sum(Dmask_fake_right)

                    loss_d_fake = loss_d_fake_top + loss_d_fake_back + loss_d_fake_front + loss_d_fake_left + loss_d_fake_right
                    loss_d_fake.backward()

                    self.optimizer_d_texture.step()

                # recon step
                # reconstruct style image
                r_steps = 4 if iter_counter < 5000 else 1
                iter_counter += 1
                for r_step in range(r_steps):
                    qxp = np.random.randint(self.styleset_len)

                    z_vector_geometry_recon = np.zeros([self.styleset_len], np.float32)
                    z_vector_geometry_recon[qxp] = 1
                    z_geometry_recon_tensor = torch.from_numpy(z_vector_geometry_recon).to(self.device).view([1, -1])

                    z_vector_texture_recon = np.zeros([self.styleset_len], np.float32)
                    z_vector_texture_recon[qxp] = 1
                    z_texture_recon_tensor = torch.from_numpy(z_vector_texture_recon).to(self.device).view([1, -1])

                    Gmask_style = torch.from_numpy(self.Gmask_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    input_style = torch.from_numpy(self.input_style[qxp]).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    back_view = torch.from_numpy(self.render_style[qxp][0]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    front_view = torch.from_numpy(self.render_style[qxp][1]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    top_view = torch.from_numpy(self.render_style[qxp][2]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    left_view = torch.from_numpy(self.render_style[qxp][3]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
                    right_view = torch.from_numpy(self.render_style[qxp][4]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()

                    self.generator.zero_grad()

                    z_geometry_recon_code = torch.matmul(z_geometry_recon_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                    z_texture_recon_code = torch.matmul(z_texture_recon_tensor, self.generator.texture_codes).view([1, -1, 1, 1, 1])
                    voxel_fake, texture_fake = self.generator(input_style, z_geometry_recon_code, z_texture_recon_code, Gmask_style, is_geometry_training=False)

                    back_rendered, front_rendered, top_rendered, left_rendered, right_rendered = self.rendering((voxel_fake > self.sampling_threshold).float(),
                                                                                                                texture_fake)
                    loss_r_top = torch.mean((top_view - top_rendered) ** 2) * self.param_beta
                    loss_r_back = torch.mean((back_view - back_rendered) ** 2) * self.param_beta
                    loss_r_front = torch.mean((front_view - front_rendered) ** 2) * self.param_beta
                    loss_r_left = torch.mean((left_view - left_rendered) ** 2) * self.param_beta
                    loss_r_right = torch.mean((right_view - right_rendered) ** 2) * self.param_beta

                    loss_r = loss_r_top + loss_r_back + loss_r_front + loss_r_left + loss_r_right
                    loss_r.backward()

                    self.optimizer_g.step()

                # G step
                g_step = 1
                for step in range(g_step):
                    self.generator.zero_grad()

                    z_geometry_code = torch.matmul(z_geometry_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                    z_texture_code = torch.matmul(z_texture_tensor, self.generator.texture_codes).view([1, -1, 1, 1, 1])
                    voxel_fake, texture_fake = self.generator(input_fake, z_geometry_code, z_texture_code, Gmask_fake, is_geometry_training=False)

                    back_fake, front_fake, top_fake, left_fake, right_fake = self.rendering((voxel_fake > self.sampling_threshold).float(), texture_fake)

                    D_out_top = self.texture_discriminator_top(top_fake, is_training=False)
                    D_out_back = self.texture_discriminator_back(back_fake, is_training=False)
                    D_out_front = self.texture_discriminator_front(front_fake, is_training=False)
                    D_out_left = self.texture_discriminator_side(left_fake, is_training=False)
                    D_out_right = self.texture_discriminator_right(right_fake, is_training=False)

                    loss_g_top = (torch.sum((D_out_top[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_fake_top) * self.param_alpha +
                                  torch.sum((D_out_top[:, -1:] - 1) ** 2 * Dmask_fake_top)) / torch.sum(Dmask_fake_top)
                    loss_g_back = (torch.sum((D_out_back[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_fake_back) * self.param_alpha +
                                   torch.sum((D_out_back[:, -1:] - 1) ** 2 * Dmask_fake_back)) / torch.sum(Dmask_fake_back)
                    loss_g_front = (torch.sum((D_out_front[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_fake_front) * self.param_alpha +
                                    torch.sum((D_out_front[:, -1:] - 1) ** 2 * Dmask_fake_front)) / torch.sum(Dmask_fake_front)
                    loss_g_left = (torch.sum((D_out_left[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_fake_left) * self.param_alpha +
                                   torch.sum((D_out_left[:, -1:] - 1) ** 2 * Dmask_fake_left)) / torch.sum(Dmask_fake_left)
                    loss_g_right = (torch.sum((D_out_right[:, z_vector_texture_idx:z_vector_texture_idx + 1] - 1) ** 2 * Dmask_fake_right) * self.param_alpha +
                                    torch.sum((D_out_right[:, -1:] - 1) ** 2 * Dmask_fake_right)) / torch.sum(Dmask_fake_right)

                    loss_g = loss_g_top + loss_g_back + loss_g_front + loss_g_left + loss_g_right
                    loss_g.backward()

                    self.optimizer_g.step()

                # visualize during training
                if epoch % 1 == 0 and (idx + 1) % (self.dataset_len // 8) == 0:
                    # save rendered texture images
                    back_rendered_view = back_fake[:, :4, :, :].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    front_rendered_view = front_fake[:, :4, :, :].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    top_rendered_view = top_fake[:, :4, :, :].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    left_rendered_view = left_fake[:, :4, :, :].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    right_rendered_view = right_fake[:, :4, :, :].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    back_rendered_view = np.round(np.clip(back_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                    front_rendered_view = np.round(np.clip(front_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                    top_rendered_view = np.round(np.clip(top_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                    left_rendered_view = np.round(np.clip(left_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                    right_rendered_view = np.round(np.clip(right_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                    Image.fromarray(back_rendered_view).save(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_back.png")
                    Image.fromarray(front_rendered_view).save(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_front.png")
                    Image.fromarray(top_rendered_view).save(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_top.png")
                    Image.fromarray(left_rendered_view).save(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_left.png")
                    Image.fromarray(right_rendered_view).save(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_right.png")

                    # save geometry with texture
                    geometry_voxel_fake = voxel_fake[0, 0].detach().cpu().numpy()
                    texture_voxel_fake = texture_fake[0, :].permute(1, 2, 3, 0).detach().cpu().numpy()
                    xmin, xmax, ymin, ymax, zmin, zmax = self.pos_content[dxb]
                    geometry_voxel = self.recover_voxel(geometry_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                    texture_voxel = self.recover_color_voxel(texture_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                    texture_voxel = np.round(np.clip(texture_voxel * 255, a_min=0, a_max=255)).astype(np.uint8)

                    vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                    vertices_1 = vertices.astype(np.int32)
                    vertices_2 = (vertices + 0.5).astype(np.int32)
                    vertices_3 = (vertices - 0.5).astype(np.int32)
                    vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                    vertices_colors = np.maximum(np.maximum(texture_voxel[vertices_1[:, 0], vertices_1[:, 1], vertices_1[:, 2]],
                                                            texture_voxel[vertices_2[:, 0], vertices_2[:, 1], vertices_2[:, 2]]),
                                                 texture_voxel[vertices_3[:, 0], vertices_1[:, 1], vertices_1[:, 2]])
                    vertices_colors = vertices_colors.astype(np.uint8)
                    write_ply_triangle_color(config.sample_dir + "/" + str(epoch) + "_" + str(idx + 1) + "_color.ply", vertices, vertices_colors, triangles)

            print("Epoch: [%d/%d] time: %.0f, loss_d_real: %.6f, loss_d_fake: %.6f, loss_r: %.6f, loss_g: %.6f " % (
                epoch, training_epoch, time.time() - start_time, loss_d_real.item(), loss_d_fake.item(), loss_r.item(), loss_g.item()))

            if epoch % self.save_epoch == 0:
                self.save(epoch)

        # if finish, save
        self.save(epoch)

    def test_texture(self, config):

        print("Start testing texture...")
        save_image = True
        save_mesh = True
        max_num_of_contents = 10

        fin = open("splits/" + self.data_content + ".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        if not self.load():
            exit(-1)

        print("testing {} contents with {} styles...".format(self.dataset_len, self.styleset_len))
        for i in range(self.dataset_len):

            print(i, self.dataset_names[i])

            voxel_path = os.path.join(self.data_dir, self.dataset_names[i] + "/model_depth_fusion.binvox")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(voxel_path).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
            elif self.output_size == 512:
                tmp_raw = get_vox_from_binvox_512(voxel_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            tmp_input, _, _, tmp_Gmask = self.get_voxel_input_Dmask_Gmask(tmp)
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()
            Gmask_fake = torch.from_numpy(tmp_Gmask).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for j in range(self.styleset_len):

                save_dir = os.path.join(config.sample_dir, self.dataset_names[i], self.styleset_names[j])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                with torch.no_grad():
                    z_vector_geometry = np.zeros([self.styleset_len], np.float32)
                    z_vector_geometry[j] = 1
                    z_geometry_tensor = torch.from_numpy(z_vector_geometry).to(self.device).view([1, -1])

                    z_vector_texture = np.zeros([self.styleset_len], np.float32)
                    z_vector_texture[j] = 1
                    z_texture_tensor = torch.from_numpy(z_vector_texture).to(self.device).view([1, -1])

                    z_geometry_code = torch.matmul(z_geometry_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                    z_texture_code = torch.matmul(z_texture_tensor, self.generator.texture_codes).view([1, -1, 1, 1, 1])
                    voxel_fake, texture_fake = self.generator(input_fake, z_geometry_code, z_texture_code, Gmask_fake, is_geometry_training=False)
                    back_fake, front_fake, top_fake, left_fake, right_fake = self.rendering((voxel_fake > self.sampling_threshold).float(), texture_fake)

                    if save_image:
                        back_rendered_view = back_fake.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        front_rendered_view = front_fake.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        top_rendered_view = top_fake.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        left_rendered_view = left_fake.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        right_rendered_view = right_fake.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                        back_rendered_view = self.recover_texture_image(back_rendered_view, ymin, ymax, zmin, zmax, asymmetry=self.asymmetry)
                        front_rendered_view = self.recover_texture_image(front_rendered_view, ymin, ymax, zmin, zmax, asymmetry=self.asymmetry)
                        top_rendered_view = self.recover_texture_image(top_rendered_view, xmin, xmax, zmin, zmax, asymmetry=self.asymmetry)
                        left_rendered_view = self.recover_texture_image(left_rendered_view, xmin, xmax, ymin, ymax, asymmetry=True)
                        right_rendered_view = self.recover_texture_image(right_rendered_view, xmin, xmax, ymin, ymax, asymmetry=True)
                        back_rendered_view = np.round(np.clip(back_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                        front_rendered_view = np.round(np.clip(front_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                        top_rendered_view = np.round(np.clip(top_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                        left_rendered_view = np.round(np.clip(left_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                        right_rendered_view = np.round(np.clip(right_rendered_view * 255, a_min=0, a_max=255)).astype(np.uint8)
                        Image.fromarray(back_rendered_view).save(save_dir + "/back.png")
                        Image.fromarray(front_rendered_view).save(save_dir + "/front.png")
                        Image.fromarray(top_rendered_view).save(save_dir + "/top.png")
                        Image.fromarray(left_rendered_view).save(save_dir + "/left.png")
                        if self.asymmetry:
                            Image.fromarray(right_rendered_view).save(save_dir + "/right.png")

                    if save_mesh:
                        # save geometry with texture
                        geometry_voxel_fake = voxel_fake[0, 0].detach().cpu().numpy()
                        geometry_voxel_fake = gaussian_filter(geometry_voxel_fake, sigma=1)
                        texture_voxel_fake = texture_fake[0, :].permute(1, 2, 3, 0).detach().cpu().numpy()
                        geometry_voxel = self.recover_voxel(geometry_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                        texture_voxel = self.recover_color_voxel(texture_voxel_fake, xmin, xmax, ymin, ymax, zmin, zmax)
                        texture_voxel = np.round(np.clip(texture_voxel * 255, a_min=0, a_max=255)).astype(np.uint8)

                        vertices, triangles = mcubes.marching_cubes(geometry_voxel, self.sampling_threshold)
                        vertices_1 = vertices.astype(np.int32)
                        vertices_2 = (vertices + 0.5).astype(np.int32)
                        vertices_3 = (vertices - 0.5).astype(np.int32)
                        vertices = (vertices + 0.5) / geometry_voxel.shape[0] - 0.5
                        vertices_colors = np.maximum(np.maximum(texture_voxel[vertices_1[:, 0], vertices_1[:, 1], vertices_1[:, 2]],
                                                                texture_voxel[vertices_2[:, 0], vertices_2[:, 1], vertices_2[:, 2]]),
                                                                texture_voxel[vertices_3[:, 0], vertices_3[:, 1], vertices_3[:, 2]])
                        vertices_colors = vertices_colors.astype(np.uint8)
                        write_ply_triangle_color(save_dir + "/color.ply", vertices, vertices_colors, triangles)

    def prepare_content_voxel_for_visualization(self, config):

        max_num_of_contents = 10
        print("Prepare content coarse voxel for visualization...")
        for i in range(min(max_num_of_contents, self.dataset_len)):

            print(i, self.dataset_names[i])

            save_dir = os.path.join(config.sample_dir, self.dataset_names[i])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            voxel_path = os.path.join(self.data_dir, self.dataset_names[i] + "/model_depth_fusion.binvox")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(voxel_path).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
            elif self.output_size == 512:
                tmp_raw = get_vox_from_binvox_512(voxel_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")

            vox_tensor = torch.from_numpy(tmp_raw).to(self.device).unsqueeze(0).unsqueeze(0).float()
            smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
            upsampled_tensor = F.interpolate(smallmaskx_tensor, scale_factor=self.upsample_rate, mode='nearest')
            upsampled = upsampled_tensor.detach().cpu().numpy()[0, 0]
            upsampled = np.round(upsampled).astype(np.uint8)

            vertices, triangles = mcubes.marching_cubes(upsampled, self.sampling_threshold)
            vertices = (vertices + 0.5) / upsampled.shape[0] - 0.5
            write_ply_triangle(save_dir + "/" + "geometry.ply", vertices, triangles)

    def prepare_style_texture_images(self, config):

        print("Prepare style texture images...")

        for i in range(self.styleset_len):

            save_dir = os.path.join(config.sample_dir, self.styleset_names[i])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            color_path = os.path.join(self.data_dir, self.styleset_names[i] + "/voxel_color.hdf5")
            data_dict = h5py.File(color_path, 'r')
            voxel_texture = data_dict["voxel_color"][:]
            data_dict.close()

            # the color is BGR, change to RGB
            geometry = voxel_texture[:, :, :, -1]
            texture = voxel_texture[:, :, :, :3]
            texture = texture[:, :, :, [2, 1, 0]]

            if self.output_size == 256:
                geometry = F.max_pool3d(torch.from_numpy(geometry).unsqueeze(0).unsqueeze(0).float(), kernel_size=2, stride=2, padding=0).numpy()[0, 0]
                texture = F.interpolate(torch.from_numpy(texture).permute(3, 0, 1, 2).unsqueeze(0).float(),
                                        scale_factor=0.5, mode='trilinear').squeeze(0).permute(1, 2, 3, 0).numpy()
                assert (texture >= 0).all()

            geometry = torch.from_numpy(geometry).to(self.device).unsqueeze(0).unsqueeze(0).float()
            texture = torch.from_numpy(texture).to(self.device).permute(3, 0, 1, 2).contiguous().unsqueeze(0).float() / 255.0
            back_texture, front_texture, top_texture, left_texture, right_texture = self.rendering(geometry, texture)

            # each (512, 512, 3+1)
            back_texture = back_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
            front_texture = front_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
            top_texture = top_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
            left_texture = left_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
            right_texture = right_texture.squeeze(0).permute(1, 2, 0).contiguous().detach().cpu().numpy()
            Image.fromarray((back_texture * 255).astype(np.uint8)).save(save_dir + "/back.png")
            Image.fromarray((front_texture * 255).astype(np.uint8)).save(save_dir + "/front.png")
            Image.fromarray((top_texture * 255).astype(np.uint8)).save(save_dir + "/top.png")
            Image.fromarray((left_texture * 255).astype(np.uint8)).save(save_dir + "/left.png")
            Image.fromarray((right_texture * 255).astype(np.uint8)).save(save_dir + "/right.png")

    def prepare_voxel_style(self, config):
        import binvox_rw_faster as binvox_rw

        result_dir = "output_for_geo_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        max_num_of_styles = 16
        max_num_of_contents = 20

        # load style shapes
        fin = open("splits/" + self.data_style + ".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        for style_id in range(self.styleset_len):
            print("preprocessing style - " + str(style_id + 1) + "/" + str(self.styleset_len))
            voxel_path = os.path.join(self.data_dir, self.styleset_names[style_id] + "/model_depth_fusion.binvox")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(voxel_path).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
            elif self.output_size == 512:
                tmp_raw = get_vox_from_binvox_512(voxel_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)
            binvox_rw.write_voxel(tmp, result_dir + "/style_" + str(style_id) + ".binvox")

    def prepare_voxel_for_eval(self, config):
        import binvox_rw_faster as binvox_rw
        #import mcubes

        result_dir = "output_for_geo_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load():
            exit(-1)

        max_num_of_styles = 16
        max_num_of_contents = 20

        # load style shapes
        fin = open("splits/" + self.data_style + ".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        # load content shapes
        fin = open("splits/" + self.data_content + ".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))
            voxel_path = os.path.join(self.data_dir, self.dataset_names[content_id] + "/model_depth_fusion.binvox")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(voxel_path).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(voxel_path).astype(np.uint8)
            elif self.output_size == 512:
                tmp_raw = get_vox_from_binvox_512(voxel_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")
            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            tmp_input, _, _, tmp_Gmask = self.get_voxel_input_Dmask_Gmask(tmp)
            binvox_rw.write_voxel(tmp_input, result_dir + "/content_" + str(content_id) + "_coarse.binvox")

            Gmask_fake = torch.from_numpy(tmp_Gmask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len], np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1, -1])

                with torch.no_grad():
                    z_tensor_g = torch.matmul(z_tensor, self.generator.geometry_codes).view([1, -1, 1, 1, 1])
                    voxel_fake_lg, _ = self.generator(input_fake, z_tensor_g, None, Gmask_fake, is_geometry_training=True)

                tmp_voxel_fake = voxel_fake_lg.detach().cpu().numpy()[0, 0]
                tmp_voxel_fake = (tmp_voxel_fake > self.sampling_threshold).astype(np.uint8)

                binvox_rw.write_voxel(tmp_voxel_fake, result_dir + "/output_content_" + str(content_id) + "_style_" + str(style_id) + ".binvox")

    def render_real_for_eval(self, config):

        self.voxel_renderer.use_gpu()

        result_dir = "render_real_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        sample_num_views = 24
        render_boundary_padding_size = 16
        half_real_size = self.real_size // 2

        # load all shapes
        fin = open("splits/" + self.data_content + ".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))
            vox_path = os.path.join(self.data_dir, self.dataset_names[content_id] + "/model_depth_fusion.binvox")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(vox_path).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(vox_path).astype(np.uint8)
            elif self.output_size == 512:
                tmp_raw = get_vox_from_binvox_512(vox_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")

            # tmp_raw = gaussian_filter(tmp_raw.astype(np.float32), sigma=1)

            for sample_id in range(sample_num_views):
                cam_alpha = np.random.random() * np.pi * 2
                cam_beta = np.random.random() * np.pi / 2 - np.pi / 4
                imgout = self.voxel_renderer.render_img_with_camera_pose_gpu(tmp_raw, self.sampling_threshold, cam_alpha, cam_beta, get_depth=False,
                                                                             processed=False)
                if self.output_size == 128:
                    imgout = cv2.resize(imgout, (self.real_size * 2, self.real_size * 2), interpolation=cv2.INTER_NEAREST)
                    imgout = imgout[half_real_size:-half_real_size, half_real_size:-half_real_size]
                cv2.imwrite(result_dir + "/" + str(content_id) + "_" + str(sample_id) + ".png", imgout)

    def render_fake_for_eval(self, config):

        self.voxel_renderer.use_gpu()

        result_dir = "render_fake_for_eval"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if not self.load():
            exit(-1)

        sample_num_views = 24
        render_boundary_padding_size = 16
        half_real_size = self.real_size // 2
        max_num_of_styles = 16
        max_num_of_contents = 100

        # load style shapes
        fin = open("splits/" + self.data_style + ".txt")
        self.styleset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.styleset_len = len(self.styleset_names)

        # load content shapes
        fin = open("splits/" + self.data_content + ".txt")
        self.dataset_names = [name.strip() for name in fin.readlines()]
        fin.close()
        self.dataset_len = len(self.dataset_names)
        self.dataset_len = min(self.dataset_len, max_num_of_contents)

        for content_id in range(self.dataset_len):
            print("processing content - " + str(content_id + 1) + "/" + str(self.dataset_len))
            vox_path = os.path.join(self.data_dir, self.dataset_names[content_id] + "/model_depth_fusion.binvox")
            if self.output_size == 128:
                tmp_raw = get_vox_from_binvox_1over2(vox_path).astype(np.uint8)
            elif self.output_size == 256:
                tmp_raw = get_vox_from_binvox(vox_path).astype(np.uint8)
            elif self.output_size == 512:
                tmp_raw = get_vox_from_binvox_512(vox_path).astype(np.uint8)
            else:
                raise NotImplementedError("Output size " + str(self.output_size) + " not supported...")

            xmin, xmax, ymin, ymax, zmin, zmax = self.get_voxel_bbox(tmp_raw)
            tmp = self.crop_voxel(tmp_raw, xmin, xmax, ymin, ymax, zmin, zmax)

            tmp_input, _, _, tmp_Gmask = self.get_voxel_input_Dmask_Gmask(tmp)
            Gmask_fake = torch.from_numpy(tmp_Gmask).to(self.device).unsqueeze(0).unsqueeze(0).float()
            input_fake = torch.from_numpy(tmp_input).to(self.device).unsqueeze(0).unsqueeze(0).float()

            tmpvoxlarger = np.zeros([self.real_size + render_boundary_padding_size * 2, self.real_size + render_boundary_padding_size * 2,
                                     self.real_size + render_boundary_padding_size * 2], np.float32)

            for style_id in range(min(self.styleset_len, max_num_of_styles)):
                z_vector = np.zeros([self.styleset_len], np.float32)
                z_vector[style_id] = 1
                z_tensor = torch.from_numpy(z_vector).to(self.device).view([1, -1])

                with torch.no_grad():
                    z_tensor_g = torch.matmul(z_tensor, self.generator.style_codes).view([1, -1, 1, 1, 1])
                    voxel_fake_lg, _ = self.generator(input_fake, z_tensor_g, None, Gmask_fake, is_geometry_training=True)

                tmp_voxel_fake = voxel_fake_lg.detach().cpu().numpy()[0, 0]

                xmin2 = xmin * self.upsample_rate - self.mask_margin
                xmax2 = xmax * self.upsample_rate + self.mask_margin
                ymin2 = ymin * self.upsample_rate - self.mask_margin
                ymax2 = ymax * self.upsample_rate + self.mask_margin
                if self.asymmetry:
                    zmin2 = zmin * self.upsample_rate - self.mask_margin
                else:
                    zmin2 = zmin * self.upsample_rate
                zmax2 = zmax * self.upsample_rate + self.mask_margin

                if self.asymmetry:
                    tmpvoxlarger[xmin2 + render_boundary_padding_size:xmax2 + render_boundary_padding_size,
                                 ymin2 + render_boundary_padding_size:ymax2 + render_boundary_padding_size,
                                 zmin2 + render_boundary_padding_size:zmax2 + render_boundary_padding_size] = tmp_voxel_fake[::-1, ::-1, :]
                else:
                    tmpvoxlarger[xmin2 + render_boundary_padding_size:xmax2 + render_boundary_padding_size,
                                 ymin2 + render_boundary_padding_size:ymax2 + render_boundary_padding_size,
                                 zmin2 + render_boundary_padding_size:zmax2 + render_boundary_padding_size] = tmp_voxel_fake[::-1, ::-1, self.mask_margin:]
                    tmpvoxlarger[xmin2 + render_boundary_padding_size:xmax2 + render_boundary_padding_size,
                                 ymin2 + render_boundary_padding_size:ymax2 + render_boundary_padding_size,
                                 zmin2 - 1 + render_boundary_padding_size:zmin2 * 2 - zmax2 - 1 + render_boundary_padding_size:-1] = tmp_voxel_fake[::-1, ::-1,
                                                                                                                        self.mask_margin:]

                for sample_id in range(sample_num_views):
                    cam_alpha = np.random.random() * np.pi * 2
                    cam_beta = np.random.random() * np.pi / 2 - np.pi / 4
                    tmpvoxlarger_tensor = torch.from_numpy(tmpvoxlarger).to(self.device).unsqueeze(0).unsqueeze(0).float()
                    imgout = self.voxel_renderer.render_img_with_camera_pose_gpu(tmpvoxlarger_tensor, self.sampling_threshold, cam_alpha, cam_beta,
                                                                                 get_depth=False, processed=True)
                    if self.output_size == 128:
                        imgout = cv2.resize(imgout, (self.real_size * 2, self.real_size * 2), interpolation=cv2.INTER_NEAREST)
                        imgout = imgout[half_real_size:-half_real_size, half_real_size:-half_real_size]
                    cv2.imwrite(result_dir + "/" + str(content_id) + "_" + str(style_id) + "_" + str(sample_id) + ".png", imgout)
