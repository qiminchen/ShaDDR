# based on https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/vis_gui.py
import glob
import h5py
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
import sys
import mcubes
import random
import cutils
import argparse
from skimage import measure
from sklearn.neighbors import KDTree

from PIL import Image

import torch
import torch.nn.functional as F

from utils import get_vox_from_binvox, get_vox_from_binvox_1over2_return_small, write_ply_triangle
from modelAEH_GD import *


parser = argparse.ArgumentParser()
parser.add_argument("--category", action="store", dest="category", default="00000000", help="data category")
args = parser.parse_args()

isMacOS = (platform.system() == "Darwin")

if torch.cuda.is_available():
    GPU_ID = 0
    device = torch.device('cuda')
    print("Using GPU 0")
else:
    GPU_ID = -1
    device = torch.device('cpu')
    print("Using CPU")


CATEGORY_SETTING = {
    "00000000": {
        "z_dim": 8,
        "g_dim": 32,
        "styleset_len": 8,
        "coarse_res": 32,
        "detailed_res": 256,
        "cpk_path": "./checkpoint/style_color_building_8_aemr_tex_demo/IM_AE.model-13.pth",
    },
}


class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(255, 255, 255)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 30000
        self.sun_intensity = 30000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.LIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


class AppWindow:

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Lit", "Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, width, height, args):

        self.brush_size = 0
        self.coarse_voxel = None
        self.coarse_voxel_for_reset = None
        self.coarse_voxel_seg = None
        self.coarse_voxel_part = None
        self.network = None
        self.mask_margin = 16

        self.styleset_len = CATEGORY_SETTING[args.category]['styleset_len']
        self.coarse_res = CATEGORY_SETTING[args.category]['coarse_res']
        self.detailed_res = CATEGORY_SETTING[args.category]['detailed_res']
        self.z_dim = CATEGORY_SETTING[args.category]['z_dim']
        self.upsample_rate = self.detailed_res // self.coarse_res
        self.upsample_level = int(np.log2(self.upsample_rate))

        self.selected_positions = []

        self.color_voxel_grid = np.zeros((self.coarse_res, self.coarse_res, self.coarse_res, 3))
        for i in range(self.coarse_res):
            for j in range(self.coarse_res):
                for k in range(self.coarse_res):
                    self.color_voxel_grid[i, j, k] = [i * 8 / 255, j * 8 / 255, k * 8 / 255]

        self.color_maps = {1: [226, 211, 107], 2: [253, 147, 70], 3: [86, 197, 125], 4: [50, 157, 156], 5: [32, 80, 114], 0: [221, 221, 218]}

        # for GUI visualization
        self.mesh = None
        self.cache_mesh = False
        self.points_occupied = None
        self.points_colors = None
        self.points_unoccupied = None
        self.style_colors = {i + 1: [random.random(), random.random(), random.random()] for i in range(self.styleset_len)}

        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window("Detailization", width, height)
        w = self.window  # to make the code more concise

        self.offscreen_renderer = rendering.OffscreenRenderer(width, height)

        self.info = gui.Label("")
        self.info.visible = False

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)
        self._scene.set_on_mouse(self._on_mouse_widget3d)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        self.em = em
        self.separation_height = separation_height

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # ---------------------- I/O control ----------------------
        io_ctrls = self._io_controls_interface(em, separation_height)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(io_ctrls)

        # ------------------- Brush size control -------------------
        # not intuitive, disable
        brush_size_ctrls = self._brush_controls_interface(em, separation_height)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(brush_size_ctrls)

        # ---------------------- Style control ----------------------
        self.selected_style_index = 1
        self.applied_style_index = 1
        style_ctrls = self._style_controls_interface(em, separation_height)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(style_ctrls)

        # ----------------- detailization control ------------------
        detailization_ctrls = self._detailization_controls_interface(em, separation_height)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(detailization_ctrls)

        # ---------------------- Field control ----------------------
        field_ctrls = self._field_controls_interface(em, separation_height)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(field_ctrls)

        # ---------------------- Lighting control ----------------------
        light_ctrls = self._lighting_controls_interface(em, separation_height)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(light_ctrls)

        # ---------------------- Material control ----------------------
        material_settings = self._material_settings_interface(em, separation_height)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)
        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.add_child(self._scene)
        w.add_child(self._settings_panel)
        w.set_on_layout(self._on_layout)
        w.add_child(self.info)

        self._apply_settings()

    def _io_controls_interface(self, em, separation_height):
        io_ctrls = gui.CollapsableVert("I/O controls", 0.25 * em,
                                          gui.Margins(em, 0, 0, 0))
        io_ctrls.set_is_open(True)

        self._import_button = gui.Button("Import")
        self._import_button.horizontal_padding_em = 2.35
        self._import_button.vertical_padding_em = 0
        self._import_button.set_on_clicked(self._on_import_button)

        self._export_button = gui.Button("Export")
        self._export_button.horizontal_padding_em = 2.35
        self._export_button.vertical_padding_em = 0
        self._export_button.set_on_clicked(self._on_export_button)

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._import_button)
        h.add_child(self._export_button)
        h.add_stretch()
        io_ctrls.add_child(h)

        return io_ctrls

    def _mode_controls_interface(self, em, separation_height):
        mode_ctrls = gui.CollapsableVert("Mode controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        mode_ctrls.set_is_open(True)

        self._label_mode_button = gui.Button("Labeling")
        self._label_mode_button.horizontal_padding_em = 1.95
        self._label_mode_button.vertical_padding_em = 0
        # self._label_mode_button.set_on_clicked()

        self._edit_mode_button = gui.Button("Editing")
        self._edit_mode_button.horizontal_padding_em = 2.25
        self._edit_mode_button.vertical_padding_em = 0
        # self._edit_mode_button.set_on_clicked()

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._label_mode_button)
        h.add_child(self._edit_mode_button)
        h.add_stretch()
        mode_ctrls.add_child(h)

        return mode_ctrls

    def _brush_controls_interface(self, em, separation_height):
        brush_ctrls = gui.CollapsableVert("Brush controls", 0.25 * em,
                                          gui.Margins(em, 0, 0, 0))
        brush_ctrls.set_is_open(True)

        self.brush_size = 0
        self._1x1x1_button = gui.Button("1x1x1")
        self._1x1x1_button.horizontal_padding_em = 0.3
        self._1x1x1_button.vertical_padding_em = 0
        self._1x1x1_button.set_on_clicked(self._set_brush_size_1x1x1)
        self._1x1x1_button.background_color = gui.Color(153, 153, 153, 0.5)

        self._3x3x3_button = gui.Button("3x3x3")
        self._3x3x3_button.horizontal_padding_em = 0.3
        self._3x3x3_button.vertical_padding_em = 0
        self._3x3x3_button.set_on_clicked(self._set_brush_size_3x3x3)

        self._5x5x5_button = gui.Button("5x5x5")
        self._5x5x5_button.horizontal_padding_em = 0.3
        self._5x5x5_button.vertical_padding_em = 0
        self._5x5x5_button.set_on_clicked(self._set_brush_size_5x5x5)

        self._7x7x7_button = gui.Button("7x7x7")
        self._7x7x7_button.horizontal_padding_em = 0.3
        self._7x7x7_button.vertical_padding_em = 0
        self._7x7x7_button.set_on_clicked(self._set_brush_size_7x7x7)

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._1x1x1_button)
        h.add_child(self._3x3x3_button)
        h.add_child(self._5x5x5_button)
        h.add_child(self._7x7x7_button)
        h.add_stretch()
        brush_ctrls.add_child(h)

        return brush_ctrls

    def _style_controls_interface(self, em, separation_height):
        style_ctrls = gui.CollapsableVert("Style controls", 0.25 * em,
                                          gui.Margins(em, 0, 0, 0))
        style_ctrls.set_is_open(True)

        self._style_options = gui.Slider(gui.Slider.INT)
        self._style_options.set_limits(1, self.styleset_len)
        self._style_options.set_on_value_changed(self._on_style)

        self._select_button = gui.Button("Assign style")
        self._select_button.horizontal_padding_em = 0.3
        self._select_button.vertical_padding_em = 0
        self._select_button.set_on_clicked(self._set_style)

        # style_ctrls.add_child(gui.Label("Choose styles"))
        h = gui.Horiz(0.25 * em)
        h.add_child(self._style_options)
        h.add_child(self._select_button)
        style_ctrls.add_child(h)

        return style_ctrls

    def _detailization_controls_interface(self, em, separation_height):
        detailization_ctrls = gui.CollapsableVert("Detailization controls", 0.25 * em,
                                                  gui.Margins(em, 0, 0, 0))
        detailization_ctrls.set_is_open(True)

        self._coarse_button = gui.Button("Coarse")
        self._coarse_button.horizontal_padding_em = 0.8
        self._coarse_button.vertical_padding_em = 0
        self._coarse_button.set_on_clicked(self._on_coarse_button)

        self._detailize_button = gui.Button("Detailize")
        self._detailize_button.horizontal_padding_em = 0.8
        self._detailize_button.vertical_padding_em = 0
        self._detailize_button.set_on_clicked(self._on_detailization_button)

        self._reset_button = gui.Button("Reset")
        self._reset_button.horizontal_padding_em = 1.1
        self._reset_button.vertical_padding_em = 0
        self._reset_button.set_on_clicked(self._on_reset_button)

        h = gui.Horiz(0.25 * em)
        h.add_stretch()
        h.add_child(self._coarse_button)
        h.add_child(self._detailize_button)
        h.add_child(self._reset_button)
        h.add_stretch()
        detailization_ctrls.add_child(h)

        return detailization_ctrls

    def _field_controls_interface(self, em, separation_height):
        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        field_ctrls = gui.CollapsableVert("Field controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))
        field_ctrls.set_is_open(False)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        field_ctrls.add_fixed(separation_height)
        field_ctrls.add_child(self._show_skybox)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        field_ctrls.add_child(grid)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        field_ctrls.add_fixed(separation_height)
        field_ctrls.add_child(self._show_axes)

        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        field_ctrls.add_fixed(separation_height)
        field_ctrls.add_child(gui.Label("Lighting profiles"))
        field_ctrls.add_child(self._profiles)

        return field_ctrls

    def _lighting_controls_interface(self, em, separation_height):
        advanced = gui.CollapsableVert("Advanced lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path +
                             "/*_ibl.ktx"):
            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = AppWindow.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        return advanced

    def _material_settings_interface(self, em, separation_height):
        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        material_settings.set_is_open(False)

        self._shader = gui.Combobox()
        self._shader.add_item(AppWindow.MATERIAL_NAMES[0])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[1])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[2])
        self._shader.add_item(AppWindow.MATERIAL_NAMES[3])
        self._shader.set_on_selection_changed(self._on_shader)
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        return material_settings

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
                self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _set_brush_size_1x1x1(self):
        self.brush_size = 0
        self._1x1x1_button.background_color = gui.Color(153, 153, 153, 0.5)
        self._3x3x3_button.background_color = gui.Color(102, 102, 102, 0.29)
        self._5x5x5_button.background_color = gui.Color(102, 102, 102, 0.29)
        self._7x7x7_button.background_color = gui.Color(102, 102, 102, 0.29)

    def _set_brush_size_3x3x3(self):
        self.brush_size = 1
        self._1x1x1_button.background_color = gui.Color(102, 102, 102, 0.29)
        self._3x3x3_button.background_color = gui.Color(153, 153, 153, 0.5)
        self._5x5x5_button.background_color = gui.Color(102, 102, 102, 0.29)
        self._7x7x7_button.background_color = gui.Color(102, 102, 102, 0.29)

    def _set_brush_size_5x5x5(self):
        self.brush_size = 2
        self._1x1x1_button.background_color = gui.Color(102, 102, 102, 0.29)
        self._3x3x3_button.background_color = gui.Color(102, 102, 102, 0.29)
        self._5x5x5_button.background_color = gui.Color(153, 153, 153, 0.5)
        self._7x7x7_button.background_color = gui.Color(102, 102, 102, 0.29)

    def _set_brush_size_7x7x7(self):
        self.brush_size = 3
        self._1x1x1_button.background_color = gui.Color(102, 102, 102, 0.29)
        self._3x3x3_button.background_color = gui.Color(102, 102, 102, 0.29)
        self._5x5x5_button.background_color = gui.Color(102, 102, 102, 0.29)
        self._7x7x7_button.background_color = gui.Color(153, 153, 153, 0.5)

    def _on_style(self, index):
        self.selected_style_index = int(index)
        self._apply_settings()

    def _set_style(self):
        self.cache_mesh = False

    def _on_detailization_button(self):
        if self.network is None:
            return

        if not self.cache_mesh:

            print(" [*] Start pre-processing voxel")

            mask_margin_x = 3
            mask_margin_y = 2
            mask_margin_z = 3
            xmin, xmax = np.nonzero(np.sum(self.coarse_voxel, axis=(1, 2)))[0][[0, -1]]
            ymin, ymax = np.nonzero(np.sum(self.coarse_voxel, axis=(0, 2)))[0][[0, -1]]
            zmin, zmax = np.nonzero(np.sum(self.coarse_voxel, axis=(0, 1)))[0][[0, -1]]
            print("xmin: {}, xmax: {}, ymin: {}, ymax: {}, zmin: {}, zmax: {}".format(xmin, xmax, ymin, ymax, zmin, zmax))
            tmp_geo = self.coarse_voxel[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]
            tmp_geo = np.pad(tmp_geo, ((mask_margin_x, mask_margin_x), (mask_margin_y, mask_margin_y), (mask_margin_z, mask_margin_z)), 'constant', constant_values=0)

            # coarse_voxel = torch.from_numpy(self.coarse_voxel).to(device).unsqueeze(0).unsqueeze(0).float()
            # Gmask_fake = coarse_voxel
            coarse_voxel = torch.from_numpy(tmp_geo).to(device).unsqueeze(0).unsqueeze(0).float()
            Gmask_fake = F.max_pool3d(coarse_voxel, kernel_size=3, stride=1, padding=1)

            print(" [*] Start detailization")

            with torch.no_grad():
                z_vector_geometry = np.zeros([self.styleset_len], np.float32)
                z_vector_geometry[self.selected_style_index - 1] = 1
                z_geometry_tensor = torch.from_numpy(z_vector_geometry).to(device).view([1, -1])
                z_vector_texture = np.zeros([self.styleset_len], np.float32)
                z_vector_texture[self.selected_style_index - 1] = 1
                z_texture_tensor = torch.from_numpy(z_vector_texture).to(device).view([1, -1])
                z_geometry_code = torch.matmul(z_geometry_tensor, self.network.geometry_codes).view([1, -1, 1, 1, 1])
                z_texture_code = torch.matmul(z_texture_tensor, self.network.texture_codes).view([1, -1, 1, 1, 1])
                voxel_fake, texture_fake = self.network(coarse_voxel, z_geometry_code, z_texture_code, Gmask_fake, is_geometry_training=False)

            print(" [*] Finish detailization")

            out_geometry = voxel_fake.detach().cpu().numpy()[0, 0]
            out_color = texture_fake[0, :].permute(1, 2, 3, 0).detach().cpu().numpy()
            # texture_voxel = np.round(np.clip(out_color * 255, a_min=0, a_max=255)).astype(np.uint8)
            texture_voxel = np.clip(out_color, a_min=0, a_max=1)
            vertices, triangles, normals, _ = measure.marching_cubes(out_geometry, 0.4)
            vertices_1 = vertices.astype(np.int32)
            vertices_2 = (vertices + 0.5).astype(np.int32)
            vertices_3 = (vertices - 0.5).astype(np.int32)
            vertices = vertices / 256 - 0.5
            colors = np.maximum(np.maximum(texture_voxel[vertices_1[:, 0], vertices_1[:, 1], vertices_1[:, 2]],
                                           texture_voxel[vertices_2[:, 0], vertices_2[:, 1], vertices_2[:, 2]]),
                                           texture_voxel[vertices_3[:, 0], vertices_3[:, 1], vertices_3[:, 2]])

            mesh = o3d.t.geometry.TriangleMesh()
            try:
                mesh.vertex.positions = o3d.core.Tensor(vertices)
                mesh.triangle.indices = o3d.core.Tensor(triangles)
                mesh.triangle.normals = o3d.core.Tensor(normals)
                mesh.triangle.colors = o3d.core.Tensor(colors)
            except:
                mesh.vertex["positions"] = o3d.core.Tensor(vertices)
                mesh.triangle["indices"] = o3d.core.Tensor(triangles)
                mesh.triangle["normals"] = o3d.core.Tensor(normals)
                mesh.vertex["colors"] = o3d.core.Tensor(colors)
            self.mesh = mesh

            self.cache_mesh = True

        material = rendering.MaterialRecord()
        material.shader = 'defaultLit'
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("detailed_model", self.mesh, material)
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _on_coarse_button(self):
        self.update_gui_coarse_vox()

        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _on_reset_button(self):
        self.coarse_voxel = self.coarse_voxel_for_reset.copy()

        self.update_gui_coarse_vox()
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _on_import_button(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to open",
                             self.window.theme)
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_import_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_button(self):
        # if self.mesh is None:
        #     return None

        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        try:
            o3d.io.write_triangle_mesh(filename, self.mesh, write_vertex_colors=True)

            print(f"Save mesh to '{filename}'")
        except Exception as e:
            print("Save mesh failed.", e)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_import_dialog_done(self, filename):
        self.window.close_dialog()

        self._scene.scene.clear_geometry()
        self.cache_mesh = False

        if filename.endswith(".binvox"):
            init_coarse_voxel = get_vox_from_binvox_1over2_return_small(filename).astype(np.uint8)
            vox_tensor = torch.from_numpy(init_coarse_voxel).to(device).unsqueeze(0).unsqueeze(0).float()
            smallmaskx_tensor = F.max_pool3d(vox_tensor, kernel_size=self.upsample_rate, stride=self.upsample_rate, padding=0)
            downsampled_geometry = smallmaskx_tensor.detach().cpu().numpy()[0, 0]
            downsampled_geometry = np.round(downsampled_geometry).astype(np.uint8)

            self.coarse_voxel = downsampled_geometry.copy()
            self.coarse_voxel_for_reset = downsampled_geometry.copy()

            xmin, xmax = np.nonzero(np.sum(self.coarse_voxel, axis=(1, 2)))[0][[0, -1]]
            ymin, ymax = np.nonzero(np.sum(self.coarse_voxel, axis=(0, 2)))[0][[0, -1]]
            zmin, zmax = np.nonzero(np.sum(self.coarse_voxel, axis=(0, 1)))[0][[0, -1]]
            print("xmin: {}, xmax: {}, ymin: {}, ymax: {}, zmin: {}, zmax: {}".format(xmin, xmax, ymin, ymax, zmin, zmax))

            self.update_gui_coarse_vox()
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())

        else:
            self.window.show_message_box("Unsupported file type.")

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name, index):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name, index):
        self.settings.new_ibl_name = gui.Application.instance.resource_path + "/" + name
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color):
        self.settings.sun_color = color
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_material_prefab(self, name, index):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_mouse_widget3d(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN:

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                    # w.window.show_message_box("Invalid region, please re-select...")
                    # self.window.show_message_box("Warning msg box", "Invalid region, please re-select...")
                else:
                    world = self._scene.scene.camera.unproject(x, y, depth,
                                                               self._scene.frame.width, self._scene.frame.height) * self.coarse_res
                    text = "(World coordinates: {:.6f}, {:.6f}, {:.6f}))".format(world[0], world[1], world[2])
                    print(text)

                    if len(self.selected_positions) >= 2:
                        self.selected_positions.pop(0)

                    selected_x = 1 if world[0] > 1 else (0 if world[0] < 0 else world[0])
                    selected_y = 1 if world[1] > 1 else (0 if world[1] < 0 else world[1])
                    selected_z = 1 if world[2] > 1 else (0 if world[2] < 0 else world[2])
                    self.selected_positions.append([selected_x, selected_y, selected_z])
                    print("Selected region: ", self.selected_positions)

                    which_axis = np.argmin(abs(np.round(world) - world))

                    # Add voxel, option key on Mac
                    if event.is_modifier_down(gui.KeyModifier.CTRL):

                        print("Add at which axis: ", which_axis)

                        # Add voxel from x-axis, be careful with boundary condition
                        if which_axis == 0:
                            if self.coarse_voxel[min(int(np.round(world[0])), self.coarse_res - 1), int(world[1]), int(world[2])] == 0:
                                x1 = max(0, int(np.round(world[0])) - self.brush_size)
                                x2 = min(self.coarse_res, int(np.round(world[0])) + self.brush_size + 1)
                            else:
                                x1 = max(0, int(np.round(world[0])) - 1 - self.brush_size)
                                x2 = min(self.coarse_res, int(np.round(world[0])) + self.brush_size)
                            y1, y2 = max(0, int(world[1]) - self.brush_size), min(self.coarse_res, int(world[1]) + 1 + self.brush_size)
                            z1, z2 = max(0, int(world[2]) - self.brush_size), min(self.coarse_res, int(world[2]) + 1 + self.brush_size)

                        # Add voxel from y-axis, be careful with boundary condition
                        elif which_axis == 1:
                            x1, x2 = max(0, int(world[0]) - self.brush_size), min(self.coarse_res, int(world[0]) + 1 + self.brush_size)
                            if self.coarse_voxel[int(world[0]), min(int(np.round(world[1])), self.coarse_res - 1), int(world[2])] == 0:
                                y1 = max(0, int(np.round(world[1])) - self.brush_size)
                                y2 = min(self.coarse_res, int(np.round(world[1])) + self.brush_size + 1)
                            else:
                                y1 = max(0, int(np.round(world[1])) - 1 - self.brush_size)
                                y2 = min(self.coarse_res, int(np.round(world[1])) + self.brush_size)
                            z1, z2 = max(0, int(world[2]) - self.brush_size), min(self.coarse_res, int(world[2]) + 1 + self.brush_size)

                        # Add voxel from z-axis, be careful with boundary condition
                        else:
                            x1, x2 = max(0, int(world[0]) - self.brush_size), min(self.coarse_res, int(world[0]) + 1 + self.brush_size)
                            y1, y2 = max(0, int(world[1]) - self.brush_size), min(self.coarse_res, int(world[1]) + 1 + self.brush_size)
                            if self.coarse_voxel[int(world[0]), int(world[1]), min(int(np.round(world[2])), self.coarse_res - 1)] == 0:
                                z1 = max(0, int(np.round(world[2])) - self.brush_size)
                                z2 = min(self.coarse_res, int(np.round(world[2])) + self.brush_size + 1)
                            else:
                                z1 = max(0, int(np.round(world[2])) - 1 - self.brush_size)
                                z2 = min(self.coarse_res, int(np.round(world[2])) + self.brush_size)
                        print("Voxel coordinates: ", x1, x2, y1, y2, z1, z2)

                        self.coarse_voxel[x1:x2, y1:y2, z1:z2] = 1
                        self.update_gui_coarse_vox()
                        self.cache_mesh = False

                    # Delete voxel, shift key on Mac
                    elif event.is_modifier_down(gui.KeyModifier.SHIFT):
                        if which_axis == 0:
                            if self.coarse_voxel[int(np.round(world)[0]), int(world[1]), int(world[2])] == 0:
                                x1 = max(0, int(np.round(world[0])) - 1 - self.brush_size)
                                x2 = min(self.coarse_res, int(np.round(world[0])) + self.brush_size)
                            else:
                                x1 = max(0, int(np.round(world[0])) - self.brush_size)
                                x2 = min(self.coarse_res, int(np.round(world[0])) + self.brush_size + 1)
                            y1, y2 = max(0, int(world[1]) - self.brush_size), min(self.coarse_res, int(world[1]) + 1 + self.brush_size)
                            z1, z2 = max(0, int(world[2]) - self.brush_size), min(self.coarse_res, int(world[2]) + 1 + self.brush_size)
                        elif which_axis == 1:
                            x1, x2 = max(0, int(world[0]) - self.brush_size), min(self.coarse_res, int(world[0]) + 1 + self.brush_size)
                            if self.coarse_voxel[int(world[0]), int(np.round(world)[1]), int(world[2])] == 0:
                                y1 = max(0, int(np.round(world[1])) - 1 - self.brush_size)
                                y2 = min(self.coarse_res, int(np.round(world[1])) + self.brush_size)
                            else:
                                y1 = max(0, int(np.round(world[1])) - self.brush_size)
                                y2 = min(self.coarse_res, int(np.round(world[1])) + self.brush_size + 1)
                            z1, z2 = max(0, int(world[2]) - self.brush_size), min(self.coarse_res, int(world[2]) + 1 + self.brush_size)
                        else:
                            x1, x2 = max(0, int(world[0]) - self.brush_size), min(self.coarse_res, int(world[0]) + 1 + self.brush_size)
                            y1, y2 = max(0, int(world[1]) - self.brush_size), min(self.coarse_res, int(world[1]) + 1 + self.brush_size)
                            if self.coarse_voxel[int(world[0]), int(world[1]), int(np.round(world)[2])] == 0:
                                z1 = max(0, int(np.round(world[2])) - 1 - self.brush_size)
                                z2 = min(self.coarse_res, int(np.round(world[2])) + self.brush_size)
                            else:
                                z1 = max(0, int(np.round(world[2])) - self.brush_size)
                                z2 = min(self.coarse_res, int(np.round(world[2])) + self.brush_size + 1)
                        print("Voxel coordinates: ", x1, x2, y1, y2, z1, z2)

                        self.coarse_voxel[x1:x2, y1:y2, z1:z2] = 0
                        self.update_gui_coarse_vox()
                        self.cache_mesh = False

            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def init_coarse_vox(self):

        self.brush_label = 1

        self._scene.scene.clear_geometry()

        init_coarse_voxel = np.zeros((self.coarse_res, self.coarse_res, self.coarse_res), np.uint8)
        init_coarse_voxel[5:28, 11:21, 2:7] = 1
        self.coarse_voxel = init_coarse_voxel.copy()
        self.coarse_voxel_for_reset = init_coarse_voxel.copy()

        vertices, triangles, normals, colors = cutils.colored_dual_contouring(self.coarse_voxel)
        colors = np.clip(colors * 2 - np.min(colors), 0, 255).astype(np.float32) / 255
        vertices = vertices / self.coarse_voxel.shape[0]

        mesh = o3d.t.geometry.TriangleMesh()
        try:
            mesh.vertex.positions = o3d.core.Tensor(vertices)
            mesh.triangle.indices = o3d.core.Tensor(triangles)
            mesh.triangle.normals = o3d.core.Tensor(normals)
            mesh.triangle.colors = o3d.core.Tensor(colors)
        except:
            #  old version of open3d
            mesh.vertex["positions"] = o3d.core.Tensor(vertices)
            mesh.triangle["indices"] = o3d.core.Tensor(triangles)
            mesh.triangle["normals"] = o3d.core.Tensor(normals)
            mesh.triangle["colors"] = o3d.core.Tensor(colors)

        self._scene.scene.add_geometry("coarse_voxel", mesh, self.settings.material)
        bounds = self._scene.scene.bounding_box
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def update_gui_coarse_vox(self):

        vertices, triangles, normals, colors = cutils.colored_dual_contouring(self.coarse_voxel)
        colors = np.clip(colors * 2 - np.min(colors), 0, 255).astype(np.float32) / 255
        vertices = vertices / self.coarse_voxel.shape[0]

        mesh = o3d.t.geometry.TriangleMesh()
        try:
            mesh.vertex.positions = o3d.core.Tensor(vertices)
            mesh.triangle.indices = o3d.core.Tensor(triangles)
            mesh.triangle.normals = o3d.core.Tensor(normals)
            mesh.triangle.colors = o3d.core.Tensor(colors)
        except:
            mesh.vertex["positions"] = o3d.core.Tensor(vertices)
            mesh.triangle["indices"] = o3d.core.Tensor(triangles)
            mesh.triangle["normals"] = o3d.core.Tensor(normals)
            mesh.triangle["colors"] = o3d.core.Tensor(colors)

        # self._scene.scene.remove_geometry("coarse_voxel")
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("coarse_voxel", mesh, self.settings.material)

    def load_model(self, args):
        g_dim = CATEGORY_SETTING[args.category]['g_dim']
        z_dim = CATEGORY_SETTING[args.category]['z_dim']
        path = CATEGORY_SETTING[args.category]['cpk_path']
        styleset_len = CATEGORY_SETTING[args.category]['styleset_len']

        if not os.path.exists(path):
            self.window.show_message_box("Error", f"Path '{path}' not exists.")
            print(f"Path '{path}' not exists.")
            return

        print(" [*][{}] Loading model...".format(path))
        self.network = generator_dual_halfsize_x8(g_dim, styleset_len, z_dim)
        self.network.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)['generator'])
        self.network.to(device)
        print(" [*][{}] Load SUCCESS".format(path))


def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1536, 768, args)

    w.init_coarse_vox()
    w.load_model(args)

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
