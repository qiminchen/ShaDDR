# ShaDDR data preparation
Data preparation code for paper ShaDDR: Interactive Example-Based Geometry and Texture Generation via 3D Shape Detailization and Differentiable Rendering. 

**NOTE**: We use the exact same content shapes as [DECOR-GAN](https://github.com/czq142857/DECOR-GAN/tree/main#datasets-and-pre-trained-weights) (Ready-to-use data is available). If you need to use your own content shapes, please refer to [DECOR-GAN](https://github.com/czq142857/DECOR-GAN/blob/main/data_preparation/README.md) for the original data preprocessing code for preparing content shapes. Here, we provide code for preparing **colored-style** shapes.

## Dependencies
Requirements:
- Python 3.x with numpy, opencv-python and cython

Build Cython module:
```
python setup.py build_ext --inplace
```

## Usage

Step 0: download ShapeNet V1 from [ShapeNet](https://www.shapenet.org/) (or prepare your own 3D colored shape in `.obj` format), and change the data directories in all python code files. Download the linux executable of [binvox](https://www.patrickmin.com/binvox/) and put it in this folder. Make sure only to put the style shapes you need in the folder, e.g.
```
├── data_preparation
├────── 03001627
│       ├── style_shape_0
│       │   ├── xxx.obj
│       │   ├── xxx.mtl
│       │   ├── ...
│       ├── style_shape_1
│       ├── ...
├────── 1_normalize_obj.py
├────── 2_voxelize.py
├────── ...
```

Step 1: run *1_normalize_obj.py* to normalize the shapes.
```
python 1_normalize_obj.py 03001627 0 1
```

Step 2: run *2_voxelize.py* to voxelize shapes using [binvox](https://www.patrickmin.com/binvox/).
```
python 2_voxelize.py 03001627 0 1
```

Step 3: run *3_depth_fusion.py* to fill the voxel.
```
python 3_depth_fusion.py 03001627 0 1
```

Step 4: run *render.py* to render texture images from `top, bottom, left, right, front, back` views.
```
./blender/blender render_views.blend --background --python render.py <class id> <process_id> <total_num_of_processes>
```

Step 5: run *4_get_colored_voxels.py* to obtain the 3D RGB color grid/voxel stored in `hdf5` format. The code runs slower than the previous ones, therefore we recommend using multiple processes:
```
python 4_get_colored_voxels.py <category_id> <process_id> <total_num_of_processes>
```
For instance, open 4 terminals and run one of the following commands in each terminal:
```
python 4_get_colored_voxels.py 03001627 0 4
python 4_get_colored_voxels.py 03001627 1 4
python 4_get_colored_voxels.py 03001627 2 4
python 4_get_colored_voxels.py 03001627 3 4
```

(Optional) Step 6: run *5_visualize.py* to export colored mesh in `.ply` format. Visualize a few shapes using MeshLab or other tools to confirm.
```
python 5_visualize.py 03001627
```
