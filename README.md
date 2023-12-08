# ShaDDR
PyTorch implementation for paper [ShaDDR: Interactive Example-Based Geometry and Texture Generation via 3D Shape Detailization and Differentiable Rendering](https://arxiv.org/abs/2306.04889), [Qimin Chen](https://qiminchen.github.io/), [Zhiqin Chen](https://czq142857.github.io/), [Hang Zhou](http://home.ustc.edu.cn/~zh2991/), [Hao Zhang](http://www.cs.sfu.ca/~haoz/).

### [Paper](https://arxiv.org/abs/2306.04889)  |  [Project page](https://qiminchen.github.io/shaddr/)  |   [Video]()

<img src='teaser.svg' />

## Citation
If you find our work useful in your research, please consider citing (to be updated):

	@misc{chen2023shaddr,
      title={ShaDDR: Real-Time Example-Based Geometry and Texture Generation via 3D Shape Detailization and Differentiable Rendering}, 
      author={Qimin Chen and Zhiqin Chen and Hang Zhou and Hao Zhang},
      year={2023},
      eprint={2306.04889},
      archivePrefix={arXiv}
}

## Dependencies
Requirements:
- Python 3.7 with numpy, pillow, h5py, scipy, sklearn and Cython
- [PyTorch 1.9](https://pytorch.org/get-started/locally/) (other versions may also work)
- [PyMCubes](https://github.com/pmneila/PyMCubes) (for marching cubes)
- [OpenCV-Python](https://opencv-python-tutroals.readthedocs.io/en/latest/) (for reading and writing images)

Build Cython module:
```
python setup.py build_ext --inplace
```

## Datasets and pre-trained weights
For data preparation, please see [data_preparation](https://github.com/qiminchen/ShaDDR/tree/main/data_preparation).

We provide the ready-to-use datasets here.

- [ShaDDR_data](https://drive.google.com/drive/folders/1IKm4mo08p-nex54esbEmaJiKbb-jjq_7?usp=sharing)

We also provide the pre-trained network weights.

- [ShaDDR_checkpoint](https://drive.google.com/drive/folders/1-6MZhvPL_OgAFnWwnGb9lWC7_xoQu6Aa?usp=sharing)

## Training
Make sure to train the geometry detailization first, then train texture generation.
```
python main.py --data_style style_color_car_16 --data_content content_car_train --data_dir ./data/02958343/ --alpha 0.2 --beta 10.0 --input_size 64 --output_size 512 --train --train_geo --gpu 0 --epoch 20
python main.py --data_style style_color_car_16 --data_content content_car_train --data_dir ./data/02958343/ --alpha 0.2 --beta 10.0 --input_size 64 --output_size 512 --train --train_tex --gpu 0 --epoch 20

python main.py --data_style style_color_plane_16 --data_content content_plane_train --data_dir ./data/02691156/ --alpha 0.1 --beta 10.0 --input_size 64 --output_size 512 --train --train_geo --gpu 0 --epoch 20
python main.py --data_style style_color_plane_16 --data_content content_plane_train --data_dir ./data/02691156/ --alpha 0.2 --beta 10.0 --input_size 64 --output_size 512 --train --train_tex --gpu 0 --epoch 20

python main.py --data_style style_color_chair_16 --data_content content_chair_train --data_dir ./data/03001627/ --alpha 0.3 --beta 10.0 --input_size 32 --output_size 256 --train --train_geo --gpu 0 --epoch 20
python main.py --data_style style_color_chair_16 --data_content content_chair_train --data_dir ./data/03001627/ --alpha 0.2 --beta 10.0 --input_size 32 --output_size 256 --train --train_tex --gpu 0 --epoch 20

python main.py --data_style style_color_building_8 --data_content content_building_train --data_dir ./data/00000000/ --alpha 0.5 --beta 10.0 --input_size 32 --output_size 256 --train --train_geo --asymmetry --gpu 0 --epoch 20
python main.py --data_style style_color_building_8 --data_content content_building_train --data_dir ./data/00000000/ --alpha 0.2 --beta 10.0 --input_size 32 --output_size 256 --train --train_tex --asymmetry --gpu 0 --epoch 20
```

## Testing
These are examples for testing a model trained with 16 detailed cars. For other categories, please change the commands accordingly.

### Rough qualitative testing
To output a few detailization results:
```
python main.py --data_style style_color_car_16 --data_content content_car_test --data_dir ./data/02958343/ --input_size 64 --output_size 512 --test --test_tex --gpu 0
```
The output mesh can be found in folder samples or you can specify `--sample_dir`.

### IOU, LP, Div
(Borrowed from [DECOR-GAN](https://github.com/czq142857/DECOR-GAN#iou-lp-div)) To test Strict-IOU, Loose-IOU, LP-IOU, Div-IOU, LP-F-score, Div-F-score:
```
python main.py --data_style style_color_chair_16 --data_content content_chair_test --data_dir ./data/03001627/ --input_size 32 --output_size 256 --prepvoxstyle --gpu 0
python main.py --data_style style_color_chair_16 --data_content content_chair_test --data_dir ./data/03001627/ --input_size 32 --output_size 256 --prepvox --gpu 0
python main.py --data_style style_color_chair_16 --data_content content_chair_test --data_dir ./data/03001627/ --input_size 32 --output_size 256 --evalvox --gpu 0
```
The first command prepares the patches in 16 detailed training shapes, thus --data_style is style_color_chair_16. Specifically, it removes duplicated patches in each detailed training shape and only keeps unique patches for faster computation in the following testing procedure. The unique patches are written to the folder unique_patches. Note that if you are testing multiple models, you do not have to run the first command every time -- just copy the folder unique_patches or make a symbolic link.

The second command runs the model and outputs the detailization results, in folder output_for_eval.

The third command evaluates the outputs. The results are written to folder eval_output ( result_IOU_mean.txt, result_LP_Div_Fscore_mean.txt, result_LP_Div_IOU_mean.txt ).

### Cls-score
(Borrowed from [DECOR-GAN](https://github.com/czq142857/DECOR-GAN#cls-score)) To test Cls-score:
```
python main.py --data_style style_color_chair_16 --data_content content_chair_all --data_dir ./data/03001627/ --input_size 32 --output_size 256 --prepimgreal --gpu 0
python main.py --data_style style_color_chair_16 --data_content content_chair_test --data_dir ./data/03001627/ --input_size 32 --output_size 256 --prepimg --gpu 0
python main.py --data_style style_color_chair_16 --data_content content_chair_all --data_dir ./data/03001627/ --input_size 32 --output_size 256 --evalimg --gpu 0
```
The first command prepares rendered views of all content shapes, thus --data_content is content_chair_all. The rendered views are written to the folder render_real_for_eval. Note that if you are testing multiple models, you do not have to run the first command every time -- just copy the folder render_real_for_eval or make a symbolic link.

The second command runs the model and outputs rendered views of the detailization results, in folder render_fake_for_eval.

The third command evaluates the outputs. The results are written to folder eval_output ( result_Cls_score.txt ).

## GUI
1. Build Cython module:
```
cd gui
python setup.py build_ext --inplace

```
2. Make sure you put the checkpoint.pth in the `checkpoint` folder, checkpoint can be found [here](https://drive.google.com/drive/folders/1BarCEue5fdOIOZGwJHQDwKq-PbsM0ZCb?usp=drive_link)
3. Change the `cpk_path` in the `gui_demo.py`
4. Run the GUI
```
python gui_demo.py --category 00000000
```
5. Some basic modeling operations of GUI
```
add voxel - ctrl + left click
delete voxel - shift + left click
rotate - left click + drag
zoom in/out - scroll wheel
```
GUI currently only supports editing voxel from scratch, more input formats will be supported in the future.
