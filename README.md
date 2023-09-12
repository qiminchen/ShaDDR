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
- Python 3.7 with numpy, h5py, scipy, sklearn and Cython
- [PyTorch 1.9](https://pytorch.org/get-started/locally/) (other versions may also work)
- [PyMCubes](https://github.com/pmneila/PyMCubes) (for marching cubes)
- [OpenCV-Python](https://opencv-python-tutroals.readthedocs.io/en/latest/) (for reading and writing images)

Build Cython module:
```
python setup.py build_ext --inplace
```
