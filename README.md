# ShapeBoost: Boosting Human Shape Estimation with Part-based Parameterization and Clothing-preserving Augmentation

The code is adapted from [Original Hybrik](https://github.com/Jeff-sjtu/HybrIK).

## Installation instructions

``` bash
# 1. Create a conda virtual environment.
conda create -n shapeboost python=3.6 -y
conda activate shapeboost

# 2. Install PyTorch
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch

# 4. Install
python setup.py develop
```

## Train from scratch

``` bash
./scripts/train_smpl_shape_ddp.sh train_shapeboost configs/hrw48_cam_2x_sratio_semi_analytical.yaml
```

## Evaluation
``` bash
./scripts/validate_smpl_shape_ddp.sh --cfg configs/NIKI_twistswing.yaml --ckpt {CKPT}
```
