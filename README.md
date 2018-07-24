# Pytorch RotationNet

This is a pytorch implementation of RotationNet.

Asako Kanezaki, Yasuyuki Matsushita and Yoshifumi Nishida.
**RotationNet: Joint Object Categorization and Pose Estimation Using Multiviews from Unsupervised Viewpoints.** 
*CVPR*, accepted, 2018.
([pdf](https://arxiv.org/abs/1603.06208))

We used caffe for the CVPR submission.
Please see [rotationnet](https://github.com/kanezaki/rotationnet) repository for more details including how to reproduce the results in our paper.

## Training/testing ModelNet dataset

### 1. Download multi-view images generated in [Su et al. 2015]
    $ bash get_modelnet_png.sh  
[Su et al. 2015] H. Su, S. Maji, E. Kalogerakis, E. Learned-Miller. Multi-view Convolutional Neural Networks for 3D Shape Recognition. ICCV2015.  
   
### 2. Prepare dataset directories for training
    $ bash link_images.sh ./modelnet40v1png ./ModelNet40v1 1  
    $ bash link_images.sh ./modelnet40v2png ./ModelNet40_20 2  

### 3. Train your own RotationNet models
#### 3-1. Case (2): Train the model w/o upright orientation (RECOMMENDED)
    $ python train_rotationnet.py --pretrained -a alexnet -b 400 --lr 0.01 --epochs 1500 ./ModelNet40_20 | tee log_ModelNet40_20_rotationnet.txt
#### 3-2. Case (1): Train the model with upright orientation
    $ python train_rotationnet.py --case 1 --pretrained -a alexnet -b 240 --lr 0.01 --epochs 1500 ./ModelNet40v1 | tee log_ModelNet40v1_rotationnet.txt 
