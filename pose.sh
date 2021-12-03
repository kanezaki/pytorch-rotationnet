#!/bin/bash
#AKBATCH -r any_1
#SBATCH -N 1
#SBATCH -J job_name
#SBATCH -o slurm-%J.out

python pose_estimation.py --resume checkpoints/rotationnet_checkpoint.pth.tar --pretrained -a alexnet -b 3 test_images/car_000000079.txt --input1 classes.txt
