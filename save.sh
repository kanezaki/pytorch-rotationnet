#!/bin/bash
#AKBATCH -r any_1
#SBATCH -N 1
#SBATCH -J job_name
#SBATCH -o slurm-%J.out

for c in $(cat ~/rotationnet/classes.txt)
do
    echo "--- now training for $c ---"
    python save_scores.py --pretrained -a alexnet -b 400 --lr 0.01 --epochs 200 ./ModelNet40v2/train_${c}.txt --input1 ./classes.txt --output_file  ./pose_reference/train_${c}_log.txt
done
