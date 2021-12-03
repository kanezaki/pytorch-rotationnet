#!/bin/bash
#AKBATCH -r any_1
#SBATCH -N 1
#SBATCH -J job_name
#SBATCH -o slurm-%J.out

for c in $(cat classes.txt)
do
   echo "--- now training for $c ---"      
   # save predicted labels and best poses
   python save_scores.py --pretrained -a alexnet -b 400 --lr 0.01 --epochs 200 train_txt/train_${c}.txt --input1 classes.txt --output_file reference_poses/train_${c}_log.txt
		        
   # make reference poses
   python make_reference_poses.py --input1 train_txt/train_${c}.txt --input2 reference_poses/train_${c}_log.txt --output_file reference_poses/reference_poses_${c}.txt

done
