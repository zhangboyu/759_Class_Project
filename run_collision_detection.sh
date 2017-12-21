#!/bin/bash
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --error=/srv/home/bzhang93/759_Class_Project/project.err
#SBATCH --output=/srv/home/bzhang93/759_Class_Project/project.out
#SBATCH --gres=gpu:gtx1080:1

uname -n
./collision_detection $1 $2 $3 $4
