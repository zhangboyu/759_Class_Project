#!/bin/bash
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --error=/srv/home/bzhang93/759_Class_Project/project.err
#SBATCH --output=/srv/home/bzhang93/759_Class_Project/project.out
#SBATCH --gres=gpu:gtx1080:1

uname -n
for j in 0.1 0.5 0.7
do
    for i in {1..16}
    do
        echo number objects = $((2**$i)), space scaling = $j
        ./collision_detection 3 $((2**$i)) $j
    done

done
