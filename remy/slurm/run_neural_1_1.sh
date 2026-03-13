#!/bin/bash
#SBATCH --job-name=remy-cca-neural
#SBATCH --account=iris
#SBATCH --partition=sc-freecpu
#SBATCH --time=300:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/remy_%j.out
#SBATCH --error=logs/remy_%j.err




conda activate remy-rl
cd .
./src/rattrainer --cf=configs/link-1x-s2.cfg --of=checkpoints-$SLURM_JOB_ID/brain --save-every=1 --num-config-evals=40 --replay-buffer-size=3500000 --utd-ratio=5 --batch-size=3000000 --accumulation-steps=32 --hidden-size=32 --value-loss-coeff=0.1



