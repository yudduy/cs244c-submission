#!/bin/bash
#SBATCH --job-name=remy-cca-neural
#SBATCH --account=iris
#SBATCH --partition=sc-freecpu
#SBATCH --time=300:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/remy_%j.out
#SBATCH --error=logs/remy_%j.err



conda activate remy-rl
cd .
./src/rattrainer --cf=configs/link-1x.cfg --of=checkpoints-$SLURM_JOB_ID/brain --save-every=1 --num-config-evals=144 --replay-buffer-size=10500000 --utd-ratio=5 --batch-size=9000000 --accumulation-steps=128 --hidden-size=64 --value-loss-coeff=0.5 --entropy-coeff=0.001 --weight-decay=0.001

