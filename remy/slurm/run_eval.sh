#!/bin/bash
#SBATCH --job-name=remy-eval
#SBATCH --account=iris
#SBATCH --partition=sc-freecpu
#SBATCH --time=300:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/remy_%j.out
#SBATCH --error=logs/remy_%j.err




cd .
. .venv/bin/activate
python ./scripts/plot.py checkpoints-14449659/brain.837 --link-ppt 0.237 9.49 --num-points 9 --delay 150 --nsenders 2 --mean-on 5000 --mean-off 5000 --results-dir brain-plot-14449659 --no-console-output-files --cf configs/link-1x.cfg --hidden-size 32 --num-hidden-layers 2 --sender neural



