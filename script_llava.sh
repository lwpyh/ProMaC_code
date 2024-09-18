#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 12      # Request 1 core
#$ -l h_rt=11:0:0  # Request 1 hour runtime
#$ -l h_vmem=11G   # Request 1GB RAM
#$ -l gpu=1     # request 1 GPU

#module load python
module load gcc/10.2.0
module load anaconda3
module load cuda/11.8.0
module load openssl/1.1.1s

#virtualenv pytorchenv_n
# source GenSAM_LLaVA/bin/activate
source /data/DERI-Gong/jl010/envLISA2/bin/activate
config=CHAMELEON
python main.py --config config/$config.yaml 

