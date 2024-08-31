#!/usr/local/bin/nosh

#$ -S /usr/local/bin/nosh
#$ -cwd

eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate vqmil

CUDA_VISIBLE_DEVICES=0 python test.py

conda deactivate