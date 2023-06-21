#!/bin/bash

### Feature Table ###
# ca_housing 8
# YearPredictionMSD 90
# slice_localization 384
dataset=ML_Ready_Data_FCR_cm

BASEDIR=$(dirname "$0")
OUTDIR="${BASEDIR}/ckpt/"

if [ ! -d "${OUTDIR}" ]
then   
    echo "Output dir ${OUTDIR} does not exist, creating..."
    mkdir -p ${OUTDIR}
fi

# Prompt user to resume training or start fresh
read -p "Resume training from ${dataset}? (y/n): " resume_choice

# Set the output file based on user's choice
if [ "$resume_choice" == "y" ]; then
    resume_choice=true
else
    resume_choice=false
fi

CUDA_VISIBLE_DEVICES=0 python -u main_reg_cv.py \
    --feat_d 6 \
    --hidden_d 32 \
    --boost_rate 0.5 \
    --lr 0.0036 \
    --L2 .0e-3 \
    --num_nets 100 \
    --data ${dataset} \
    --tr ${BASEDIR}/../newData/${dataset}_tr.npz \
    --te ${BASEDIR}/../newData/${dataset}_te.npz \
    --batch_size 1024 \
    --epochs_per_stage 3 \
    --correct_epoch 3 \
    --normalization True \
    --cv True \
    --out_f ${OUTDIR}/${dataset}_cls.pt \
    --resume $resume_choice \
    --cuda
