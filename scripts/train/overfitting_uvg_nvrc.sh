#!/bin/bash
if [ "$#" -ne 8 ]; then
    echo "Incorrect number of parameters"
    exit 1
fi
GPU_ID=$1
VID=$2
LAMB=$3
SCALE=$4
LR_S1=$5
LR_S2=$6
GRAD_ACCUM=$7
BATCH_SIZE=$8
shift 8

echo "GPU_ID: ${GPU_ID}"
echo "VID: ${VID}"
echo "LAMB: ${LAMB}"
echo "SCALE: ${SCALE}"
echo "LR_S1: ${LR_S1}"
echo "LR_S2: ${LR_S2}"
echo "GRAD_ACCUM: ${GRAD_ACCUM}"
echo "BATCH_SIZE: ${BATCH_SIZE}"

WORK_DIR=${WORK_DIR:-${HOME}}
ROOT=${WORK_DIR}/Programs
TRAIN_TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim-5x5.yaml
EVAL_TASK_CFG=scripts/configs/tasks/overfit/l1_ms-ssim.yaml
COMPRESS_MODEL_CFG_S1=scripts/configs/nvrc/compress_models/nvrc_s1.yaml
COMPRESS_MODEL_CFG_S2=scripts/configs/nvrc/compress_models/nvrc_s2.yaml
MODEL_CFG_S1=scripts/configs/nvrc/models/uvg_hinerv-v2-${SCALE}_1920x1080.yaml
MODEL_CFG_S2=${MODEL_CFG_S1}
EXP_CFG_S1=scripts/configs/nvrc/overfit/s1-360e.yaml
EXP_CFG_S2=scripts/configs/nvrc/overfit/s2-30e.yaml
DATASET_DIR=${WORK_DIR}/Datasets/UVG/1920x1080
DATASET=${VID}
START_FRAME=-1
NUM_FRAMES=-1
INTRA_PERIOD=-1
FMT=png
T=-1
H=-1
W=-1
T_PATCH=1
H_PATCH=120
W_PATCH=120
NUM_PROC=1
GRAD_ACCUM=${GRAD_ACCUM}
TRAIN_BATCH_SIZE=${BATCH_SIZE}
EVAL_BATCH_SIZE=1
MODEL_NAME=nvrc_uvg_hinerv-v2-${SCALE}_1920x1080_x${GRAD_ACCUM}x${TRAIN_BATCH_SIZE}

OUTPUT=${WORK_DIR}/Outputs/NVRC/${DATASET}
EXP_NAME_S1=${MODEL_NAME}_lamb-${LAMB}_lr-${LR_S1}_s1
EXP_NAME_S2=${MODEL_NAME}_lamb-${LAMB}_lr-${LR_S2}_s2

MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}

echo "Start running the script with experiment:"
echo "    Dataset: ${DATASET}"
echo "    Output: ${OUTPUT}"
echo "    Stage 1: ${EXP_NAME_S1}"
echo "    Stage 2: ${EXP_NAME_S2}"

# S1 training
# check if output/${EXP_NAME_S1} exists, if not create it
if [ ! -d ${OUTPUT}/${EXP_NAME_S1} ]; then
    . ${WORK_DIR}/miniconda3/bin/activate
    conda activate NVRC
    cd $ROOT/NVRC && \
    accelerate launch --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
                      --gpu_ids=${GPU_ID} --num_processes=${NUM_PROC} --mixed_precision=fp16 --dynamo_backend=inductor \
    main_nvrc.py --exp-config ${EXP_CFG_S1} \
                 --output ${OUTPUT} --exp-name ${EXP_NAME_S1} \
                 --train-task-config ${TRAIN_TASK_CFG} --eval-task-config ${EVAL_TASK_CFG} \
                 --compress-model-config ${COMPRESS_MODEL_CFG_S1} --model-config ${MODEL_CFG_S1} \
                 --train-dataset-dir ${DATASET_DIR} --train-dataset ${DATASET} --train-fmt ${FMT} \
                 --lamb ${LAMB} \
                 --start-frame ${START_FRAME} --num-frames ${NUM_FRAMES} --intra-period ${INTRA_PERIOD} \
                 --train-video-size ${T} ${H} ${W} --eval-video-size ${T} ${H} ${W} \
                 --train-patch-size ${T_PATCH} ${H_PATCH} ${W_PATCH} --eval-patch-size 1 -1 -1 \
                 --grad-accum ${GRAD_ACCUM} --rate-steps 8 \
                 --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} \
                 --train-enable-log false --eval-enable-log false --log-epochs -2 \
                 --opt adam --sched cosine \
                 --lr ${LR_S1} --warmup-lr 1e-5 --min-lr 1e-4 --auto-lr-scaling true --max-norm 1.0 \
                 --workers 4 --prefetch-factor 4
else
    echo "Experiment ${EXP_NAME_S1} already exists, skipping S1 training."
fi

if [[ ! -f "${OUTPUT}/${EXP_NAME_S1}/results/all.txt" ]]; then
    echo "S1 training not completed, please check the logs."
    exit 1
fi

# S2 training
if [ ! -d ${OUTPUT}/${EXP_NAME_S2} ]; then
    . ${WORK_DIR}/miniconda3/bin/activate
    conda activate NVRC
    cd $ROOT/NVRC && \
    accelerate launch --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT} \
                      --gpu_ids=${GPU_ID} --num_processes=${NUM_PROC} --mixed_precision=fp16 --dynamo_backend=inductor \
    main_nvrc.py --exp-config ${EXP_CFG_S2} \
                 --output ${OUTPUT} --exp-name ${EXP_NAME_S2} \
                 --train-task-config ${TRAIN_TASK_CFG} --eval-task-config ${EVAL_TASK_CFG} \
                 --compress-model-config ${COMPRESS_MODEL_CFG_S2} --model-config ${MODEL_CFG_S2} \
                 --train-dataset-dir ${DATASET_DIR} --train-dataset ${DATASET} --train-fmt ${FMT} \
                 --lamb ${LAMB} \
                 --start-frame ${START_FRAME} --num-frames ${NUM_FRAMES} --intra-period ${INTRA_PERIOD} \
                 --train-video-size ${T} ${H} ${W} --eval-video-size ${T} ${H} ${W} \
                 --train-patch-size ${T_PATCH} ${H_PATCH} ${W_PATCH} --eval-patch-size 1 -1 -1 \
                 --grad-accum ${GRAD_ACCUM} --rate-steps 8 \
                 --train-batch-size ${TRAIN_BATCH_SIZE} --eval-batch-size ${EVAL_BATCH_SIZE} \
                 --train-enable-log false --eval-enable-log true --log-epochs -2 \
                 --opt adam --sched cosine \
                 --lr ${LR_S2} --warmup-lr 1e-5 --min-lr 1e-5 --auto-lr-scaling true --max-norm 1.0 \
                 --workers 4 --prefetch-factor 4 \
                 --resume ${OUTPUT}/${EXP_NAME_S1} --resume-model-only true
else
    echo "Experiment ${EXP_NAME_S2} already exists, skipping S2 training."
fi

if [ ! -f "${OUTPUT}/${EXP_NAME_S2}/results/all.txt" ]; then
    echo "S2 training not completed, please check the logs."
    exit 1
fi