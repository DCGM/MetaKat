#$ -l ram_free=5G,gpu=0.16,gpu_ram=24G,tmp_free=5G
#$ -o /mnt/matylda1/ikohut/SGE_test/out.out
#$ -e /mnt/matylda1/ikohut/SGE_test/err.err
#$ -q long.q@supergpu1,long.q@supergpu2,long.q@supergpu3,long.q@supergpu4,long.q@supergpu5,long.q@supergpu6,long.q@supergpu7,long.q@supergpu8,long.q@supergpu9,long.q@supergpu10,long.q@supergpu11,long.q@supergpu12,long.q@supergpu13,long.q@supergpu14,long.q@supergpu15,long.q@supergpu16,long.q@supergpu17,long.q@supergpu18
#$ -pe smp 6

source ~/Desktop/bakalarka/MetaKat/venv/bin/activate

ulimit -t unlimited
ulimit -v unlimited

rm -r ~/.nv/

echo
echo WORK_DIRECTORY $WORK_DIRECTORY
echo HOST $(hostname)

DIGILINKA=~/Desktop/bakalarka/MetaKat
export PYTHONPATH=${DIGILINKA}
HG_CACHE=~/Desktop/bakalarka/huggingface/hub
export HUGGINGFACE_HUB_CACHE=${HG_CACHE}
export HF_HUB_OFFLINE=0

OUTPUT=~/Desktop/bakalarka/experiments/training

#DATASET
IMAGES_DIR_HOST=~/Desktop/bakalarka/data/images
DATASET_BASE_DIR=~/Desktop/bakalarka/data/ann
TRN_FILE=${DATASET_BASE_DIR}/pages.trn
TST_FILE=${DATASET_BASE_DIR}/pages.tst

DATALOADER_NUM_WORKERS=4

#MODEL
MODEL_NAME=google/vit-base-patch16-224
NET=vit-base-patch16-224

#TRAINING
LEARNING_RATE=0.00005
MAX_STEPS=25000
WARMPUP_STEPS=2500
LR_SCHEDULER_TYPE="constant_with_warmup"
#LR_SCHEDULER_TYPE="reduce_lr_on_plateau"
#LR_SCHEDULER_KWARGS='{"factor": 0.5, "patience": 10, "min_lr": 0.00001, "cooldown": 40}'
TRAIN_BATCH_SIZE=32 #128 orig

#EVALUATION
EVAL_STEPS=100
EVAL_BATCH_SIZE=32 #128 orig

#LOGGING
LOGGING_STEPS=100
LOGGING_LEVEL="INFO"

DATE=$(date +%Y-%m-%d)
if [ $# -eq 0 ]; then
  SUFFIX=""
else
  SUFFIX="."$1
fi

OUTPUT_BASE_DIR=${OUTPUT}/${NET}.${DATE}${SUFFIX}
LOG_NAME=${NET}.${DATE}${SUFFIX}.log

mkdir -p "${OUTPUT_BASE_DIR}"

PROJECT_NAME="page_type.${DATE}"
TASK_NAME=${NET}.${DATE}${SUFFIX}

#SHOW
RENDER_DIR=${OUTPUT_BASE_DIR}/show/

mkdir -p "${RENDER_DIR}"


#SAVE
SAVE_STEPS=5000
CHECKPOINT_DIR=${OUTPUT_BASE_DIR}/checkpoints/

mkdir -p "${CHECKPOINT_DIR}"

python -u ${DIGILINKA}/metakat/page_type/nets/train.py --project-name ${PROJECT_NAME} \
                   --task-name "${TASK_NAME}" \
                   --images-dir ${IMAGES_DIR_HOST} \
		   --train-pages ${TRN_FILE} \
		   --eval-pages ${TST_FILE} \
                   --dataloader-num-workers ${DATALOADER_NUM_WORKERS} \
                   --model-name ${MODEL_NAME} \
                   --fp16 \
		   --eval-train-dataset \
		   --render-dir ${RENDER_DIR} \
		   --eval-steps ${EVAL_STEPS} \
                   --learning-rate ${LEARNING_RATE} \
                   --max-steps ${MAX_STEPS} \
                   --warmup-steps ${WARMPUP_STEPS} \
                   --lr-scheduler-type ${LR_SCHEDULER_TYPE} \
		   --train-batch-size ${TRAIN_BATCH_SIZE} \
                   --eval-batch-size ${EVAL_BATCH_SIZE} \
                   --save-steps ${SAVE_STEPS} \
                   --checkpoint-dir ${CHECKPOINT_DIR} \
                   --logging-steps ${LOGGING_STEPS} \
                   --logging-level ${LOGGING_LEVEL} >> "${OUTPUT_BASE_DIR}"/"${LOG_NAME}" 2>&1

echo DONE

