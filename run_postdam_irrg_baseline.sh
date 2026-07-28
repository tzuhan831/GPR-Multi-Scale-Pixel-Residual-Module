#!/usr/bin/env bash
# Postdam IRRG (NIR+R+G) — baseline (no PR).
# JPEGs are IRRG patches; labels / id lists / SegmentationClass shared with original Postdam.
set -e
START_TIME=$(date +%s)

EXP=Postdam_irrg_baseline
OUT=saved_model/${EXP}
LAYER=12
CAM_DIR=${OUT}/cam-npy-layer${LAYER}

DATA_IRRG=../data/Postdam_IRRG/voc12/VOCdevkit/VOC2012
DATA_META=../data/Postdam
GT_DIR=../data/Postdam/voc12/VOCdevkit/VOC2012/SegmentationClass
IMG_DIR=../data/Postdam_IRRG/voc12/VOCdevkit/VOC2012/JPEGImages

mkdir -p ${OUT}

# Train
python main.py  --data-path ${DATA_IRRG} \
                --img-list ${DATA_META} \
                --data-set Postdam \
                --label-file-path ${DATA_META}/cls_labels.npy \
                --output_dir ${OUT} \
                --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
                --batch-size 32

# gen_attention_maps (CAM)
python main.py  --data-set PostdamMS \
                --img-list ${DATA_META} \
                --data-path ${DATA_IRRG} \
                --label-file-path ${DATA_META}/cls_labels.npy \
                --output_dir ${OUT} \
                --gen_attention_maps \
                --cam-npy-dir ${CAM_DIR} \
                --resume ${OUT}/checkpoint.pth \
                --layer-index ${LAYER}

# Evaluation
python evaluation.py --list ${DATA_META}/train_id.txt \
                     --gt_dir ${GT_DIR} \
                     --logfile ${CAM_DIR}/evallog.txt \
                     --type npy \
                     --curve True \
                     --predict_dir ${CAM_DIR} \
                     --num_classes 5 \
                     --dataset postdam \
                     --no-background \
                     --out-crf \
                     --img_dir ${IMG_DIR} \
                     --out-dir ${OUT}/pseudo-mask-crf-layer${LAYER} \
                     --comment "${EXP}_layer${LAYER}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
RESULT_FILE="${OUT}/evalresult.txt"
{
    echo "Experiment : ${EXP}"
    echo "Channels   : IRRG (Postdam 3_Ortho_IRRG)"
    echo "PR flags   : (none, baseline)"
    echo "Eval split : train_id.txt"
    printf 'Total time : %02d:%02d:%02d\n' "$HOURS" "$MINUTES" "$SECONDS"
    echo "---"
    cat "${CAM_DIR}/evallog.txt"
    echo "====================================="
} >> "${RESULT_FILE}"
