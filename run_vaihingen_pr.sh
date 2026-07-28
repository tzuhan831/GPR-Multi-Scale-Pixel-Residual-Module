#!/usr/bin/env bash
# ============================================================
# Vaihingen + GPR (PixelResidualStem, kernel (1,3,5), instance norm, lr×1)
# 主實驗設定，對應論文表4「+ GPR (本研究)」該列（mIoU 49.04）。
# 跟 run_vaihingen.sh 只差 PR_FLAGS；其餘 config 完全相同。
# ============================================================
set -e
START_TIME=$(date +%s)
cd "$(dirname "$0")"

EXP=vaihingen_pr_ni_lr1
OUT=saved_model/${EXP}
LAYER=12
CAM_DIR=${OUT}/cam-npy-layer${LAYER}

DATA_PATH=../data/Vaihingen/voc12/VOCdevkit/VOC2012
IMG_LIST=../data/Vaihingen
LABEL_FILE=../data/Vaihingen/cls_labels.npy
PRETRAINED=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

PR_FLAGS="--use-pixel-residual --pr-norm instance --pr-lr-mult 1"

mkdir -p ${OUT}

# ============= Train =============
echo "==================== Vaihingen + GPR (ni_lr1) train ===================="
if [ ! -f ${OUT}/checkpoint.pth ]; then
    python main.py  --data-path ${DATA_PATH} \
                    --img-list ${IMG_LIST} \
                    --data-set Vaihingen \
                    --label-file-path ${LABEL_FILE} \
                    --output_dir ${OUT} \
                    --finetune ${PRETRAINED} \
                    --input-size 448 \
                    --batch-size 32 \
                    --epochs 45 \
                    --warmup-epochs 5 \
                    --decay-epochs 30 \
                    --cooldown-epochs 10 \
                    --seed 0 \
                    ${PR_FLAGS}
else
    echo "SKIP train（ckpt 已存在）"
fi

# ============= gen_attention_maps =============
echo "==================== gen_attention_maps ===================="
if [ ! -f ${CAM_DIR}/.gen_done ]; then
    python main.py  --data-set VaihingenMS \
                    --img-list ${IMG_LIST} \
                    --data-path ${DATA_PATH} \
                    --label-file-path ${LABEL_FILE} \
                    --output_dir ${OUT} \
                    --gen_attention_maps \
                    --cam-npy-dir ${CAM_DIR} \
                    --resume ${OUT}/checkpoint.pth \
                    --layer-index ${LAYER} \
                    --input-size 448 \
                    ${PR_FLAGS}
    touch ${CAM_DIR}/.gen_done
else
    echo "SKIP gen_attention_maps（.gen_done 已存在）"
fi

# ============= Evaluation（ISPRS 用 CRF）=============
echo "==================== Evaluation ===================="
python evaluation.py --list ${IMG_LIST}/train_id.txt \
                     --gt_dir ${DATA_PATH}/SegmentationClass \
                     --logfile ${CAM_DIR}/evallog.txt \
                     --type npy \
                     --t 0 \
                     --predict_dir ${CAM_DIR} \
                     --num_classes 5 \
                     --dataset vaihingen \
                     --no-background \
                     --out-crf \
                     --img_dir ${DATA_PATH}/JPEGImages \
                     --comment "${EXP}_layer${LAYER}"

# ============= 記錄 =============
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
RESULT_FILE="${OUT}/evalresult.txt"
{
    echo "Experiment : ${EXP}"
    echo "Settings   : input=448, batch=32, epochs=45 (warmup=5 decay=30 cooldown=10), num_classes=5, no-background, CRF"
    echo "PR Flags   : ${PR_FLAGS}"
    echo "Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Total time : %02d:%02d:%02d\n' "$HOURS" "$MINUTES" "$SECONDS"
    echo "---"
    cat "${CAM_DIR}/evallog.txt"
} >> "${RESULT_FILE}"

echo
echo "==================================================="
echo "[Done] Vaihingen + GPR (ni_lr1)"
printf "Total time: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
echo "Result: ${RESULT_FILE}"
echo "==================================================="
