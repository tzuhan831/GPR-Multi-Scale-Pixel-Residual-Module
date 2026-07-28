#!/usr/bin/env bash
# ============================================================
# ISPRS Vaihingen (5 classes, IRRG)
# baseline (MCTformer+，無 GPR) 訓練 → gen_attention_maps → eval
#
# 對應論文表4「MCTformer+ (baseline)」該列（mIoU 40.92）。
# 要跑「+ GPR」預設設定，改用 run_vaihingen_pr.sh。
# ============================================================
set -e
START_TIME=$(date +%s)
cd "$(dirname "$0")"

EXP=vaihingen_baseline
OUT=saved_model/${EXP}
LAYER=12
CAM_DIR=${OUT}/cam-npy-layer${LAYER}

DATA_PATH=../data/Vaihingen/voc12/VOCdevkit/VOC2012
IMG_LIST=../data/Vaihingen
LABEL_FILE=../data/Vaihingen/cls_labels.npy
PRETRAINED=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

mkdir -p ${OUT}

# ============= Train =============
echo "==================== Vaihingen baseline train ===================="
if [ ! -f ${OUT}/checkpoint.pth ]; then
    python main.py  --data-path ${DATA_PATH} \
                    --img-list ${IMG_LIST} \
                    --data-set Vaihingen \
                    --label-file-path ${LABEL_FILE} \
                    --output_dir ${OUT} \
                    --finetune ${PRETRAINED} \
                    --input-size 448 \
                    --batch-size 32
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
                    --input-size 448
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
    echo "Settings   : input=448, batch=32, epochs=45 (default), num_classes=5, no-background, CRF"
    echo "Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Total time : %02d:%02d:%02d\n' "$HOURS" "$MINUTES" "$SECONDS"
    echo "---"
    cat "${CAM_DIR}/evallog.txt"
} >> "${RESULT_FILE}"

echo
echo "==================================================="
echo "[Done] Vaihingen baseline"
printf "Total time: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
echo "Result: ${RESULT_FILE}"
echo "==================================================="
