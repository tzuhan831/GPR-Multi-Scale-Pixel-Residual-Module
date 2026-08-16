#!/usr/bin/env bash
# ============================================================
# Postdam：PixelResidualStem branch5「depthwise 可分離序列」變體 (--pr-strip)
#   論文表12「strip5」= 1×1 lift → dw 1×5 → dw 5×1
#   （SegNeXt MSCA 風格，完整 5×5 方形含對角，算力最省）
#   full5  = 一條 dense 5×5（對照組）
#
# 公平對照 = 同 config 的原版 full5 ni_lr1：
#   saved_model/Postdam_pr_ni_lr1  (mIoU 65.46) / k13 (52.00)
# 本腳本唯一差別 = --pr-strip；其餘 config 完全相同。
#
# 注意：train 與 gen_attention_maps 都帶 --pr-strip，否則 load_state_dict 出錯。
# ============================================================
set -e
cd "$(dirname "$0")"

LAYER=12
DATA_PATH=../data/Postdam/voc12/VOCdevkit/VOC2012
IMG_LIST=../data/Postdam
LABEL_FILE=../data/Postdam/cls_labels.npy
PRETRAINED=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
PR_BASE="--use-pixel-residual --pr-norm instance --pr-lr-mult 1"

run_one () {
    local EXP=$1
    local EXTRA=$2          # "--pr-strip"=depthwise 序列方形（論文「strip5」）
    local OUT=saved_model/${EXP}
    local CAM_DIR=${OUT}/cam-npy-layer${LAYER}
    mkdir -p ${OUT}
    echo "==================== [${EXP}] train ===================="
    if [ ! -f ${OUT}/checkpoint.pth ]; then
        python main.py  --data-path ${DATA_PATH} \
                        --img-list ${IMG_LIST} \
                        --data-set Postdam \
                        --label-file-path ${LABEL_FILE} \
                        --output_dir ${OUT} \
                        --finetune ${PRETRAINED} \
                        --input-size 448 --batch-size 32 \
                        --epochs 45 --warmup-epochs 5 --decay-epochs 30 --cooldown-epochs 10 \
                        --seed 0 \
                        ${PR_BASE} ${EXTRA}
    else
        echo "SKIP train（ckpt 已存在）"
    fi

    echo "==================== [${EXP}] gen_attention_maps ===================="
    if [ ! -f ${CAM_DIR}/.gen_done ]; then
        python main.py  --data-set PostdamMS \
                        --img-list ${IMG_LIST} \
                        --data-path ${DATA_PATH} \
                        --label-file-path ${LABEL_FILE} \
                        --output_dir ${OUT} \
                        --gen_attention_maps \
                        --cam-npy-dir ${CAM_DIR} \
                        --resume ${OUT}/checkpoint.pth \
                        --layer-index ${LAYER} \
                        --input-size 448 \
                        ${PR_BASE} ${EXTRA}
        touch ${CAM_DIR}/.gen_done
    else
        echo "SKIP gen（.gen_done 已存在）"
    fi

    echo "==================== [${EXP}] eval ===================="
    python evaluation.py --list ${IMG_LIST}/train_id.txt \
                         --gt_dir ${DATA_PATH}/SegmentationClass \
                         --logfile ${CAM_DIR}/evallog.txt \
                         --type npy --t 0 \
                         --predict_dir ${CAM_DIR} \
                         --num_classes 5 --dataset postdam --no-background \
                         --out-crf \
                         --img_dir ${DATA_PATH}/JPEGImages \
                         --out-dir ${OUT}/pseudo-mask-crf-layer${LAYER} \
                         --comment "${EXP}_layer${LAYER}"

    { echo "Experiment : ${EXP}"
      echo "PR flags   : ${PR_BASE} ${EXTRA}"
      echo "Settings   : input=448 batch=32 epochs=45 seed=0, num_classes=5 no-background"
      echo "Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
      echo "---"; cat "${CAM_DIR}/evallog.txt"
      echo "====================================="; } >> "${OUT}/evalresult.txt"
    echo "[Done] ${EXP} -> ${OUT}/evalresult.txt"
}

run_one postdam_pr_strip5 "--pr-strip"

echo "=================================================================="
echo "[All Done] 比較 full5 65.46 / k13 52.00 vs postdam_pr_strip5"
echo "=================================================================="
