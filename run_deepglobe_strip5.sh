#!/usr/bin/env bash
# ============================================================
# DeepGlobe：PixelResidualStem branch5「depthwise 可分離序列」變體 (--pr-strip)
#   論文表12「strip5」= 1×1 lift → dw 1×5 → dw 5×1
#   （SegNeXt MSCA 風格，完整 5×5 方形含對角，算力最省）
#   full5  = 一條 dense 5×5（對照組）
#
# 公平對照 = 同 config 的原版 full5 ni_lr1：
#   saved_model/deepglobe_pr_ni_lr1  (mIoU 77.36)
# 本腳本唯一差別 = --pr-strip；其餘 config 完全相同。
#
# 注意：train 與 gen_attention_maps 都帶 --pr-strip，否則 load_state_dict 出錯。
#   DeepGlobe 64200 tile：延用 --t 0 無 CRF（+CRF 約 13h）。
# ============================================================
set -e
START_TIME=$(date +%s)
cd "$(dirname "$0")"

EXP=deepglobe_pr_strip5
OUT=saved_model/${EXP}
LAYER=12
CAM_DIR=${OUT}/cam-npy-layer${LAYER}

DATA_PATH=../data/DeepGlobe/voc12/VOCdevkit/VOC2012
IMG_LIST=../data/DeepGlobe
LABEL_FILE=../data/DeepGlobe/cls_labels.npy
PRETRAINED=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

PR_FLAGS="--use-pixel-residual --pr-norm instance --pr-lr-mult 1 --pr-strip"

mkdir -p ${OUT}

# ============= Train =============
echo "==================== [${EXP}] train ===================="
if [ ! -f ${OUT}/checkpoint.pth ]; then
    python main.py  --data-path ${DATA_PATH} \
                    --img-list ${IMG_LIST} \
                    --data-set DeepGlobe \
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
echo "==================== [${EXP}] gen_attention_maps ===================="
if [ ! -f ${CAM_DIR}/.gen_done ]; then
    python main.py  --data-set DeepGlobeMS \
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

# ============= Evaluation =============
echo "==================== [${EXP}] eval ===================="
python evaluation.py \
                    --list ${IMG_LIST}/train_id.txt \
                    --gt_dir ${DATA_PATH}/SegmentationClass \
                    --logfile ${CAM_DIR}/evallog.txt \
                    --type npy \
                    --t 0 \
                    --predict_dir ${CAM_DIR} \
                    --num_classes 6 \
                    --dataset deepglobe \
                    --no-background \
                    --img_dir ${DATA_PATH}/JPEGImages \
                    --out-dir ${OUT}/pseudo-mask-layer${LAYER} \
                    --comment "${EXP}_layer${LAYER}"

# ============= 記錄 =============
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600)); MINUTES=$(( (ELAPSED % 3600) / 60 )); SECS=$((ELAPSED % 60))
{
    echo "Experiment : ${EXP}"
    echo "Settings   : input=448, batch=32, epochs=45 (warmup=5 decay=30 cooldown=10), num_classes=6, no-background"
    echo "PR Flags   : ${PR_FLAGS}"
    echo "Train/Val  : 64200 / 16100 tiles (by-parent split, seed=0)"
    echo "Eval split : train_id.txt"
    echo "Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Total time : %02d:%02d:%02d\n' "$HOURS" "$MINUTES" "$SECS"
    echo "---"
    cat "${CAM_DIR}/evallog.txt"
    echo "====================================="
} >> "${OUT}/evalresult.txt"

echo
echo "==================================================="
echo "[Done] ${EXP} -> ${OUT}/evalresult.txt"
printf "Total time: %02d:%02d:%02d\n" $HOURS $MINUTES $SECS
echo "比較對照：saved_model/deepglobe_pr_ni_lr1 (full5, 77.36) vs deepglobe_pr_strip5"
echo "==================================================="
