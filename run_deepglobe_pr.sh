#!/usr/bin/env bash
# ============================================================
# DeepGlobe + PR (PixelResidualStem, ni_lr10)
# 跟 run_deepglobe.sh 只差兩處：
#   1. EXP 名稱 deepglobe_pr_ni_lr10
#   2. 加 --use-pixel-residual --pr-norm instance --pr-lr-mult 10
#
# 訓練設定保持一致，方便對比 baseline。
# 預期：跟 Postdam 比較類似（PR alone +5.09），不像 Vaihingen 弱效
#   - DG class 分布跟 Postdam 像（大區塊類）
#   - DG baseline mIoU 未知（baseline script 先跑）
# ============================================================
set -e
START_TIME=$(date +%s)
cd "$(dirname "$0")"

# 2026-05-19：改跑 lr×1（Postdam_11111111 顯示 lr×1 比 lr×10 高 0.95 mIoU，gate 動少反而結果好）
# 完整 train data 64200 tile，不跑 19k subset 計畫
# EXP=deepglobe_pr_ni_lr10
EXP=deepglobe_pr_ni_lr1
OUT=saved_model/${EXP}
LAYER=12
CAM_DIR=${OUT}/cam-npy-layer${LAYER}

DATA_PATH=../data/DeepGlobe/voc12/VOCdevkit/VOC2012
IMG_LIST=../data/DeepGlobe
LABEL_FILE=../data/DeepGlobe/cls_labels.npy
PRETRAINED=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

# PR_FLAGS="--use-pixel-residual --pr-norm instance --pr-lr-mult 10"
PR_FLAGS="--use-pixel-residual --pr-norm instance --pr-lr-mult 1"

mkdir -p ${OUT}

# ============= Train =============
echo "==================== DeepGlobe + PR (ni_lr10) train ===================="
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
echo "==================== gen_attention_maps ===================="
# 2026-05-19：舊版用 evallog.txt 當 proxy，但 evallog 是 evaluation 跑完才寫，
# eval 中斷時會誤判 gen 沒做 → 多花 52 min 重 gen。改用獨立 .gen_done marker。
# if [ ! -f ${CAM_DIR}/evallog.txt ]; then
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
echo "==================== Evaluation ===================="
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
                    --comment "${EXP}_layer${LAYER}"
                    # 2026-05-18 拿掉 --out-crf：DG 64200 tile + CRF 預估 13h，無 CRF ~1.5h
                    # 等 PR 跑出量級後再決定要不要補 CRF

# ============= 記錄 =============
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))
RESULT_FILE="${OUT}/evalresult.txt"
{
    echo "Experiment : ${EXP}"
    echo "Settings   : input=448, batch=32, epochs=45 (warmup=5 decay=30 cooldown=10), num_classes=6, no-background"
    echo "PR Flags   : ${PR_FLAGS}"
    echo "Train/Val  : 64200 / 16100 tiles (by-parent split, seed=0)"
    echo "Eval split : train_id.txt"
    echo "Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
    printf 'Total time : %02d:%02d:%02d\n' "$HOURS" "$MINUTES" "$SECONDS"
    echo "---"
    cat "${CAM_DIR}/evallog.txt"
    echo "====================================="
} >> "${RESULT_FILE}"

echo
echo "==================================================="
echo "[Done] DeepGlobe + PR (ni_lr10)"
printf "Total time: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
echo "Result: ${RESULT_FILE}"
echo "==================================================="
