#!/usr/bin/env bash
# ============================================================
# DeepGlobe Land Cover Classification (6 classes)
# baseline 訓練 → gen_attention_maps → eval
#
# DeepGlobe 跟 Vaihingen/Postdam 的差別：
#   - 64200 train tiles（Postdam 19391、Vaihingen 1504），總量 33× 大
#   - tiles 是 native 224×224（不需要 448 augmentation 放大）
#   - 6 類（不是 5）：Urban, Agriculture, Rangeland, Forest, Water, Barren
#   - Agriculture 嚴重主導（train 內 44289/64200 ≈ 69%）
#   - val split = 161 source parents / 16100 tiles（已 by-parent 切，無 leakage）
#
# 訓練設定（對齊 Postdam upsample ratio + scheduler shape）：
#   --input-size 448     ← Postdam tile 也是 224 native + 448 input，2× upsample 當 augmentation
#   --batch-size 32      ← in448 attn N² 16× 大，配對 Postdam 設定
# 2026-05-19：改用 Postdam 完整 45ep 配方對齊（之前 5ep 等比縮放 PR 沒學起來）
#   --epochs 45          ← Postdam default
#   --warmup-epochs 5    ← Postdam default
#   --decay-epochs 30    ← Postdam default
#   --cooldown-epochs 10 ← Postdam default
#
# 總 iter = 45ep × 64200/32 = 90281（≈ Postdam 45ep 27270 iter 的 3.3×）
# 預估時間：train ~45-55h + gen_attn ~50 min + eval ~20 min ≈ 2 天/run
# ============================================================
set -e
START_TIME=$(date +%s)
cd /workspace/MCTformer

# 2026-05-19：lr1 對照系列，避免覆蓋舊 baseline 結果（deepglobe_baseline 73.41 mIoU）
# EXP=deepglobe_baseline
EXP=deepglobe_baseline_v2
OUT=saved_model/${EXP}
LAYER=12
CAM_DIR=${OUT}/cam-npy-layer${LAYER}

DATA_PATH=../data/DeepGlobe/voc12/VOCdevkit/VOC2012
IMG_LIST=../data/DeepGlobe
LABEL_FILE=../data/DeepGlobe/cls_labels.npy
PRETRAINED=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

mkdir -p ${OUT}

# ============= Train =============
echo "==================== DeepGlobe baseline train ===================="
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
                    --seed 0
else
    echo "SKIP train（ckpt 已存在）"
fi

# ============= gen_attention_maps (CAM on train_aug split) =============
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
                    --input-size 448
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
                    --out-dir ${OUT}/pseudo-mask-layer${LAYER} \
                    --comment "${EXP}_layer${LAYER}"
                    # 2026-05-18 拿掉 --out-crf：DG 64200 tile + CRF 預估 13h，無 CRF ~1.5h
                    # pseudo-mask-layer12 是未套 CRF 的原始 argmax 結果（無「-crf-」字樣以免誤導）

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
echo "[Done] DeepGlobe baseline"
printf "Total time: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
echo "Result: ${RESULT_FILE}"
echo "==================================================="