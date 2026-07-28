#!/usr/bin/env bash
# Postdam 14 個剩餘訓練（PR / Stk×2 / Stk×3）× （norm: instance/batch/group）×（lr: 1/10）
# 已有 4 個 lr×1 ckpt 不重跑：pr_ni_lr1, pr_nb_lr1, pr_ng_lr1, stk2_ni_lr1
# 每個 EXP：訓練 → train_id eval → val_id eval（已存在會自動跳過）

set -e

DSET=Postdam
DSET_MS=PostdamMS
DSET_LOWER=postdam
NUM_CLASSES=5
PREFIX=../data/Postdam
IMG_CH=3
INPUT=448
NO_BG="--no-background"
PRETRAINED=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

run_full() {
    EXP=$1
    shift
    MODEL_FLAGS="$*"

    OUT=saved_model/${EXP}
    CKPT=${OUT}/checkpoint.pth
    CAM_T=${OUT}/cam-npy-layer12
    CAM_V=${OUT}/cam-npy-layer12-val
    CRF_T=${OUT}/pseudo-mask-crf-layer12
    CRF_V=${OUT}/pseudo-mask-crf-layer12-val

    echo
    echo "================================ ${EXP} ================================"
    START=$(date +%s)
    mkdir -p ${OUT}

    # ---------- 1) 訓練（已有 ckpt 跳過） ----------
    if [ ! -f ${CKPT} ]; then
        python main.py  --data-path ${PREFIX}/voc12/VOCdevkit/VOC2012 \
                        --img-list ${PREFIX} \
                        --data-set ${DSET} \
                        --label-file-path ${PREFIX}/cls_labels.npy \
                        --input-size ${INPUT} \
                        --image-ch ${IMG_CH} \
                        --output_dir ${OUT} \
                        --finetune ${PRETRAINED} \
                        --batch-size 32 \
                        ${MODEL_FLAGS}
    else
        echo "SKIP train: ${CKPT} 已存在"
    fi

    # ---------- 2) train_id eval（已存在跳過） ----------
    if [ ! -f ${CAM_T}/evallog.txt ]; then
        python main.py  --data-set ${DSET_MS} \
                        --img-list ${PREFIX} \
                        --data-path ${PREFIX}/voc12/VOCdevkit/VOC2012 \
                        --label-file-path ${PREFIX}/cls_labels.npy \
                        --input-size ${INPUT} \
                        --image-ch ${IMG_CH} \
                        --output_dir ${OUT} \
                        --gen_attention_maps \
                        --cam-npy-dir ${CAM_T} \
                        --resume ${CKPT} \
                        --layer-index 12 \
                        ${MODEL_FLAGS}

        python evaluation.py --list ${PREFIX}/train_id.txt \
                             --gt_dir ${PREFIX}/voc12/VOCdevkit/VOC2012/SegmentationClass \
                             --logfile ${CAM_T}/evallog.txt \
                             --type npy --curve True \
                             --predict_dir ${CAM_T} \
                             --num_classes ${NUM_CLASSES} --dataset ${DSET_LOWER} ${NO_BG} \
                             --out-crf \
                             --img_dir ${PREFIX}/voc12/VOCdevkit/VOC2012/JPEGImages \
                             --out-dir ${CRF_T} \
                             --comment "${EXP}_layer12"

        ELAPSED=$(($(date +%s) - START))
        {
            echo "Experiment : ${EXP}"
            echo "Eval split : train_id.txt"
            echo "Model flags: ${MODEL_FLAGS}"
            printf 'Train+eval : %02d:%02d:%02d\n' $((ELAPSED/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60))
            echo "---"
            cat ${CAM_T}/evallog.txt
            echo "====================================="
        } >> ${OUT}/evalresult.txt
    else
        echo "SKIP train_id eval: ${CAM_T}/evallog.txt 已存在"
    fi

    # ---------- 3) val_id eval（已存在跳過） ----------
    if ! grep -q 'Eval split : val_id.txt' ${OUT}/evalresult.txt 2>/dev/null; then
        VAL_START=$(date +%s)
        python main.py  --data-set ${DSET_MS} \
                        --img-list ${PREFIX} \
                        --data-path ${PREFIX}/voc12/VOCdevkit/VOC2012 \
                        --label-file-path ${PREFIX}/cls_labels.npy \
                        --input-size ${INPUT} \
                        --image-ch ${IMG_CH} \
                        --output_dir ${OUT} \
                        --gen_attention_maps --cam-on-val \
                        --cam-npy-dir ${CAM_V} \
                        --resume ${CKPT} \
                        --layer-index 12 \
                        ${MODEL_FLAGS}

        python evaluation.py --list ${PREFIX}/val_id.txt \
                             --gt_dir ${PREFIX}/voc12/VOCdevkit/VOC2012/SegmentationClass \
                             --logfile ${CAM_V}/evallog.txt \
                             --type npy --curve True \
                             --predict_dir ${CAM_V} \
                             --num_classes ${NUM_CLASSES} --dataset ${DSET_LOWER} ${NO_BG} \
                             --out-crf \
                             --img_dir ${PREFIX}/voc12/VOCdevkit/VOC2012/JPEGImages \
                             --out-dir ${CRF_V} \
                             --comment "${EXP}_layer12_VAL"

        VAL_ELAPSED=$(($(date +%s) - VAL_START))
        {
            echo "Experiment : ${EXP}"
            echo "Eval split : val_id.txt"
            printf 'Val time   : %02d:%02d:%02d\n' $((VAL_ELAPSED/3600)) $(((VAL_ELAPSED%3600)/60)) $((VAL_ELAPSED%60))
            echo "---"
            cat ${CAM_V}/evallog.txt
            echo "====================================="
        } >> ${OUT}/evalresult.txt
    else
        echo "SKIP val_id eval: 已有結果"
    fi
}

# ============================== PR (lr×10) — 3 個 ==============================
run_full Postdam_pr_ni_lr10  --use-pixel-residual --pr-norm instance --pr-lr-mult 10
run_full Postdam_pr_nb_lr10  --use-pixel-residual --pr-norm batch    --pr-lr-mult 10
run_full Postdam_pr_ng_lr10  --use-pixel-residual --pr-norm group    --pr-lr-mult 10

# ============================== Stk×2 — 5 個（缺 ni_lr10, nb×2, ng×2） =========
run_full Postdam_stk2_ni_lr10 --use-stacked-pr --stacked-pr-layers 2 --pr-norm instance --pr-lr-mult 10
run_full Postdam_stk2_nb_lr1  --use-stacked-pr --stacked-pr-layers 2 --pr-norm batch    --pr-lr-mult 1
run_full Postdam_stk2_nb_lr10 --use-stacked-pr --stacked-pr-layers 2 --pr-norm batch    --pr-lr-mult 10
run_full Postdam_stk2_ng_lr1  --use-stacked-pr --stacked-pr-layers 2 --pr-norm group    --pr-lr-mult 1
run_full Postdam_stk2_ng_lr10 --use-stacked-pr --stacked-pr-layers 2 --pr-norm group    --pr-lr-mult 10

# ============================== Stk×3 — 6 個（全新） ============================
run_full Postdam_stk3_ni_lr1  --use-stacked-pr --stacked-pr-layers 3 --pr-norm instance --pr-lr-mult 1
run_full Postdam_stk3_ni_lr10 --use-stacked-pr --stacked-pr-layers 3 --pr-norm instance --pr-lr-mult 10
run_full Postdam_stk3_nb_lr1  --use-stacked-pr --stacked-pr-layers 3 --pr-norm batch    --pr-lr-mult 1
run_full Postdam_stk3_nb_lr10 --use-stacked-pr --stacked-pr-layers 3 --pr-norm batch    --pr-lr-mult 10
run_full Postdam_stk3_ng_lr1  --use-stacked-pr --stacked-pr-layers 3 --pr-norm group    --pr-lr-mult 1
run_full Postdam_stk3_ng_lr10 --use-stacked-pr --stacked-pr-layers 3 --pr-norm group    --pr-lr-mult 10

echo
echo "==== Postdam 14 個全部完成 ===="
