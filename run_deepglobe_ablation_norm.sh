#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Ablation: PRS norm type comparison on DeepGlobe（對齊 run_vaihingen_ablation_norm.sh / run_postdam_ablation_norm.sh）
# 控制變因：只換 --pr-norm，其他超參與 run_deepglobe_pr.sh 主實驗完全一致
# （input 448 / batch 32 / 45 epoch warmup5-decay30-cooldown10 / seed 0 / 無 CRF，見 run_deepglobe_pr.sh 說明）

DATA_PATH=../data/DeepGlobe/voc12/VOCdevkit/VOC2012
IMG_LIST=../data/DeepGlobe
LABEL_FILE=../data/DeepGlobe/cls_labels.npy
PRETRAINED=https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
LAYER=12

NORMS="instance none group batch"

for NORM in $NORMS; do
    EXP="deepglobe_pr-norm-${NORM}"
    OUT="saved_model/${EXP}"
    CAM_DIR="${OUT}/cam-npy-layer${LAYER}"
    START_TIME=$(date +%s)

    echo "======================================================"
    echo "  ${EXP}"
    echo "======================================================"

    mkdir -p ${OUT}

    # Train
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
                        --use-pixel-residual \
                        --pr-norm ${NORM}
    else
        echo "SKIP train（ckpt 已存在）"
    fi

    # gen_attention_maps
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
                        --use-pixel-residual \
                        --pr-norm ${NORM}
        touch ${CAM_DIR}/.gen_done
    else
        echo "SKIP gen_attention_maps（.gen_done 已存在）"
    fi

    # Evaluation（DeepGlobe 無 CRF，見 run_deepglobe_pr.sh 註記：64200 tile + CRF 預估 13h）
    python evaluation.py --list ${IMG_LIST}/train_id.txt \
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

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(( (ELAPSED % 3600) / 60 ))
    SECONDS=$((ELAPSED % 60))
    RESULT_FILE="${OUT}/evalresult.txt"
    {
        echo "Experiment : ${EXP}"
        echo "Norm type  : ${NORM}"
        echo "Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
        printf 'Total time : %02d:%02d:%02d\n' "$HOURS" "$MINUTES" "$SECONDS"
        echo "---"
        cat "${CAM_DIR}/evallog.txt"
    } >> "${RESULT_FILE}"
    echo "[Done] Results written to ${RESULT_FILE}"
    echo ""
done
