#!/usr/bin/env bash
set -e

# Ablation: PRS norm type comparison on Vaihingen
# 控制變因：只換 --pr-norm，其他超參完全不動

NORMS="instance none group batch"

for NORM in $NORMS; do
    EXP="vaihingen_pr-norm-${NORM}"
    OUT="saved_model/${EXP}"
    LAYER=12
    CAM_DIR="${OUT}/cam-npy-layer${LAYER}"
    START_TIME=$(date +%s)

    echo "======================================================"
    echo "  ${EXP}"
    echo "======================================================"

    # Train
    python main.py  --data-path ../data/Vaihingen/voc12/VOCdevkit/VOC2012 \
                    --img-list ../data/Vaihingen \
                    --data-set Vaihingen \
                    --label-file-path ../data/Vaihingen/cls_labels.npy \
                    --output_dir ${OUT} \
                    --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
                    --batch-size 32 \
                    --use-pixel-residual \
                    --pr-norm ${NORM}

    # gen_attention_maps
    python main.py  --data-set VaihingenMS \
                    --img-list ../data/Vaihingen \
                    --data-path ../data/Vaihingen/voc12/VOCdevkit/VOC2012 \
                    --label-file-path ../data/Vaihingen/cls_labels.npy \
                    --output_dir ${OUT} \
                    --gen_attention_maps \
                    --cam-npy-dir ${CAM_DIR} \
                    --resume ${OUT}/checkpoint.pth \
                    --layer-index ${LAYER} \
                    --use-pixel-residual \
                    --pr-norm ${NORM}

    # Evaluation
    python evaluation.py --list ../data/Vaihingen/train_id.txt \
                         --gt_dir ../data/Vaihingen/voc12/VOCdevkit/VOC2012/SegmentationClass \
                         --logfile ${CAM_DIR}/evallog.txt \
                         --type npy \
                         --curve True \
                         --predict_dir ${CAM_DIR} \
                         --num_classes 5 \
                         --dataset vaihingen \
                         --no-background \
                         --out-crf \
                         --img_dir ../data/Vaihingen/voc12/VOCdevkit/VOC2012/JPEGImages \
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
        echo "Norm type  : ${NORM}"
        echo "Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
        printf 'Total time : %02d:%02d:%02d\n' "$HOURS" "$MINUTES" "$SECONDS"
        echo "---"
        cat "${CAM_DIR}/evallog.txt"
    } >> "${RESULT_FILE}"
    echo "[Done] Results written to ${RESULT_FILE}"
    echo ""
done
