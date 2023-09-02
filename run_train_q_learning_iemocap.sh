#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/mnt/M3NET-main"

DATASET="IEMOCAP"
EXP_NO="q_learning_v"

echo "${DATASET}"

LOG_PATH="${WORK_DIR}/logs/${DATASET}"

GAMMA="0"  # [8 ,16, 32, 64]
APLHA="0.2 0.5 0.7 1" # [0.0001, 0.0003]
LR="0.0005 0.001"   # [0.0005, 0.001]
L2="0.0001 0.0005"  # [0.0001, 0.0005]
DP="0.2 0.4" # [0.2 0.4]


for lr in ${LR}
do
for l2 in ${L2}
do
for dropout in ${DP}
do
for alpha in ${APLHA}
do
for gamma in ${GAMMA}
do
    echo ""
    echo "======================================================================================================================"
    echo "DATASET: ${DATASET}, LR: ${lr}, L2: ${l2}, APLHA: ${alpha}, GAMMA: ${gamma}"
    python -u RL_train.py \
    --base-model="GRU" \
    --dataset_name ${DATASET} \
    --dropout 0.4 \
    --lr ${lr} \
    --l2 ${l2} \
    --batch-size 8 \
    --graph_type="GCN3" \
    --epochs=50 \
    --graph_construct="direct" \
    --multi_modal \
    --mm_fusion_mthd="concat_subsequently" \
    --modals="avl" \
    --dropout ${dropout} \
    --norm BN \
    --gamma ${gamma} \
    --alpha2 ${alpha} \
    >> ${LOG_PATH}/${EXP_NO}.out 2>&1

done
done
done
done
done