#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

WORK_DIR="/mnt/M3NET-main"

DATASET="MELD"
EXP_NO="q_learning_avl"

echo "${DATASET}"

LOG_PATH="${WORK_DIR}/logs/${DATASET}"

GAMMA="0.9 0.95 0.99"  # [8 ,16, 32, 64]
APLHA="0.1 0.3" # [0.0001, 0.0003]
LR="0.0005 0.001"   # [0.0005, 0.001]
L2="0.0001 0.0005"  # [0.0001, 0.0005]
DP="0.2 0.5" # [0.2 0.4]
BATCH="8 16" # [0.2 0.4]

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
for batch in ${BATCH}
do
    echo ""
    echo "======================================================================================================================"
    echo "DATASET: ${DATASET}, LR: ${lr}, L2: ${l2}, APLHA: ${alpha}, GAMMA: ${gamma}"
    python -u RL-train.py \
    --base-model="GRU" \
    --dataset_name ${DATASET} \
    --dropout ${dropout} \
    --lr ${lr} \
    --l2 ${l2} \
    --batch-size ${batch} \
    --graph_type="GCN3" \
    --epochs=50 \
    --graph_construct="direct" \
    --multi_modal \
    --mm_fusion_mthd="concat_subsequently" \
    --modals="avl" \
    --norm BN \
    --gamma ${gamma} \
    --alpha2 ${alpha} \
    >> ${LOG_PATH}/${EXP_NO}.out 2>&1

done
done
done
done
done
done
