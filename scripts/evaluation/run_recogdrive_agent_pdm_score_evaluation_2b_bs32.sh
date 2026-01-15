set -x

TRAIN_TEST_SPLIT=navtest

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/dataset/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp/exp_mlp_public"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/dataset"
export PYTHONPATH="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/HunyuanWorld-Mirror:${PYTHONPATH}"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=1


CHECKPOINT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp/exp_mlp_public/training_qwen3vl_backbone_agent_dit_alignment_100epoch_LoRA_batchsize_32/checkpoints_32/lightning_logs/version_0/checkpoints/epoch=44-step=119700-EMA.ckpt"


# 1. Set NAVSIM dataset and related environment variables
# 2. Configure torchrun (e.g., single machine: --nproc_per_node=8; adjust for multi-node)
# 3. Set agent.vlm_path and agent.checkpoint_path CHECKPOINT



torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --nproc_per_node=8 \
    --node_rank=$MLP_ROLE_INDEX \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.action_head_checkpoint_path="'/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp/training_recogdrive_agent_qwen3vl_worldmirror_alignment_45epoch_LoRA_batchsize_32_post_100epoch/2026.01.02.12.33.21/lightning_logs/version_0/checkpoints/epoch=99-step=66500-EMA.ckpt'" \
    agent.vlm_path='/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/checkpoints/Qwen3-VL-2B-Instruct' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    agent.vlm_type="qwen3vl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    experiment_name='recogdrive_agent_eval/batchsize_32_post-training_epoch100' \
    agent.evaluation=True \
    agent.use_lora=True \
    metric_cache_path='/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp/exp_mlp_public/metric_cache'