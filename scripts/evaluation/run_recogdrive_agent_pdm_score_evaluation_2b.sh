set -x

TRAIN_TEST_SPLIT=navtest

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/data/data/wlb/ReCogDrive_github/dataset/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/data/data/wlb/ReCogDrive_github/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/data/data/wlb/ReCogDrive_github"
export OPENSCENE_DATA_ROOT="/mnt/data/data/wlb/ReCogDrive_github/dataset"
export PYTHONPATH="/mnt/data/data/wlb/ReCogDrive_github/HunyuanWorld-Mirror:/mnt/data/data/wlb/ReCogDrive_github:${PYTHONPATH}"
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


CHECKPOINT="/mnt/data/data/wlb/ReCogDrive_github/exp/training_qwen3vl_backbone_agent_dit_alignment_10epoch_LoRA_batchsize_1/2025.12.26.03.37.55/lightning_logs/version_0/checkpoints/epoch=9-step=106390.ckpt"


# 1. Set NAVSIM dataset and related environment variables
# 2. Configure torchrun (e.g., single machine: --nproc_per_node=8; adjust for multi-node)
# 3. Set agent.vlm_path and agent.checkpoint_path CHECKPOINT



torchrun \
    --nproc_per_node=8 \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_recogdrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=recogdrive_agent \
    agent.checkpoint_path="'$CHECKPOINT'" \
    agent.vlm_path='/mnt/data/data/wlb/Qwen3-VL-2B-Instruct' \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    agent.vlm_type="qwen3vl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    cache_path="/mnt/data/data/wlb/ReCogDrive_github/exp/recogdrive_agent_cache_dir_train_qwen3vl_worldmirror_alignment_10epoch_LoRA" \
    experiment_name=recogdrive_agent_eval \
    agent.evaluation=True \
    agent.use_lora=True

