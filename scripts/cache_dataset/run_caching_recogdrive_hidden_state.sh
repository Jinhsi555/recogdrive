TRAIN_TEST_SPLIT=navtrain
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/data/data/wlb/ReCogDrive_github/dataset/maps"
export NAVSIM_EXP_ROOT="/mnt/data/data/wlb/ReCogDrive_github/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/data/data/wlb/ReCogDrive_github"
export OPENSCENE_DATA_ROOT="/mnt/data/data/wlb/ReCogDrive_github/dataset"
CACHE_PATH=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_qwen3vl_worldmirror_alignment_10epoch

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export PYTHONPATH="/mnt/data/data/wlb/ReCogDrive_github/HunyuanWorld-Mirror:$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}
export HYDRA_FULL_ERROR=1  # 启用全量错误日志
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=36000

echo "GPUS: ${GPUS}"

torchrun \
    --nproc_per_node=8 \
    --master_port=$MASTER_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
    agent=recogdrive_agent \
    experiment_name=recogdrive_agent_cache \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    agent.freeze_backbone=False \
    agent.vlm_type="qwen3vl" \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent.vlm_path="/mnt/data/data/wlb/Qwen3-VL-2B-Instruct" \
    agent.checkpoint_path="'/mnt/data/data/wlb/ReCogDrive_github/exp/training_qwen3vl_backbone_agent_dit_new_cache/2025.12.23.16.31.18/lightning_logs/version_0/checkpoints/epoch=9-step=106390.ckpt'" \
    cache_path=$CACHE_PATH
    # > caching_dataset.txt 2>&1