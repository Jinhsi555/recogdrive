export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/data/data/wlb/ReCogDrive_github/dataset/maps/nuplan-maps-v1.0"
export NAVSIM_EXP_ROOT="/mnt/data/data/wlb/ReCogDrive_github/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/data/data/wlb/ReCogDrive_github"
export OPENSCENE_DATA_ROOT="/mnt/data/data/wlb/ReCogDrive_github/dataset"
export PYTHONPATH="/mnt/data/data/wlb/ReCogDrive_github/HunyuanWorld-Mirror:/mnt/data/data/wlb/ReCogDrive_github:${PYTHONPATH}"
TRAIN_TEST_SPLIT=navtrain
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
export NCCL_TIMEOUT=3600
export HYDRA_FULL_ERROR=1  # 启用全量错误日志
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo $PYTHONPATH



torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPUS} \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_ema.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.vlm_path='/mnt/data/data/wlb/Qwen3-VL-2B-Instruct' \
    agent.checkpoint_path="'/mnt/data/data/wlb/ReCogDrive_github/exp/training_qwen3vl_backbone_agent_dit_alignment_10epoch_LoRA_batchsize_8/2025.12.26.03.37.55/lightning_logs/version_0/checkpoints/epoch=9-step=106390.ckpt'" \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=False \
    agent.freeze_backbone=False \
    agent.vlm_type="qwen3vl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    trainer.params.max_epochs=100 \
    trainer.params.num_nodes=1 \
    trainer.params.devices=8 \
    experiment_name=training_recogdrive_agent_qwen3vl_worldmirror_alignment_10epoch_LoRA_batchsize_128 \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/mnt/data/data/wlb/ReCogDrive_github/exp/recogdrive_agent_cache_dir_train_qwen3vl_worldmirror_alignment_10epoch_LoRA_batchsize_8" \
    use_cache_without_dataset=True \
    force_cache_computation=False
    # > train_recogdrive_exp_ema_2b.txt &
