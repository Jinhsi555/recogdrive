export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/dataset/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp"
export NAVSIM_DEVKIT_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive"
export OPENSCENE_DATA_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/dataset"
export PYTHONPATH="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/HunyuanWorld-Mirror:${PYTHONPATH}"
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
    --nnodes=$MLP_WORKER_NUM \
    --nproc_per_node=${GPUS} \
    --node_rank=$MLP_ROLE_INDEX \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_recogdrive_ema.py \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.vlm_path='/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/checkpoints/Qwen3-VL-2B-Instruct' \
    agent.checkpoint_path="'/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp/exp_mlp_public/training_qwen3vl_backbone_agent_dit_alignment_100epoch_LoRA_batchsize_32/checkpoints_32/lightning_logs/version_0/checkpoints/epoch=44-step=119700-EMA.ckpt'" \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=False \
    agent.freeze_backbone=False \
    agent.vlm_type="qwen3vl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    agent.use_lora=True \
    trainer.params.max_epochs=100 \
    trainer.params.num_nodes=$MLP_WORKER_NUM \
    trainer.params.devices=8 \
    dataloader.params.batch_size=4 \
    experiment_name=training_recogdrive_agent_qwen3vl_worldmirror_alignment_45epoch_LoRA_batchsize_32_post_100epoch \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp/exp_mlp_public/recogdrive_agent_cache_dir_train_qwen3vl_worldmirror_alignment_45epoch_LoRA_batchsize_32" \
    use_cache_without_dataset=True \
    force_cache_computation=False
    # > train_recogdrive_exp_ema_2b.txt &
