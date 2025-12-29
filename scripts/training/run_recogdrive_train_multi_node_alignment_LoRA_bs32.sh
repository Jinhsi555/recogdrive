export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/dataset/maps"
export NAVSIM_EXP_ROOT="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp/exp_mlp_public"
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
export HYDRA_FULL_ERROR=1  # 启用全量错误日志
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=36000

echo "GPUS: ${GPUS}"
export CUDA_LAUNCH_BLOCKING=1

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
    agent.cam_type='single' \
    agent.cache_hidden_state=False \
    agent.cache_mode=False \
    agent.freeze_backbone=True \
    agent.vlm_type="qwen3vl" \
    agent.dit_type="small" \
    agent.sampling_method="ddim" \
    agent.use_lora=True \
    trainer.params.max_epochs=100 \
    trainer.params.accumulate_grad_batches=1 \
    trainer.params.num_nodes=4 \
    dataloader.params.batch_size=1 \
    experiment_name=training_qwen3vl_backbone_agent_dit_alignment_100epoch_LoRA_batchsize_32 \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path="/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/exp/recogdrive_agent_cache_train_no_hidden_state_with_worldmirror_qwen3vl" \
    use_cache_without_dataset=True \
    force_cache_computation=False
    # > train_recogdrive_exp.txt 2>&1