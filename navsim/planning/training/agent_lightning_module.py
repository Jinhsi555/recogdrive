import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple,Any

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        #prediction = self.agent.forward(features,targets)
        action_loss, alignment_loss = self.agent.compute_loss(features, targets, prediction)
        loss = action_loss + 0.5 * alignment_loss
        self.log(f"{logging_prefix}/action_loss", action_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/alignment_loss", alignment_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        print(f"[{logging_prefix}] action_loss: {action_loss.item():.4f}, "
            f"alignment_loss: {alignment_loss.item():.4f}, "
            f"total_loss: {loss.item():.4f}")
        return loss
    
    # def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     """
    #     每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
    #     """
    #     filtered_sd = {
    #         k: v
    #         for k, v in checkpoint['state_dict'].items()
    #         if not k.startswith('agent.model')
    #     }
    #     checkpoint['state_dict'] = filtered_sd
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        智能保存检查点：
        1. 始终保存 action_head 全部参数
        2. 保存 backbone 中需要梯度的参数（解冻层）
        3. 过滤掉 backbone 中冻结的预训练参数以节省空间
        4. 保留其他必要参数（如优化器状态等）
        """
        
        print("💾 正在保存检查点...")
        state_dict = checkpoint['state_dict']
        
        filtered_sd = {
            k: v
            for k, v in state_dict.items()
            if 'lora' in k or 'action_head' in k
        }
        checkpoint['state_dict'] = filtered_sd
        
        # # 统计信息
        # original_size = sum(t.numel() for t in state_dict.values())
        # saved_size = sum(t.numel() for t in filtered_sd.values())
        # saved_params = len(filtered_sd)
        # original_params = len(state_dict)
        
        # print(f"📊 检查点压缩: {saved_params}/{original_params} 个参数")
        # print(f"📦 空间节省: {saved_size}/{original_size:,} 元素 ({saved_size/original_size:.1%})")

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
    
    def on_train_start(self):
        # 所有参数及其 requires_grad 状态
        for name, param in self.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
        
        optimizer = self.optimizers()
        param_groups = optimizer.param_groups
        print(f"优化器参数组数量: {len(param_groups)}")
        for i, group in enumerate(param_groups):
            print(f"组 {i}: {len(group['params'])} 个参数")

class AgentLightningDiT(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens_list = batch
        prediction = self.agent.forward(features,targets,tokens_list)
        if logging_prefix == 'train':
            predictions = self.agent.compute_loss(features, targets, prediction)

            loss = predictions.loss
            reward = predictions.reward
            policy_loss = predictions.policy_loss
            bc_loss = predictions.bc_loss
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/reward", reward, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/policy_loss", policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/bc_loss", bc_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            prediction = self.agent.forward(features,targets)
            loss = self.agent.compute_loss(features, targets, prediction)
            self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        每次保存 checkpoint 时，只保留 state_dict 中不以 'agent.model' 开头的条目。
        """
        filtered_sd = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if not k.startswith('agent.model')
        }
        checkpoint['state_dict'] = filtered_sd

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        #print(batch_idx)
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()