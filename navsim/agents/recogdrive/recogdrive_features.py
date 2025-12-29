from typing import Dict, Optional
import torch
import numpy as np
import gzip
import pickle
from PIL import Image
from pathlib import Path

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .recogdrive_backbone import RecogDriveBackbone
from .utils.internvl_preprocess import load_image

from src.utils.inference_utils import prepare_images_to_tensor
from src.models.models.worldmirror import WorldMirror

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


class ReCogDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_hidden_state: bool = True,
                 model_type: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 cache_mode: bool = False, ):
        """
        Initializes the feature builder.

        Args:
            cache_hidden_state (bool): If True, operates in online mode, initializes the backbone,
                                       and computes the hidden state. If False, operates in offline
                                       mode, does not initialize the backbone, and returns
                                       pre-computable tensors, including a tensorized representation
                                       of the image file path.
            model_type (str, optional): The type of model to load ('internvl' or 'qwen'). Required if cache_hidden_state is True.
            checkpoint_path (str, optional): Path to the model checkpoint. Required if cache_hidden_state is True.
            device (str): The device to load the model onto.
        """
        super().__init__()
        self.cache_hidden_state = cache_hidden_state
        self.backbone = None
        self.cache_mode = cache_mode
        self.model_type = model_type
        self.device = device

        if self.cache_hidden_state and self.cache_mode:
            if not model_type or not checkpoint_path:
                raise ValueError("In online mode (cache_hidden_state=True), `model_type` and `checkpoint_path` must be provided.")
            self.backbone = RecogDriveBackbone(
                model_type=model_type,
                cache_hidden_state=cache_hidden_state,
                checkpoint_path=checkpoint_path,
                device=device
            )
        
        if not self.cache_hidden_state and self.cache_mode:
            self.geometry_backbone = WorldMirror.from_pretrained("/vepfs-mlp2/c20250502/haoce/wlb/recogdrive/checkpoints/HunyuanWorld-Mirror").to(device).eval()
            if self.geometry_backbone:
                print("Geometry Backbone Loaded")

    def get_unique_name(self) -> str:
        return f"{self.model_type}_features"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:

        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras

        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32
        )
        high_command_one_hot = torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32)
        status_feature = torch.cat([
            high_command_one_hot.clone(),
            torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ], dim=-1)


        if not self.cache_hidden_state:
            image_path = str(cameras[-1].cam_f0.image)
            
            path_as_ordinals = [ord(char) for char in image_path]
            
            path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)
            
            if not self.cache_hidden_state and self.cache_mode: 
                # 3D Geometry model feature cache
                views = {}
                imgs = prepare_images_to_tensor([str(cameras[-1].cam_f0.image)]).to(self.device)  # [1,S,3,H,W], in [0,1]
                # 内参
                intrinsics_list = [torch.tensor(cameras[-1].cam_f0.intrinsics, dtype=torch.float32, device=self.device)]
                # 外参
                c2w = np.eye(4)
                c2w[:3, :3] = cameras[-1].cam_f0.sensor2lidar_rotation
                c2w[:3, 3] = cameras[-1].cam_f0.sensor2lidar_translation
                # w2c = np.linalg.inv(c2w)
                extrinsics_list = [torch.tensor(c2w, dtype=torch.float32, device=self.device)]
                
                views["img"] = imgs
                views["camera_poses"] = torch.stack(extrinsics_list, dim=0).unsqueeze(0)
                views["camera_intrs"] = torch.stack(intrinsics_list, dim=0).unsqueeze(0)
                
                cond_flags = [1, 0, 1]  # [camera_pose, depth, intrinsics]
                
                priors = self.geometry_backbone.extract_priors(views)
                geometry_features_list, patch_start_idx = self.geometry_backbone.visual_geometry_transformer(views["img"], priors, cond_flags=cond_flags)  # list: [4 * hidden_state], patch_start_idx = 7 (camera_token, register_token*4, pose_token, ray_token)
                last_geometry_feature = geometry_features_list[-1][:, :, patch_start_idx:]
                last_geometry_feature = last_geometry_feature.view(-1, 21, 37, last_geometry_feature.shape[-1])
            
                return {
                    "geometry_features": last_geometry_feature.cpu(),
                    "history_trajectory": history_trajectory.cpu(),
                    "high_command_one_hot": high_command_one_hot.cpu(),
                    "status_feature": status_feature.cpu(),
                    "image_path_tensor": path_tensor.cpu(),
                }
            else:
                return {
                    "history_trajectory": history_trajectory.cpu(),
                    "high_command_one_hot": high_command_one_hot.cpu(),
                    "status_feature": status_feature.cpu(),
                    "image_path_tensor": path_tensor.cpu(),
                }
        else:
            if self.backbone is None:
                raise RuntimeError("FeatureBuilder is in online mode, but the backbone was not initialized.")
            
            if self.backbone.model_type == 'internvl':
                pixel_values = load_image(str(cameras[-1].cam_f0.image)).unsqueeze(0)
                pixel_values_squeezed = pixel_values.squeeze(1)
                num_patches_list = [pv.shape[0] for pv in pixel_values_squeezed]
                pixel_values_cat = torch.cat(list(pixel_values_squeezed), dim=0)

                navigation_commands = ['turn left', 'go straight', 'turn right']
                command_str = next((navigation_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1), "unknown")
                history_str = " ".join([f'   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, {format_number(history_trajectory[i, 1].item())}, {format_number(history_trajectory[i, 2].item())})' for i in range(4)])
                
                prompt = f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{history_str}\n3. Active navigation command: [{command_str.upper()}]"
                output_requirements = "\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
                questions = [f"{prompt}{output_requirements}"]

                outputs = self.backbone(pixel_values_cat.cuda(), questions, num_patches_list=num_patches_list, agent_input=agent_input)
                last_hidden_state = outputs.hidden_states[-1]
                
                return {
                    "history_trajectory": history_trajectory.cpu(),
                    "high_command_one_hot": high_command_one_hot.cpu(),
                    "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
                    "status_feature": status_feature.cpu(),
                }
                
            else:
                pixel_values = None
                navigation_commands = ['turn left', 'go straight', 'turn right']
                command_str = next((navigation_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1), "unknown")
                history_str = " ".join([f'   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, {format_number(history_trajectory[i, 1].item())}, {format_number(history_trajectory[i, 2].item())})' for i in range(4)])
                
                prompt = f"As an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{history_str}\n3. Active navigation command: [{command_str.upper()}]"
                output_requirements = "\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
                questions = [f"{prompt}{output_requirements}"]
                
                outputs, visual_feature_idx = self.backbone(pixel_values, questions, num_patches_list=None, agent_input=agent_input)
                last_hidden_state = outputs.hidden_states[-1]

                return {
                    "history_trajectory": history_trajectory.cpu(),
                    "high_command_one_hot": high_command_one_hot.cpu(),
                    "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
                    "status_feature": status_feature.cpu(),
                }


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}
