from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, Qwen3VLForConditionalGeneration, AutoProcessor, Qwen3VLProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from navsim.agents.abstract_agent import AgentInput
from qwen_vl_utils import process_vision_info

from .utils.conversation import get_conv_template

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""

class RecogDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """
    def __init__(self,
                 model_type: str,
                 cache_hidden_state: bool,
                 checkpoint_path: str,
                 device: str = "cuda"):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None  
        self.model_type = model_type.lower()
        self.cache_hidden_state = cache_hidden_state
        self.device = device

        print(f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'")

        if self.model_type == 'internvl':
            # --- Load InternVL Model and Tokenizer ---
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=self.device
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                use_fast=False
            )
            # Load model-specific configuration
            self._configure_internvl()
            self.num_image_token = 256

        elif self.model_type == 'qwen3vl':
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="flash_attention_2",
            ).eval()
            self.processor: Qwen3VLProcessor = AutoProcessor.from_pretrained(
                checkpoint_path,
            )
            
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'.")


        print(f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'.")

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")
    
    def forward(self, pixel_values: torch.Tensor, questions: List[str], num_patches_list: List[int] = None, agent_input: AgentInput = None):
        if not self.model:
            raise RuntimeError("Backbone model has not been initialized. Call initialize() on the agent first.")
            
        if self.model_type == 'internvl':
            queries = []
            for idx, num_patches in enumerate(num_patches_list):
                question = questions[idx]
                if pixel_values is not None and '<image>' not in question:
                    question = '<image>\n' + question
                
                template = get_conv_template("internvl2_5")
                template.system_message = system_message
                template.append_message(template.roles[0], question)
                template.append_message(template.roles[1], None)
                query = template.get_prompt()

                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)
                queries.append(query)
            self.tokenizer.padding_side = 'left'
            model_inputs = self.tokenizer(queries, return_tensors='pt', padding='max_length', max_length=2800)
            device = torch.device('cuda')
            input_ids = model_inputs['input_ids'].to(device)
            attention_mask = model_inputs['attention_mask'].to(device)

            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            num_patches = pixel_values.size(0)
            image_flags = torch.tensor([1] * num_patches, dtype=torch.long)


            return self.model(
                    pixel_values=pixel_values.bfloat16(),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_flags=image_flags.squeeze(-1),
                    output_hidden_states=True,
                    return_dict=True,
            )
        
        elif self.model_type == 'qwen3vl':
            if self.cache_hidden_state:
                question = questions[0]
                cameras = agent_input.cameras
                image_path = str(cameras[-1].cam_f0.image)
            else:
                question = questions[0]
                image_path = str(pixel_values[0])
            
            messages = [
                {
                    "role": "system", 
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image", 
                            "image": image_path
                        },
                        {
                            "type": "text", 
                            "text": question
                        }
                    ]
                },
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            self.processor.tokenizer.padding_side = 'left'
            
            # self.model 的 device
            device = self.model.device
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, do_resize=False, padding=True, return_tensors="pt").to(device)
        
            return self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
            ), torch.where(inputs.input_ids == 151655)[1]
    