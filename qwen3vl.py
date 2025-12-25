import torch
from torch import nn
from typing import Optional, Tuple, Union, Dict, Any
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast
from dataclasses import dataclass

@dataclass
class CustomQwen3VLOutput(Qwen3VLCausalLMOutputWithPast):
    """自定义输出类，可以添加额外属性"""
    custom_attribute = "custom_value"

class CustomQwen3VL(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CustomQwen3VLOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        hidden_states = outputs[0]
        
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
        
        if not return_dict:
            return (loss, logits) + outputs[1:]
        
        # 返回自定义输出
        return CustomQwen3VLOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            rope_deltas=outputs.rope_deltas if hasattr(outputs, 'rope_deltas') else None,
        )

# 使用示例
def main():
    # 1. 使用自定义类加载模型
    config = AutoConfig.from_pretrained("/mnt/data/data/wlb/Qwen3-VL-2B-Instruct")
    model = CustomQwen3VL.from_pretrained(
        "/mnt/data/data/wlb/Qwen3-VL-2B-Instruct",
        config=config,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    processor = AutoProcessor.from_pretrained("/mnt/data/data/wlb/Qwen3-VL-2B-Instruct")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs = None 
    video_inputs = None
    
    processor.tokenizer.padding_side = 'left'
    inputs = processor(
        text=[text], 
        images=image_inputs, 
        videos=video_inputs, 
        do_resize=False, 
        padding=True, 
        return_tensors="pt"
    )
    
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    print(outputs)
    print(f"Custom attribute: {outputs.custom_attribute}")

if __name__ == "__main__":
    main()
