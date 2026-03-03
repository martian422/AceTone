# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from transformers import AutoModel, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
from model.config_acetone import AceToneConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLCausalLMOutputWithPast
)
# from typing import Optional, Union
# from transformers.cache_utils import Cache


class AceToneModel(Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel):
    model_type = "acetone"
    config_class = AceToneConfig
    def __init__(self, config: AceToneConfig):
        # just call Qwen2_5_VLModel init
        super().__init__(config)
    

class AceToneVLM(Qwen2_5_VLForConditionalGeneration):
    model_type = "acetone"
    config: AceToneConfig

    def __init__(self, config: AceToneConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = AceToneModel(config)

        # We donot need additional head, just use the original lm_head. Qwen has preserved positions.
        assert hasattr(config, "mm_vocab_size"), "Config must define mm_vocab_size"
        # <MM0>:151667,..., <MM255>:151922
        self.vq_range = (151667, 151667 + config.mm_vocab_size -1)

    # @property
    # def device(self):
    #     return next(iter(self.parameters())).device

    # @can_return_tuple
    # @auto_docstring
    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     pixel_values_videos: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     rope_deltas: Optional[torch.LongTensor] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     second_per_grid_ts: Optional[torch.Tensor] = None,
    #     logits_to_keep: Union[int, torch.Tensor] = 0,
    #     **kwargs
    # ):
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )

    #     outputs = self.model(
    #         input_ids=input_ids,
    #         pixel_values=pixel_values,
    #         pixel_values_videos=pixel_values_videos,
    #         image_grid_thw=image_grid_thw,
    #         video_grid_thw=video_grid_thw,
    #         second_per_grid_ts=second_per_grid_ts,
    #         position_ids=position_ids,
    #         attention_mask=attention_mask,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=True,
    #         cache_position=cache_position,
    #         **kwargs,
    #     )
        
    #     hidden_states = outputs[0]
    #     hidden_states = hidden_states[:, -logits_to_keep:, :]

    #     # Combine logits from both heads
    #     logits = hidden_states.new_full(
    #         (*hidden_states.shape[:-1], self.vocab_size),
    #         torch.finfo(hidden_states.dtype).min,
    #     )
    #     logits[:, :, :self.vocab_size] = self.lm_head(hidden_states)
    #     logits = logits.float()

    #     loss = None
    #     if labels is not None:
    #         loss = self.loss_function(
    #             logits=logits, labels=labels, vocab_size=self.config.vocab_size + self.config.mm_vocab_size, **kwargs
    #         )

    #     return Qwen2_5_VLCausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )


# Register model classes so they can be loaded from config
AutoModel.register(AceToneModel.config_class, AceToneModel)
AutoModelForCausalLM.register(AceToneVLM.config_class, AceToneVLM)
