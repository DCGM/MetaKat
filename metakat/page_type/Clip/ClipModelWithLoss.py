from typing import Optional, Union, Tuple

import torch
from transformers import CLIPModel
from transformers.models.clip.modeling_clip import CLIPOutput


class ClipWithLoss(CLIPModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CLIPOutput]:

        return super().forward(input_ids,
                               pixel_values,
                               attention_mask,
                               position_ids,
                               return_loss,
                               output_attentions,
                               output_hidden_states,
                               return_dict)
