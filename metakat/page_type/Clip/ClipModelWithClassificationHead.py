from dataclasses import dataclass
import torch
import os
from torch import nn
from transformers import CLIPModel, CLIPConfig
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional, Union, Tuple, Any

@dataclass
class CLIPWithClassificationHeadOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`)
            Logits of combined image and text features
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )

class ClipWithClassificationHead(CLIPModel):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels

        self.classificationHead = None
        self.init_classification_head()

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        """
        # Combines one image with more text inputs for Classification
        
        batch_size = text_embeds.size(0)
        image_embeds = image_embeds.unsqueeze(0).expand(batch_size, -1, -1).squeeze(1)
        print(f"EMBDEDS {image_embeds.size()}")
        """

        combined_features = torch.cat((image_embeds, text_embeds), dim=-1)

        logits = self.classificationHead(combined_features)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPWithClassificationHeadOutput(
            loss=loss,
            logits=logits,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def init_classification_head(self):
        text_dim = self.config.text_config.projection_dim
        vision_dim = self.config.vision_config.projection_dim

        num_labels = self.num_labels

        self.classificationHead = nn.Sequential(
            nn.Linear(text_dim + vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.init_classification_head()

        return model