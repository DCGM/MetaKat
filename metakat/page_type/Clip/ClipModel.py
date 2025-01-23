import torch
from torch import nn
from transformers import CLIPModel, CLIPConfig
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional, Union, Tuple

"""
class CLIPOutput(ModelOutput):
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
"""
class ClipWithClassificationHead(CLIPModel):
    def __init__(self, config: CLIPConfig, num_classes: int):
        super().__init__(config)

        text_config = config.text_config
        vision_config = config.vision_config

        # text_dim = text_config.hidden_size
        # vision_dim = vision_config.hidden_size

        text_dim = text_config.projection_dim
        vision_dim = vision_config.projection_dim

        print(f"TEXT DIM: {text_dim} \nVISION DIM: {vision_dim}")

        #TODO add changeable parameters (from config?)
        self.classificationHead = nn.Sequential(
            nn.Linear(text_dim + vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

        #TODO weight init

        # print(nn.Linear.weight)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            return_loss: Optional[bool] = None,
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

        batch_size = text_embeds.size(0)
        image_embeds = image_embeds.unsqueeze(0).expand(batch_size, -1, -1).squeeze(1)
        print(f"EMBDEDS {image_embeds.size()}")

        combined_features = torch.cat((image_embeds, text_embeds), dim=-1)

        print(combined_features.size())

        logits = self.classificationHead(combined_features)

        print("LOGITS")
        print(logits.size())
        print("LABELS")
        print(labels.size())

        loss = None
        if return_loss:
            if input_ids is not None:
                #TODO change loss function based on config?
                loss_fcn = nn.CrossEntropyLoss()
                loss = loss_fcn(logits, labels)
        """
        if not return_dict:
            return (loss, logits) if loss is not None else logits

        
        return BaseModelOutputWithPooling(
            loss=loss,
            logits=logits,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_outputs=text_outputs,
            vision_outputs=vision_outputs,
        )
        """
        return (loss, logits) if loss is not None else logits

