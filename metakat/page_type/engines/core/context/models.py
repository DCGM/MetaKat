"""Deployment architecture: BERT, five separate feature MLPs and five task heads."""
import torch
from torch import nn

from .labels import PAGE_TYPES, PLACEMENTS


class BertBackbone(nn.Module):
    """Page embeddings enter BERT's normal position/type embedding + LayerNorm path once."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.hidden_size = model.config.hidden_size
        self.max_length = model.config.max_position_embeddings
        # Measure checkpoint content scale before removing the unused vocabulary table.
        with torch.no_grad():
            w = model.get_input_embeddings().weight.float()
            self.initial_scale = float((w - w.mean(dim=-1, keepdim=True)).square().mean().sqrt())
        # inputs_embeds bypasses this table. Replacing rather than freezing removes ~7.8M params.
        model.set_input_embeddings(nn.Embedding(1, self.hidden_size))
        model.get_input_embeddings().weight.requires_grad_(False)
        model.config.vocab_size = 1

    @classmethod
    def create(cls, pretrained=True, checkpoint='google/bert_uncased_L-4_H-256_A-4',
               revision=None, config=None, local_files_only=False):
        from transformers import BertConfig, BertModel
        if config is not None:
            if pretrained:
                raise ValueError('use checkpoint/revision for pretrained weights, config for offline/random construction')
            model = BertModel(BertConfig.from_dict(config), add_pooling_layer=False)
        elif pretrained:
            model = BertModel.from_pretrained(checkpoint, revision=revision, add_pooling_layer=False,
                                               local_files_only=local_files_only)
        else:
            cfg = BertConfig.from_pretrained(checkpoint, revision=revision, local_files_only=local_files_only)
            model = BertModel(cfg, add_pooling_layer=False)
        return cls(model)

    def forward(self, x, attention_mask):
        if x.shape[1] > self.max_length:
            raise ValueError('window exceeds backbone position limit')
        return self.model(inputs_embeds=x, attention_mask=attention_mask,
                          token_type_ids=torch.zeros_like(attention_mask, dtype=torch.long),
                          return_dict=True).last_hidden_state

    def specification(self):
        return {'kind': 'bert', 'config': self.model.config.to_dict(), 'initial_scale': self.initial_scale}


class Fusion(nn.Module):
    def __init__(self, dims, hidden_size, scale, kind='separate_mlp'):
        super().__init__()
        if kind != 'separate_mlp':
            raise ValueError('Context inference requires separate_mlp fusion')
        self.kind = kind
        self.names = ('image', 'text', 'stats', 'pagination', 'boundaries')
        self.branches = nn.ModuleDict()
        for name in self.names:
            self.branches[name] = nn.Sequential(nn.Linear(dims[name], hidden_size), nn.GELU(),
                                                nn.LayerNorm(hidden_size))
        self.projection = nn.Linear(len(self.names) * hidden_size + 3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, features):
        flags = features['presence']
        parts = []
        for i, name in enumerate(self.names):
            x = self.branches[name](features[name])
            # Mask after biased/nonlinear branch operations, including future custom feature sources.
            # Unknown boundary state is meaningful: keep this branch active.
            presence = (1 if name == 'boundaries' else features['pagination'][..., 4:5]
                        if name == 'pagination' else flags[..., i:i + 1])
            parts.append(x * presence)
        return self.scale * self.norm(self.projection(torch.cat(parts + [flags], dim=-1)))


class PageTagger(nn.Module):
    def __init__(self, backbone, dims, fusion='separate_mlp', num_types=len(PAGE_TYPES)):
        super().__init__()
        self.backbone, self.dims, self.fusion_kind, self.num_types = backbone, dims, fusion, num_types
        h = backbone.hidden_size
        self.fusion = Fusion(dims, h, backbone.initial_scale, fusion)
        self.type_head = nn.Linear(h, num_types)
        self.placement_head = nn.Linear(h, len(PLACEMENTS))
        self.start_head = nn.Sequential(nn.Linear(4 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.page_start_head = nn.Linear(h, 1)
        self.page_end_head = nn.Linear(h, 1)

    def forward(self, features, attention_mask):
        if not attention_mask.any(dim=-1).all():
            raise ValueError('empty sequence in batch')
        h = self.backbone(self.fusion(features), attention_mask)
        a, b = h[:, :-1], h[:, 1:]
        starts = self.start_head(torch.cat((a, b, (a - b).abs(), a * b), dim=-1)).squeeze(-1)
        # Position zero has no predecessor. Its zero is a placeholder, never a negative target.
        starts = torch.cat((h.new_zeros(h.shape[0], 1), starts), dim=1)
        pair_mask = attention_mask.clone()
        pair_mask[:, 0] = False
        pair_mask[:, 1:] &= attention_mask[:, :-1]
        output = {'type_logits': self.type_head(h), 'placement_logits': self.placement_head(h),
                  'starts_document_logits': starts, 'pair_mask': pair_mask}
        output.update(page_start_logits=self.page_start_head(h).squeeze(-1),
                      page_end_logits=self.page_end_head(h).squeeze(-1))
        return output

    def specification(self):
        spec = {'backbone': self.backbone.specification(), 'dims': self.dims,
                'fusion': self.fusion_kind, 'num_types': self.num_types}
        spec['auxiliary_edges'] = True
        return spec


def from_specification(spec):
    if (spec.get('auxiliary_edges') is not True or spec.get('backbone', {}).get('kind') != 'bert'
            or spec.get('fusion') != 'separate_mlp' or spec.get('dims', {}).get('pagination') != 5
            or spec.get('scan_borders') or spec.get('dims', {}).get('boundaries') != 5
            or set(spec.get('dims', {})) != {'image', 'text', 'stats', 'pagination', 'boundaries'}):
        raise ValueError('Context inference requires five-bit boundary and pagination-aware BERT with separate MLPs and auxiliary start/end heads')
    b = spec['backbone']
    backbone = BertBackbone.create(pretrained=False, config=b['config'])
    if 'initial_scale' in b:
        backbone.initial_scale = b['initial_scale']
    return PageTagger(backbone, spec['dims'], spec['fusion'], spec['num_types'])
