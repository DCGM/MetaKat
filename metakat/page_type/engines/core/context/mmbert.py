"""mmBERT masked-mean page text vectors, matching the stored training vectors."""
from itertools import islice

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def patch_padding_nan(model):
    """Keep fully masked ModernBERT padding queries finite (Transformers 4.57)."""
    original = model._update_attention_mask

    def update(attention_mask, output_attentions):
        global_mask, local_mask = original(attention_mask, output_attentions=output_attentions)
        diagonal = torch.eye(global_mask.shape[-1], dtype=torch.bool, device=global_mask.device)
        return global_mask.masked_fill(diagonal, 0.), local_mask.masked_fill(diagonal, 0.)

    model._update_attention_mask = update


class MmbertEncoder:
    def __init__(self, model_dir, device='cpu', batch_size=8, max_tokens_per_batch=16384,
                 max_length=8192, max_chars=4000):
        if min(batch_size, max_tokens_per_batch, max_length, max_chars) < 1 or max_length < 2:
            raise ValueError('Invalid text encoding limits')
        if max_tokens_per_batch < max_length:
            raise ValueError('text token budget must accommodate max_length')
        self.device, self.batch_size = torch.device(device), batch_size
        self.max_tokens_per_batch, self.max_length, self.max_chars = max_tokens_per_batch, max_length, max_chars
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True,
                                               attn_implementation='sdpa')
        patch_padding_nan(self.model)
        self.model.to(self.device).eval()
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts, *, count=None):
        """Encode a sequence or a counted stream without retaining all OCR text."""
        count = len(texts) if count is None else count
        if count < 0:
            raise ValueError('text count must be nonnegative')
        values = np.zeros((count, self.dim), np.float32)
        present = np.zeros(count, bool)
        stream = iter(texts)
        # Tokenize in bounded groups; token batching below never changes output order.
        for first in range(0, count, self.batch_size):
            size = min(self.batch_size, count - first)
            chunk = list(islice(stream, size))
            if len(chunk) != size:
                raise ValueError('Text stream shorter than declared count')
            present[first:first + size] = [t is not None for t in chunk]
            indices = [first + i for i, text in enumerate(chunk) if text is not None]
            if not indices:
                continue
            inputs = [chunk[i - first][:self.max_chars] if chunk[i - first].strip() else '' for i in indices]
            ids = self.tokenizer(inputs, add_special_tokens=True, truncation=False,
                                 return_attention_mask=False)['input_ids']
            ids = [x if len(x) <= self.max_length else x[:self.max_length - 1] + x[-1:] for x in ids]
            order = sorted(range(len(ids)), key=lambda i: len(ids[i]), reverse=True)
            while order:
                length = len(ids[order[0]])
                batch_count = min(len(order), self.batch_size, max(1, self.max_tokens_per_batch // length))
                batch, order = order[:batch_count], order[batch_count:]
                tokens = torch.full((batch_count, length), self.tokenizer.pad_token_id,
                                    dtype=torch.long, device=self.device)
                mask = torch.zeros_like(tokens)
                for row, i in enumerate(batch):
                    tokens[row, :len(ids[i])] = torch.tensor(ids[i], device=self.device)
                    mask[row, :len(ids[i])] = 1
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                    hidden = self.model(input_ids=tokens, attention_mask=mask).last_hidden_state
                # Mask with where rather than multiplication, including any padding NaNs.
                hidden = hidden.float().masked_fill(~mask.bool().unsqueeze(-1), 0)
                vectors = hidden.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
                values[[indices[i] for i in batch]] = vectors.cpu().numpy().astype(np.float16).astype(np.float32)
        sentinel = object()
        if next(stream, sentinel) is not sentinel:
            raise ValueError('Text stream longer than declared count')
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite mmBERT features')
        return values, present
