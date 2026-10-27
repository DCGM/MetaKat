"""DINOv2 CLS vectors using the image preprocessing of the training dump."""
from pathlib import Path
import hashlib

import cv2
import numpy as np
from PIL import Image
import torch

from .dinov2_checkpoint import load_dinov2_with_registers


class DinoEncoder:
    def __init__(self, checkpoint_path, device='cpu', batch_size=32, expected_sha256=None):
        if batch_size < 1:
            raise ValueError('image batch_size must be positive')
        if expected_sha256 is not None:
            with Path(checkpoint_path).open('rb') as source:
                actual = hashlib.file_digest(source, 'sha256').hexdigest()
            if actual != expected_sha256:
                raise ValueError('DINOv2 checkpoint SHA256 does not match the configured training encoder')
        self.device, self.batch_size = torch.device(device), batch_size
        self.model = load_dinov2_with_registers(Path(checkpoint_path), 'vitb14').to(self.device).eval()
        self.dim = self.model.config.hidden_size

    @staticmethod
    def load_image(path):
        with Image.open(path) as image:
            if image.format == 'JPEG':
                image.draft('RGB', (224, 224))
            elif image.format == 'JPEG2000':
                reduce = 0
                while min(image.size) >> (reduce + 1) >= 224 and reduce < 5:
                    reduce += 1
                image.reduce = reduce
            rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
        return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

    @torch.inference_mode()
    def encode(self, paths):
        values = np.zeros((len(paths), self.dim), np.float32)
        present = np.zeros(len(paths), bool)
        indices = [i for i, p in enumerate(paths) if p is not None]
        mean = torch.tensor((.485, .456, .406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((.229, .224, .225), device=self.device).view(1, 3, 1, 1)
        for first in range(0, len(indices), self.batch_size):
            batch = indices[first:first + self.batch_size]
            images = np.stack([self.load_image(paths[i]) for i in batch])
            pixels = torch.from_numpy(images).to(self.device).permute(0, 3, 1, 2).float()
            pixels = pixels.div_(255).sub_(mean).div_(std)
            # The image dump used FP16 autocast and FP16 storage, then L2 normalization.
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
                vectors = self.model(pixel_values=pixels).last_hidden_state[:, 0]
            values[batch] = vectors.float().cpu().numpy().astype(np.float16).astype(np.float32)
            present[batch] = True
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite DINOv2 features')
        return values, present
