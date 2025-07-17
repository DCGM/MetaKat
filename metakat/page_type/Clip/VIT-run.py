import os
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, image_path, processor):
        self.image_paths = [
            os.path.join(image_path, f_name)
            for f_name in os.listdir(image_path)
        ]
        self.processor = processor


    def  __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "path": path
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ViT checkpoint folder")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for")

    args = parser.parse_args()

    model = ViTForImageClassification.from_pretrained(args.checkpoint)
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model.eval()

    dataset = ImageDataset(args.image_folder, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"]
            outputs = model(pixel_values)
            preds = torch.argmax(outputs.logits, dim=-1)

            for path, pred in zip(batch["path"], preds):
                label = model.config.id2label[pred.item()]
                results.append((path, label))

    for path, label in results:
        print(f"{os.path.basename(path)} -> {label}")

if __name__ == "__main__":
    main()