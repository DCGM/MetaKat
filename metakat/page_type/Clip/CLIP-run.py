import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from metakat.tools.mods_helper import page_type_classes
from transformers import CLIPProcessor
from ClipModelWithLoss import ClipWithLoss
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_folder, processor):
        self.image_paths = [
            os.path.join(image_folder, f_name)
            for f_name in os.listdir(image_folder)
        ]
        self.processor = processor

    def __len__(self):
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
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to CLIP checkpoint folder")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for")

    args = parser.parse_args()

    prompts = [f"A document page of type {label}" for label in page_type_classes.values()]

    model = ClipWithLoss.from_pretrained(args.checkpoint)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = processor.tokenizer
    model.eval()

    token_out = tokenizer(text=prompts, return_tensors='pt', padding=True)
    input_ids = token_out['input_ids']
    attention_mask = token_out['attention_mask']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model = model.to(device)

    dataset = ImageDataset(args.image_folder, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            paths = batch["path"]

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=False
            )

            logits = outputs.logits_per_image
            probs = logits.softmax(dim=-1)

            for path, prob in zip(paths, probs):
                pred_idx = prob.argmax().item()
                label = list(page_type_classes.values())[pred_idx]
                print(f"{os.path.basename(path)} -> {label}")

if __name__ == "__main__":
    main()