import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from PIL import Image
from ClipModelWithClassificationHead import ClipWithClassificationHead
from metakat.tools.mods_helper import page_type_classes

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, text_folder, processor):
        self.text_folder = text_folder
        self.image_paths = [
            os.path.join(image_folder, f_name)
            for f_name in os.listdir(image_folder)
        ]
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        basename = os.path.splitext(os.path.basename(image_path))[0]
        text_path = os.path.join(self.text_folder, basename + ".txt")

        try:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"Text file {text_path} not found.")


        return {
            "image": image,
            "text": text,
            "path": image_path,
        }

class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        paths = [item["path"] for item in batch]

        processed = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return {
            "pixel_values": processed["pixel_values"],
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "path": paths,
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to CLIP checkpoint with classification head")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--text_folder", type=str, required=True, help="Path to folder with corresponding text files")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    model = ClipWithClassificationHead.from_pretrained(args.checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = ImageTextDataset(args.image_folder, args.text_folder, processor)

    collator = Collator(processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    id2label = model.config.id2label or {i: label for i, label in enumerate(page_type_classes.values())}

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

            preds = torch.argmax(outputs.logits, dim=-1)
            for path, pred in zip(batch["path"], preds):
                label = id2label.get(pred.item(), f"label_{pred.item()}")
                print(f"{os.path.basename(path)} -> {label}")


if __name__ == "__main__":
    main()