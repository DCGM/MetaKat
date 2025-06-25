# Trainer, predicter and evaluater for LayoutLMv3 fine-tuning with just geometric position of tokens
# author: Marie Parilova
# date: 13.5.2025

import os
import argparse
import json
import torch
import numpy as np
from PIL import Image
from torch.nn import BCEWithLogitsLoss
from transformers import AutoProcessor, AutoConfig, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
from datasets import load_dataset, Features, Sequence, Value, Array2D, Array3D
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Define the label names and create mappings between IDs and labels
names = [
    "O",
    "B-TITULEK", "I-TITULEK", "B-PODTITULEK", "I-PODTITULEK", "B-MISTO_VYDANI", "I-MISTO_VYDANI",
    "B-AUTOR", "I-AUTOR", "B-DATUM_VYDANI", "I-DATUM_VYDANI", "B-NAKLADATEL", "I-NAKLADATEL",
    "B-SERIE", "I-SERIE", "B-CISLO_SERIE", "I-CISLO_SERIE", "B-TISKAR", "I-TISKAR",
    "B-MISTO_TISKU", "I-MISTO_TISKU", "B-VYDANI", "I-VYDANI", "B-PREKLADATEL", "I-PREKLADATEL",
    "B-DIL", "I-DIL", "B-NAZEV_DILU", "I-NAZEV_DILU", "B-EDITOR", "I-EDITOR", "B-ILUSTRATOR", "I-ILUSTRATOR"
]
id2label = {i: names[i] for i in range(len(names))}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(names)  # Total number of distinct labels in the dataset

# Initialize the processor for LayoutLMv3 without OCR (assume pre-extracted tokens)
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
# For custom training without image and text context
white_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)

def convert_to_multihot(labels_batch, num_labels):
    """
    Convert a batch of single-label tag sequences to multi-hot vectors.
    For each token, create a one-hot (or multi-hot if label is list) vector of length num_labels.
    Tokens with label -100 (ignored tokens) get a vector of all -100s.
    """
    result = []
    for sequence in labels_batch:
        seq_multihot = []
        for lbl in sequence:
            if lbl == -100:
                # Preserve the ignore index for padding or truncated tokens
                seq_multihot.append([-100] * num_labels)
            else:
                multihot = [0] * num_labels
                # Allow for multiple labels per token if provided as list
                ids = lbl if isinstance(lbl, (list, tuple)) else [lbl]
                for i in ids:
                    if 0 <= i < num_labels:
                        multihot[i] = 1
                seq_multihot.append(multihot)
        result.append(seq_multihot)
    return result

class MultiLabelNERTrainer(Trainer):
    """
    Custom Trainer that implements multi-label token-level classification using BCEWithLogitsLoss.
    Allows for class weights to handle label imbalance.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            cw = class_weights.float().to(self.args.device)
            print("Using multi-label classification with class weights", cw)
        self.loss_fct = BCEWithLogitsLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract the multi-hot labels tensor (shape: batch x seq_len x num_labels)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # Raw logits from the model

        # Match sequence lengths between logits and labels
        min_len = min(logits.size(1), labels.size(1))
        logits = logits[:, :min_len, :]
        labels = labels[:, :min_len, :]

        # Create mask for valid label entries (0 or 1) to ignore padding (-100 entries)
        valid_mask = ((labels == 0) | (labels == 1)).all(dim=-1)
        flat_outputs = logits[valid_mask]
        flat_labels = labels[valid_mask].float()

        # Compute multi-label BCE loss
        loss = self.loss_fct(flat_outputs, flat_labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    """
    Compute precision, recall, and F1-score for evaluation.
    Applies a 0.5 threshold on sigmoid probabilities for multi-label predictions.
    Ignores padding tokens in computation.
    """
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int()
    labels = torch.tensor(labels).int()

    # Align sequence lengths and mask out invalid tokens
    min_len = min(preds.shape[1], labels.shape[1])
    preds, labels = preds[:, :min_len, :], labels[:, :min_len, :]
    valid_mask = ((labels == 0) | (labels == 1)).all(dim=-1)
    preds_f = preds[valid_mask]
    labels_f = labels[valid_mask]

    # Logging basic statistics
    print("Sigmoid stats → min:", probs.min().item(), "max:", probs.max().item(), "mean:", probs.mean().item())
    print("Label nonzero count:", labels_f.nonzero().size(0))
    print("Nonzero preds:", preds_f.sum().item(), "/", preds_f.numel())

    # Return macro-averaged metrics across all labels
    return {
        'precision': precision_score(labels_f.cpu(), preds_f.cpu(), average='macro', zero_division=0),
        'recall': recall_score(labels_f.cpu(), preds_f.cpu(), average='macro', zero_division=0),
        'f1': f1_score(labels_f.cpu(), preds_f.cpu(), average='macro', zero_division=0)
    }

def find_best_threshold(logits, labels):
    """
    Search for the optimal probability threshold (between 0.05 and 0.94) that maximizes macro F1.
    """
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thresh = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(labels.reshape(-1, labels.shape[-1]), preds.reshape(-1, preds.shape[-1]), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"Best threshold: {best_thresh}, F1: {best_f1:.4f}")
    return best_thresh

def prepare_dataset(cache_dir, split):
    """
    Load and preprocess the dataset for a given split ('train' or 'test').
    - Loads the custom HuggingFace dataset script.
    - Tokenizes images and tokens with LayoutLMv3 processor.
    - Converts labels to multi-hot vectors and pads/truncates to fixed length.
    """
    dataset = load_dataset(
        "./layoutlmv3.py", trust_remote_code=True,
        cache_dir=cache_dir, name="Biblio"
    )

    # Define features schema to match model inputs
    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3,224,224)),
        'input_ids': Sequence(Value(dtype="int64")),
        'attention_mask': Sequence(Value(dtype="int64")),
        'bbox': Array2D(dtype="int64", shape=(512,4)),
        'labels': Array2D(dtype="int64", shape=(512,num_labels)),
    })

    def prepare(ex):
        images = [white_image] * len(ex['bboxes'])
        words = [["x"] * len(b) for b in ex['tokens']] # without text context
        enc = processor(
            images, words, boxes=ex['bboxes'],
            word_labels=ex['ner_tags'], truncation=True, padding="max_length"
        )
        # Convert per-token labels to multi-hot vectors
        multihot = convert_to_multihot(enc.pop('labels'), num_labels)
        enc['labels'] = np.array(multihot, dtype=np.int64)
        return enc

    ds = dataset[split].map(
        prepare, batched=True,
        remove_columns=dataset[split].column_names,
        features=features
    )
    return ds, num_labels

def train(args):
    """
    Training pipeline:
    - Prepare datasets
    - Initialize model and config
    - Compute class weights for handling imbalance
    - Set up Trainer with custom loss and metrics
    - Start training and evaluate at each epoch
    """
    train_ds, num_labels = prepare_dataset(args.cache_dir, 'train')
    eval_ds, _ = prepare_dataset(args.cache_dir, 'test')

    config = AutoConfig.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", config=config
    )

    # Compute class weights
    #counts = torch.tensor(args.counts, dtype=torch.float)
    counts = torch.tensor([
        34444,1519,9334,545,3434,1340,356,1234,1539,1177,162,
        1380,3001,281,695,218,3,500,838,262,150,
        192,22,97,107,116,1,75,231,101,154,86,109
    ], dtype=torch.float)
    total = counts.sum()
    pos_weight = torch.log1p((total - counts) / counts)
    pos_weight[0] = 1.0
    pos_weight = torch.clamp(pos_weight, max=5.0)
    torch.save(pos_weight, args.pos_weight_path)
    #pos_weight = torch.load('pos_weight.pt')

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )

    trainer = MultiLabelNERTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        class_weights=pos_weight
    )
    trainer.train()
    print("✅ Training complete.")


def evaluate(args):
    """
    Evaluate the saved model on the test set and print classification report.
    Uses a fixed threshold to binarize probabilities.
    """
    _, num_labels = prepare_dataset(args.cache_dir, 'train')
    eval_ds, _ = prepare_dataset(args.cache_dir, 'test')

    # Load model
    config = AutoConfig.from_pretrained(
        args.model_dir, num_labels=num_labels
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_dir, config=config
    )
    processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    training_args = TrainingArguments(output_dir=args.output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=processor,
        data_collator=default_data_collator
    )

    output = trainer.predict(eval_ds)
    logits, labels = output.predictions, output.label_ids
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > args.threshold).int().numpy()

    mask = labels.sum(-1) > 0
    preds, labels = preds[mask], labels[mask]
    report = classification_report(
        labels.reshape(-1, num_labels),
        preds.reshape(-1, num_labels),
        zero_division=0
    )
    print(report)
    print("✅ Evaluation complete.")

def predict(args):
    """
    Run inference on new data, find best threshold, and save predictions per file as JSON.
    Each JSON contains token indices, bboxes, true & predicted labels, and top class scores.
    """
    # Load model and trainer
    config = AutoConfig.from_pretrained(
        args.model_dir, num_labels=num_labels
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_dir, config=config
    )
    training_args = TrainingArguments(output_dir=args.output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=processor,
        data_collator=default_data_collator
    )

    # Define features
    features_proc = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3,224,224)),
        'input_ids': Sequence(Value(dtype="int64")),
        'attention_mask': Sequence(Value(dtype="int64")),
        'bbox': Array2D(dtype="int64", shape=(512,4)),
        'labels': Array2D(dtype="int64", shape=(512,num_labels)),
        'image_path': Value("string"),
        'tokens': Sequence(Value("string")),
        'word_ids': Sequence(Value("int64")),
        'input_tokens': Sequence(Value("string")), 
    })

    # Prepare examples similar to training pipeline
    def prepare_examples(examples):
        images = [white_image] * len(examples['bboxes'])
        words = [["x"] * len(b) for b in examples['tokens']] # without text context
        enc = processor(
            images, words,
            boxes=examples["bboxes"],
            word_labels=examples["ner_tags"],
            truncation=True,
            padding="max_length"
        )
        token_labels = enc.pop("labels")
        multihot = convert_to_multihot(token_labels, num_labels)
        enc["labels"] = np.array(multihot, dtype=np.int64)
        enc["image_path"] = examples['image_path']
        enc["tokens"] = examples['tokens']
        enc["word_ids"] = [
            [-1 if wid is None else wid for wid in enc.encodings[i].word_ids]
            for i in range(len(enc.encodings))
        ]
        return enc
    
    dataset = load_dataset("./layoutlmv3.py", trust_remote_code=True, cache_dir=args.cache_dir, name="Biblio")

    eval_ds = dataset['test'].map(
        prepare_examples,
        batched=True,
        remove_columns = [c for c in dataset["test"].column_names if c not in ['file_name']],
        features=features_proc
    )

    # Get predictions and compute optimal threshold
    output = trainer.predict(eval_ds)
    logits = output.predictions
    logits = logits[:, :512, :]
    labels = output.label_ids
    mask = labels.sum(-1) > 0
    logits = logits[mask]
    labels = labels[mask]
    best_t = find_best_threshold(logits, labels)

    # Apply threshold and save results to JSON files
    output_dir = "prediction_outputs"
    os.makedirs(output_dir, exist_ok=True)
    logits = output.predictions[:, :512, :]  # tvar (N, 512, C)
    labels = output.label_ids                # tvar (N, 512, C)

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds_multihot = (probs > best_t).astype(int)

    # Iterate over samples and write JSON per image
    N, L, C = labels.shape

    for i in range(N):
        filename = eval_ds[i].get("image_path", f"sample_{i+1}")
        filename_no_ext = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(output_dir, f"{filename_no_ext}.json")
        sample_output = {
            "file_name": filename,
            "tokens": []
        }
        word_ids = eval_ds[i]["word_ids"]
        tokens = eval_ds[i]["tokens"]
        prev_wid = -1
        for j in range(L):
            if mask[i, j] == 0:
                continue

            wid = word_ids[j]
            if wid is None:
                continue                                 # skip subword/padding
            if wid == -1 or wid == prev_wid:
                continue
            prev_wid = wid
            token = tokens[wid]

            true_vec = labels[i, j]
            pred_vec = preds_multihot[i, j]
            prob_vec = probs[i, j]

            true_indices = np.where(true_vec == 1)[0].tolist()
            pred_indices = np.where(pred_vec == 1)[0].tolist()

            true_classes = [id2label[idx] for idx in true_indices]
            pred_classes = [
                {"label": id2label[idx], "score": float(prob_vec[idx])}
                for idx in pred_indices
            ]

            if true_classes == ["O"] and all(p["label"] == "O" for p in pred_classes):
                continue

            top_n = 3
            top_indices = prob_vec.argsort()[-top_n:][::-1]
            top_classes = [
                {"label": id2label[idx], "score": float(prob_vec[idx])}
                for idx in top_indices
            ]

            try:
                box = eval_ds[i]["bbox"][wid]
            except:
                box = ["?", "?", "?", "?"]


            sample_output["tokens"].append({
                "token_idx": j,
                "token": token,
                "bbox": box,
                "true_labels": true_classes,
                "predicted_labels": pred_classes,
                "top_predictions": top_classes
            })


        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_output, f, ensure_ascii=False, indent=2)

    print(f"✅ Predictions saved in dir: {output_dir}")

def main():
    """
    Parse command-line arguments and dispatch to train, evaluate, or predict functions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default="/storage/brno2/home/parilovam/.cache/huggingface/datasets/")
    parser.add_argument('--output_dir', type=str, default='./out')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    #parser.add_argument('--counts', nargs='+', type=float, required=True,
    #                    help='List of counts for each label for weight computation')
    parser.add_argument('--pos_weight_path', type=str, default='pos_weight.pt')
    parser.add_argument('--model_dir', type=str, default='./out')
    parser.add_argument('--threshold', type=float, default=0.5)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')
    group.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    elif args.predict:
        predict(args)

if __name__ == '__main__':
    main()
