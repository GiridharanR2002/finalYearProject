# eval.py
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import open_clip
import os
from tqdm import tqdm
import pandas as pd
from PIL import Image

from models import MultimodalHateClassifier  # make sure this matches your filename
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------- Dataset ----------
class MemeDataset(Dataset):
    def __init__(self, jsonl_path, image_folder, tokenizer, image_preprocess):
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.image_preprocess = image_preprocess

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        img_path = os.path.join(self.image_folder, os.path.basename(sample["img"]))
        image = self.image_preprocess(Image.open(img_path).convert('RGB'))
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        return {
            'id': sample["id"],
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'image': image,
            'label': sample.get("label", -1)  # If label not present
        }


# ---------- Inference Function ----------
def evaluate(model, dataloader):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)

            logits = model(input_ids, attention_mask, images)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            for i in range(len(preds)):
                results.append({
                    "id": batch['id'][i],
                    "pred_label": preds[i].item(),
                    "prob": probs[i][1].item(),  # probability of being hateful
                    "gold_label": batch['label'][i] if batch['label'][i] != -1 else None
                })

    return results


# ---------- Main ----------
def main():
    # Paths
    test_jsonl = "test_seen.jsonl"
    image_folder = "img"
    model_path = "best_model.pt"

    # Load tokenizer & image preprocessor
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
    _, image_preprocess_train, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    # Dataset & DataLoader
    dataset = MemeDataset(test_jsonl, image_folder, tokenizer, image_preprocess_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load Model
    model = MultimodalHateClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    results = evaluate(model, dataloader)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("test_predictions.csv", index=False)
    print("✅ Saved predictions to test_predictions.csv")

    # Optional: Metrics if labels available
    if df['gold_label'].notnull().all():
        from sklearn.metrics import classification_report, confusion_matrix
        y_true = df['gold_label'].astype(int)
        y_pred = df['pred_label'].astype(int)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
