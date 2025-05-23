# main.py
import torch
from src.dataset import HateMemeDataset
from src.models import MultimodalHateClassifier
from src.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = HateMemeDataset(
    jsonl_path="data/train_caption.jsonl",
    image_dir="data/img",
    mode='train'
)

val_dataset = HateMemeDataset(
    jsonl_path="data/dev_seen.jsonl",
    image_dir="data/img",
    mode='val'
)

model = MultimodalHateClassifier()

train_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device=device,
    epochs=5,
    batch_size=16,
    lr=2e-5,
    output_dir='checkpoints'
)
