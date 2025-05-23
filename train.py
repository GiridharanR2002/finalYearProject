# train.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os

def train_model(model, train_dataset, val_dataset, device, epochs=5, batch_size=16, lr=2e-5, output_dir='checkpoints'):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                               num_warmup_steps=0,
                               num_training_steps=len(train_loader) * epochs)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(output_dir, exist_ok=True)
    best_val_f1 = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask, images)
                pred_labels = torch.argmax(logits, dim=1)
                preds.extend(pred_labels.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")
        print(f"Val Accuracy: {acc:.4f}, F1: {f1:.4f}")

        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            ckpt_path = os.path.join(output_dir, f"best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Best model saved to {ckpt_path}")

