# dataset.py
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from transformers import AutoTokenizer

class HateMemeDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, tokenizer_name='GroNLP/hateBERT', max_length=128, mode='train'):
        self.samples = [json.loads(line) for line in open(jsonl_path)]
        self.image_dir = image_dir
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4815, 0.4578, 0.4082],
                                 std=[0.2686, 0.2613, 0.2758])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.image_dir, os.path.basename(item['img']))
        image = self.image_transform(Image.open(img_path).convert("RGB"))

        # Use both text and caption
        text = item["text"]
        if "caption" in item and item["caption"]:
            text += " " + item["caption"]

        encoded_text = self.tokenizer(text, padding='max_length', truncation=True,
                                      max_length=128, return_tensors='pt')

        label = int(item["label"]) if self.mode != 'test' else -1

        return {
            'image': image,
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'id': item['id']
        }
