# models.py
import torch
import torch.nn as nn
from transformers import AutoModel
import open_clip

class MultimodalHateClassifier(nn.Module):
    def __init__(self,
                 text_model_name='GroNLP/hateBERT',
                 image_model_name='ViT-B-32',
                 image_pretrained='laion2b_s34b_b79k'):
        super().__init__()

        # Text Encoder: HateBERT
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, 512)

        # Image Encoder: OpenCLIP ViT-B/32
        self.image_encoder, _, _ = open_clip.create_model_and_transforms(
            model_name=image_model_name,
            pretrained=image_pretrained
        )
        self.image_proj = nn.Linear(self.image_encoder.visual.output_dim, 512)

        # Fusion + Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)  # Binary classification: [non-hateful, hateful]
        )

    def forward(self, input_ids, attention_mask, image):
        # Text
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0]  # CLS token
        text_emb = self.text_proj(text_feat)  # (B, 512)

        # Image
        image_feat = self.image_encoder.encode_image(image)  # (B, D)
        image_emb = self.image_proj(image_feat)  # (B, 512)

        # Fusion
        combined = torch.cat([text_emb, image_emb], dim=1)  # (B, 1024)

        # Classification
        logits = self.classifier(combined)  # (B, 2)
        return logits
