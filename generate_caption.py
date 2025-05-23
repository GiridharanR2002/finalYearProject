# generate_caption.py
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import os

import warnings
warnings.filterwarnings("ignore") # Suppress Python warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: suppress oneDNN info
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)  # Suppress Hugging Face warnings 
# Load the ViT-GPT2 model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    features = feature_extractor(images=image, return_tensors="pt")
    pixel_values = features.pixel_values.to(device)
    attention_mask = features.attention_mask.to(device) if "attention_mask" in features else None

    # Generate the caption
    with torch.no_grad():
        if attention_mask is not None:
            output_ids = model.generate(pixel_values, attention_mask=attention_mask, max_length=50)
        else:
            output_ids = model.generate(pixel_values, max_length=50)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

# Example usage
if __name__ == "__main__":
    img_path = "C:\\Users\\girid\\Desktop\\Hate_meme\\data\\img\\98764.png"  # Replace with your meme image path
    caption = generate_caption(img_path)
    print("Generated Caption:", caption)