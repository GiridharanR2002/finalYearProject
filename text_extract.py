# text_extract.py
import os
import cv2
import numpy as np
import easyocr
import torch
from PIL import Image, ImageEnhance, ImageFilter
from spellchecker import SpellChecker

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')

    # Resize if too small or too large
    max_dim = 1024
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        # Ensure compatibility with both old and new Pillow versions
        resample_method = getattr(Image, 'Resampling', Image).LANCZOS
        image = image.resize((int(w * scale), int(h * scale)), resample_method)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)

    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)

    # Convert to numpy and denoise
    img_np = np.array(image)
    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)

    return img_np



def extract_text(image_path, lang='en'):
    reader = easyocr.Reader([lang], gpu=torch.cuda.is_available())
    img_np = preprocess_image(image_path)
    result = reader.readtext(img_np)
    extracted = []
    for line in result:
        try:
            text_piece = line[1]
            extracted.append(text_piece)
        except Exception:
            continue
    raw_text = " ".join(extracted).strip() or "no text found"
    # Spell correction with pyspellchecker (recommended for memes)
    spell = SpellChecker()
    corrected_words = []
    for word in raw_text.split():
        corrected_word = spell.correction(word)
        corrected_words.append(corrected_word if corrected_word else word)
    corrected_text_spell = " ".join(corrected_words)
    return corrected_text_spell

if __name__ == "__main__":
    image_path = "temp_uploaded_image.png" 
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        exit(1)
    text_spell = extract_text(image_path)
    print("PySpellChecker Spell-corrected:")
    print(text_spell)