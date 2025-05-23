import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer
import open_clip
from src.models import MultimodalHateClassifier
import numpy as np
import sys
import os
from src.generate_caption import generate_caption
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.text_extract import extract_text

css_path = os.path.join(os.path.dirname(__file__), "src", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Load model and preprocessing
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
    _, image_preprocess, _ = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    model = MultimodalHateClassifier()
     
    model_path = os.path.join(os.path.dirname(__file__), "src", "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, tokenizer, image_preprocess

model, tokenizer, image_preprocess = load_model()

st.markdown("<h1 style='text-align: center; color: #2d3436;'>🖼️ Hateful Meme Detection</h1>", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_image = st.file_uploader("Upload meme image", type=["png", "jpg", "jpeg"])
        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded Meme", use_container_width=True)
    with col2:
        st.markdown("#### About")
        st.write(
            "This app uses AI to detect whether a meme contains hateful content. "
            "Upload an image to get started!"
        )

ocr_text = ""
generated_caption = ""
if uploaded_image:
    temp_path = "temp_uploaded_image.png"
    image = Image.open(uploaded_image).convert("RGB")
    image.save(temp_path)
    ocr_text = extract_text(temp_path)
    generated_caption = generate_caption(temp_path)
    st.image(image, caption="Uploaded Meme", use_container_width=True)

if st.button("Predict"):
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        image_tensor = image_preprocess(image).unsqueeze(0)

        # Combine OCR text and generated caption
        full_text = ocr_text.strip()
        if generated_caption.strip():
            full_text += " the image shows " + generated_caption.strip()

        # Preprocess text
        inputs = tokenizer(full_text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Predict
        with torch.no_grad():
            logits = model(input_ids, attention_mask, image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            st.write(f"**Prediction:** {'🛑 Hateful' if pred == 1 else '✅ Not Hateful'}")
            st.write(f"Probability (Hateful): {probs[0,1].item():.4f}")
    else:
        st.warning("Please upload an image before predicting.")
st.markdown(
    """
    <hr>
    <div style='text-align:center; color: #636e72; font-size: 0.9em;'>
        Made by hardwork and efforts | <a href='https://github.com/your-repo' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)