from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import tempfile
import os

app = Flask(__name__)
CORS(app)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

index = faiss.read_index("image_index.faiss")
with open("names.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({"error": "Нет файла"}), 400
    file = request.files['image']
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        file.save(tmp.name)
        img = Image.open(tmp.name).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs.to(device))
    query_emb = outputs.cpu().numpy().astype(np.float32)
    distances, indices = index.search(query_emb, 1)
    best_idx = indices[0][0]
    name = names[best_idx]
    os.unlink(tmp.name)
    return jsonify({"name": name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
