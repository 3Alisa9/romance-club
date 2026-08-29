import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_folder = "images/"
embeddings = []
names = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_folder, filename)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs.to(device))
        emb = outputs.detach().cpu().numpy().astype(np.float32)
        embeddings.append(emb)
        name = os.path.splitext(filename)[0]
        names.append(name)
        print(f"Индексирован: {name}")

embeddings = np.vstack(embeddings)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "image_index.faiss")

with open("names.txt", "w", encoding="utf-8") as f:
    for name in names:
        f.write(name + "\n")

print("Индексация завершена!")
