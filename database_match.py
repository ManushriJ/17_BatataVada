import clip
import torch
from PIL import Image
import numpy as np
import os
import sqlite3
import pickle
from scipy.spatial.distance import cosine
import gradio as gr

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Connect to SQLite database (create if it doesn't exist)
conn = sqlite3.connect("criminal_faces.db", check_same_thread=False)
cursor = conn.cursor()

# Create table for storing image data
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    image_path TEXT,
    embedding BLOB
)
''')
conn.commit()

# Function to check if an image is blank
def is_blank_image(image):
    image_array = np.array(image.convert("L"))  # Convert to grayscale
    return np.std(image_array) < 5  # Check if pixel variance is very low (almost uniform)

# Function to get CLIP embedding
def get_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    
    if is_blank_image(image):  # Check if the image is completely blank
        return None
    
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    embedding = image_features.cpu().numpy().flatten()
    
    if np.allclose(embedding, 0, atol=1e-6):  # Check if embedding is near zero
        return None
    
    return embedding

# Function to process folder and store embeddings in database
def process_image_folder(image_folder):
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        embedding = get_clip_embedding(image_path)
        if embedding is not None:
            cursor.execute("INSERT INTO faces (name, image_path, embedding) VALUES (?, ?, ?)",
                           (image_name, image_path, pickle.dumps(embedding)))
            conn.commit()

# Call function to process images (set the correct folder path)
image_folder = "image_final"  
process_image_folder(image_folder)

# Function to find top matches
def find_top_matches(uploaded_image):
    uploaded_image.save("temp_uploaded.jpg")
    generated_embedding = get_clip_embedding("temp_uploaded.jpg")
    
    if generated_embedding is None:
        return None, "No match found", None, "", None, ""
    
    cursor.execute("SELECT name, image_path, embedding FROM faces")
    
    matches = []
    for name, image_path, embedding_blob in cursor.fetchall():
        stored_embedding = pickle.loads(embedding_blob)
        similarity = 1 - cosine(generated_embedding, stored_embedding.flatten())  # Convert to similarity score
        
        if similarity > 0.5:  # Set a threshold to ignore weak matches
            matches.append((name, image_path, similarity))
    
    matches.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity (higher is better)
    
    # Avoid duplicate matches
    seen_images = set()  # Track seen images to avoid duplicates
    top_matches = []

    for name, img_path, similarity in matches:
        if img_path not in seen_images:
            top_matches.append((name, img_path, similarity))
            seen_images.add(img_path)
        if len(top_matches) == 3:  # Only keep the top 3 matches
            break
    
    # If no matches found
    if not top_matches:
        return None, "No match found", None, "", None, ""
    
    result_images = []
    result_texts = []
    for name, img_path, similarity in top_matches:
        result_images.append(img_path)
        result_texts.append(f"{name} - Confidence: {similarity:.2f}")
    
    return result_images[0] if result_images else None, result_texts[0] if result_texts else "No match", \
           result_images[1] if len(result_images) > 1 else None, result_texts[1] if len(result_texts) > 1 else "", \
           result_images[2] if len(result_images) > 2 else None, result_texts[2] if len(result_texts) > 2 else ""

# Gradio Interface
demo = gr.Interface(
    fn=find_top_matches,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Image(label="Top Match 1"), gr.Text(label="Match 1 Confidence"),
        gr.Image(label="Top Match 2"), gr.Text(label="Match 2 Confidence"),
        gr.Image(label="Top Match 3"), gr.Text(label="Match 3 Confidence")
    ],
    title="Criminal Face Matching",
    description="Upload an image to find the top 3 matches from the database."
)

if __name__ == "__main__":
    demo.launch()
    conn.close()
