# Necessary imports
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# Load the pre-trained EfficientNetB0 model using PyTorch
model = models.efficientnet_b0(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Image transformation pipeline (resizing, converting to tensor, and normalizing)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Function to compute cosine similarity between two images
def model_sim(img_path_1, img_path_2):
    img_1 = Image.open(img_path_1).convert("RGB")
    img_1 = preprocess(img_1).unsqueeze(0)  # Add batch dimension

    img_2 = Image.open(img_path_2).convert("RGB")
    img_2 = preprocess(img_2).unsqueeze(0)

    with torch.no_grad():
        outputs_1 = model(img_1)
        outputs_2 = model(img_2)
    
    # Compute cosine similarity
    dot_product = torch.dot(outputs_1.squeeze(), outputs_2.squeeze())
    magnitude_1 = torch.norm(outputs_1)
    magnitude_2 = torch.norm(outputs_2)
    cosine_sim = dot_product / (magnitude_1 * magnitude_2)
    
    return cosine_sim.item()

# Folder containing images
folder_path = "path/to/directory/containing/images"
files = os.listdir(folder_path)

# List to store the filenames and their scores
filenames = []
cosine_scores = []

# Image path for reference image
img_path = "path/to/reference/image"

# Streamlit app UI
st.title("Image Similarity Leaderboard")

# Loop through all images and calculate cosine similarity
for name in files:
    image_path = os.path.join(folder_path, name)
    
    filenames.append(name)
    
    # Calculate similarity score
    score = model_sim(img_path, image_path)
    cosine_scores.append(score)

# Sort the scores and filenames together in descending order
sorted_scores_indices = np.argsort(cosine_scores)[::-1]

sorted_scores = [cosine_scores[i] for i in sorted_scores_indices]
sorted_filenames = [filenames[i] for i in sorted_scores_indices]

# Display the winning image
winning_image = sorted_filenames[0]
winning_score = sorted_scores[0]

st.header("Winning Image")
winning_image_path = os.path.join(folder_path, winning_image)
image = Image.open(winning_image_path)
st.image(image, caption=f"Winning Image: {winning_image} - Score: {winning_score:.5f}")

# Display the leaderboard
st.header("Leaderboard")
for idx, (filename, score) in enumerate(zip(sorted_filenames, sorted_scores)):
    st.write(f"{idx + 1}. {filename} - Score: {score:.5f}")