import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import time

# Set device to MPS if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to extract CLIP features in batches
def extract_clip_vid_vectors(video_path, output_file, batch_size=64):
    # Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    print(f"num_frames", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"fps", cap.get(cv2.CAP_PROP_FPS))
    print(f"frame_width", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"frame_height", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # List to hold all the image data
    frames = []
    frame_vectors = []

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Append the frame to the list (we'll process in batches later)
        frames.append(frame_rgb)

        if len(frames) == batch_size:
            inputs = processor(images=frames, return_tensors="pt").to(device)

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            frame_vectors.append(image_features.cpu().numpy())
            frames = []

    if len(frames) > 0:
        inputs = processor(images=frames, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        frame_vectors.append(image_features.cpu().numpy())

    cap.release()
    frame_vectors = np.concatenate(frame_vectors, axis=0)
    np.save(output_file, frame_vectors)
    print(f"Feature vectors saved to {output_file}")


def extract_clip_text_vectors(text_list, output_file, batch_size=64):
    texts = []
    text_vectors = []

    for text in text_list:
        texts.append(text)

        if len(texts) == batch_size:
            inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                text_features = model.get_text_features(**inputs)

            text_vectors.append(text_features.cpu().numpy())
            texts = []

    if len(texts) > 0:
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        text_vectors.append(text_features.cpu().numpy())

    text_vectors = np.concatenate(text_vectors, axis=0)
    np.save(output_file, text_vectors)
    print(f"Feature vectors saved to {output_file}")
