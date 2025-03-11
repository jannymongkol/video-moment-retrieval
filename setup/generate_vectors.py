import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import time

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
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
        
        # When batch is full, process the batch
        if len(frames) == batch_size:
            # Preprocess the frames for CLIP model
            inputs = processor(images=frames, return_tensors="pt")
            
            # Get CLIP features (image embeddings)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            # Append the features (no normalization)
            frame_vectors.append(image_features.cpu().numpy())
            
            # Clear the frames list for the next batch
            frames = []

    # If there are remaining frames after the loop (less than batch_size)
    if len(frames) > 0:
        # Preprocess the remaining frames
        inputs = processor(images=frames, return_tensors="pt")
        
        # Get CLIP features (image embeddings)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        
        # Append the features (no normalization)
        frame_vectors.append(image_features.cpu().numpy())

    # Release the video capture object
    cap.release()
    
    # Concatenate all vectors and save them to a file
    frame_vectors = np.concatenate(frame_vectors, axis=0)
    np.save(output_file, frame_vectors)
    print(f"Feature vectors saved to {output_file}")


def extract_clip_text_vectors(text_list, output_file, batch_size=64):
    # Open the video using OpenCV
    
    texts = []
    text_vectors = []

    for text in text_list:
        texts.append(text)
        
        # When batch is full, process the batch
        if len(texts) == batch_size:
            # Preprocess the frames for CLIP model
            inputs = processor(text=texts, return_tensors="pt", padding=True)
            
            # Get CLIP features (image embeddings)
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
            
            # Append the features (no normalization)
            text_vectors.append(text_features.cpu().numpy())
            
            # Clear the frames list for the next batch
            texts = []

    # If there are remaining frames after the loop (less than batch_size)
    if len(texts) > 0:
        # Preprocess the frames for CLIP model
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        
        # Get CLIP features (image embeddings)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        
        # Append the features (no normalization)
        text_vectors.append(text_features.cpu().numpy())
    
    # Concatenate all vectors and save them to a file
    text_vectors = np.concatenate(text_vectors, axis=0)
    np.save(output_file, text_vectors)
    print(f"Feature vectors saved to {output_file}")
