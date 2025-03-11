# Given an input json file, where outermost keys are video ids, 
# calculate the feature vectors for each video and save them to their own files.
#

import json
from generate_vectors import extract_clip_text_vectors

# Take path of input file as cmd line arg
import sys
input_file = sys.argv[1]

# Load the input json file
with open(input_file, "r") as f:
    data = json.load(f)

# Loop over each video in the input file
for video_id in data.keys():
    print(f"Processing video: {video_id}")
    
    texts = data[video_id]['sentences']
    
    # Output file path
    output_file = f"data/clip_text_feature_vector/{video_id}.npy"
    extract_clip_text_vectors(texts, output_file, batch_size=128)