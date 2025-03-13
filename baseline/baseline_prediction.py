import json
import sys
import numpy as np

def pred(video_id, pred_clip_length=128):

    # video embeddings shape: (num_frames, embedding_size)
    video_embeddings = np.load(f'data/clip_video_feature_vector/{video_id}.npy')
    video_embeddings = video_embeddings / np.linalg.norm(video_embeddings, axis=1, keepdims=True)
    # sentence embeddings shape: (num_sentences, embedding_size)
    sentence_embeddings = np.load(f'data/clip_text_feature_vector/{video_id}.npy')
    sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    # similarity_matrix shape: (num_frames, num_sentences)
    similarity_matrix = video_embeddings @ sentence_embeddings.T
    
    # predictions shape: (num_sentence,)
    # predictions[i] is the frame index that is most similar to the i-th sentence
    predictions = np.argmax(similarity_matrix, axis=0)
    
    num_frames = video_embeddings.shape[0]
    
    # The videos are 16 fps. I want to get a 6 second clip of the video, with the predicted frame in the middle
    # Give me lower and upper bounds for the clip
    lower_bound = np.maximum(0, predictions - pred_clip_length//2)
    upper_bound = np.minimum(num_frames, predictions + pred_clip_length//2)
    
    # Concatenate the lower and upper bounds to form a (num_sentences, 2) array, and return it
    predictions = np.stack([lower_bound, upper_bound], axis=0)
    return predictions.T
    
    

def get_predictions(json_file_path, prediction_output_path):
    # Open and load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    output = {}
    
    # Loop through all the keys of the dictionary
    for key in data.keys():
        predictions = pred(key)
        output[key] = predictions.tolist()
    
    # Save json file to the output path
    with open(prediction_output_path, 'w') as file:
        json.dump(output, file)
        

if __name__ == "__main__":
    # Get the json file path and the prediction output path from the command line
    json_file_path = sys.argv[1]
    prediction_output_path = sys.argv[2]
    
    get_predictions(json_file_path, prediction_output_path)