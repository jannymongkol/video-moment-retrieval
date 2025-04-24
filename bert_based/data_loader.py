import json
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch

class MomentDataset(Dataset):
    """
    Dataset for Charades video data with video embeddings
    """
    def __init__(self, dataset_json_file, embedding_dir, frame_rate=16):
        """
        Args:
            json_file (str): Path to the JSON file with charades data
            embedding_dir (str): Directory containing the video embeddings
        """
        assert(frame_rate <= 16)
        assert(16 % frame_rate == 0)
        self.frame_rate = frame_rate

        # Load the dataset
        with open(dataset_json_file, 'r') as f:
            self.data = json.load(f)
        
        # Filter video IDs to only include those with existing embedding files
        self.video_ids = []
        for video_id in self.data.keys():
            embedding_path = os.path.join(embedding_dir, f"{video_id}.npy")
            if os.path.exists(embedding_path):
                self.video_ids.append(video_id)
        
        # Optionally print how many videos were filtered out
        print(f"Kept {len(self.video_ids)} out of {len(self.data)} videos (filtered out {len(self.data) - len(self.video_ids)} with missing embeddings)")
        
        # Store embedding directory and model ID
        self.embedding_dir = embedding_dir
    
    def __len__(self):
        """Return the number of videos in the dataset"""
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        """
        Get a video and its associated data
        Downsampled by the frate rate
        Args:
            idx (int): Index of the video to return
        
        Returns:
            dict: Dictionary containing the video data and embedding
        """
        # Get the video ID
        video_id = self.video_ids[idx]
        
        # Get the video data
        video_data = self.data[video_id]
        
        # Load the video embedding
        embedding_path = os.path.join(self.embedding_dir, f"{video_id}.npy")
        embedding = np.load(embedding_path)
        # downsample
        if self.frame_rate < 16:
            downsample_ratio = 16 // self.frame_rate
            embedding = embedding[::downsample_ratio]
            
        # Prepare return values
        result = {
            # metadata as strings
            'video_id': video_id,
            'sentences': video_data['sentences'],

            # raw outputs as strings
            'timestamps': video_data['timestamps'],
            'video_duration': video_data['video_duration'],

            # sampling related
            'decode_fps': self.frame_rate,
            'embedding': embedding, # (num_frames, clip_dim)
        }

        # Downsample
        framestamps = video_data['framestamps']
        if self.frame_rate < 16:
            downsample_ratio = 16 // self.frame_rate
            framestamps = [ 
                [round(start / downsample_ratio), 
                 round(end / downsample_ratio)]
                    for start, end in video_data['framestamps']
            ]
        result['framestamps'] = framestamps

        return result
    
class MomentPairDataset(Dataset):
    """
    Dataset of (video, text query) pairs with corresponding timestamp labels.
    Each sample is one video-text pair.
    """
    def __init__(self, dataset_json_file, embedding_dir, frame_rate=16):
        assert frame_rate <= 16 and 16 % frame_rate == 0
        self.frame_rate = frame_rate

        with open(dataset_json_file, 'r') as f:
            self.raw_data = json.load(f)

        self.embedding_dir = embedding_dir
        self.samples = []

        for video_id, video_data in self.raw_data.items():
            embedding_path = os.path.join(embedding_dir, f"{video_id}.npy")
            if not os.path.exists(embedding_path):
                continue  # skip missing embeddings

            # Each sentence-query gets its own sample
            for sentence, framestamp in zip(video_data["sentences"], video_data["framestamps"]):
                self.samples.append({
                    "video_id": video_id,
                    "sentence": sentence,
                    "framestamp": framestamp,
                    "video_duration": video_data["video_duration"]
                })

        print(f"Created {len(self.samples)} video-text pairs from {len(self.raw_data)} videos.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        sentence = sample["sentence"]
        start_f, end_f = sample["framestamp"]

        # Load and downsample embeddings
        embedding_path = os.path.join(self.embedding_dir, f"{video_id}.npy")
        embedding = np.load(embedding_path)
        if self.frame_rate < 16:
            downsample_ratio = 16 // self.frame_rate
            embedding = embedding[::downsample_ratio]
            start_f = round(start_f / downsample_ratio)
            end_f = round(end_f / downsample_ratio)

        return {
            "video_id": video_id,
            "sentence": sentence,
            "embedding": embedding,  # (num_frames, clip_dim)
            "framestamp": [start_f, end_f],
            "video_duration": sample["video_duration"],
            "decode_fps": self.frame_rate,
        }

def MomentPairDataset_collate_fn(batch):
    """
    Custom collate function for MomentPairDataset.
    Pads video embeddings and keeps metadata.
    """
    # Get max number of frames
    max_len = max(item['embedding'].shape[0] for item in batch)
    clip_dim = batch[0]['embedding'].shape[1]

    # Pad video embeddings
    padded_embeddings = []
    original_lengths = []

    for item in batch:
        emb = item['embedding']
        original_len = emb.shape[0]
        original_lengths.append(original_len)

        # Pad with zeros to max_len
        if original_len < max_len:
            pad_width = ((0, max_len - original_len), (0, 0))
            emb = np.pad(emb, pad_width, mode='constant')
        
        padded_embeddings.append(torch.tensor(emb, dtype=torch.float32))

    # Stack embeddings: (B, max_len, clip_dim)
    embedding_tensor = torch.stack(padded_embeddings, dim=0)
    original_lengths = torch.tensor(original_lengths, dtype=torch.long)

    # Keep metadata list (excluding embeddings)
    metadata = [
        {
            key: item[key] for key in item if key != 'embedding'
        }
        for item in batch
    ]

    return {
        'embedding': embedding_tensor,         # (B, max_len, clip_dim)
        'original_lengths': original_lengths,  # (B,)
        'metadata': metadata                   # List of dicts
    }


# Example usage:
def create_charades_dataloader(json_file, embedding_dir, batch_size=1, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the Charades dataset with video embeddings
    
    Args:
        json_file (str): Path to the JSON file with charades data
        embedding_dir (str): Directory containing the video embeddings
        batch_size (int): Batch size for the dataloader
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of workers for the dataloader
    
    Returns:
        DataLoader: PyTorch DataLoader for the Charades dataset
    """
    dataset = MomentDataset(json_file, embedding_dir, frame_rate=16)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

# Example usage
if __name__ == "__main__":
    # Create dataloader
    json_file = "data/Charades-CD/charades_val.json"
    embedding_dir = "data/clip_video_feature_vector"
    dataloader = create_charades_dataloader(json_file, embedding_dir, shuffle=False)
    
    # Iterate through the dataloader
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Just show first 3 examples
            break

        print(batch['embedding'])
        print(batch['framestamps'])
        print(batch['timestamps'])
        print(batch['sentences'])