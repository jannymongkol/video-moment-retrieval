from torch.utils.data import Dataset
import torch
import numpy as np
import random
import json

class TemporalGroundingDataset(Dataset):
    def __init__(self, video_embeddings_dir, annotations_file, num_pos=2, num_neg=4):
        self.video_embeddings_dir = video_embeddings_dir
        self.num_pos = num_pos
        self.num_neg = num_neg

        self.samples = []
        self.video_lengths = {}
        
        # load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        for vid, ann in annotations.items():
            for sentence, (start, end) in zip(ann['sentences'], ann['framestamps']):
                self.samples.append((vid, sentence, start, end))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, query, start, end = self.samples[idx]
        clip_path = f"{self.video_embeddings_dir}/{vid}.npy"
        clip_array = np.load(clip_path, mmap_mode="r")
        num_frames = clip_array.shape[0]
        
        # Ensure start and end are within bounds
        end = min(end, num_frames - 1)
        start = max(start, 0)

        # Positive sampling
        pos_ids = list(range(start, end + 1))
        pos_sampled = random.choices(pos_ids, k=min(self.num_pos, len(pos_ids)))

        # Negative sampling
        neg_ids = list(set(range(num_frames)) - set(pos_ids))
        neg_sampled = random.choices(neg_ids, k=min(self.num_neg, len(neg_ids)))

        # Load embeddings
        frame_ids = pos_sampled + neg_sampled
        labels = [1] * len(pos_sampled) + [0] * len(neg_sampled)

        clip_embeddings = torch.stack([
            torch.tensor(clip_array[i], dtype=torch.float32) for i in frame_ids
        ])

        return {
            'text': query,
            'clip_embeddings': clip_embeddings,
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def collate_fn(batch):
    texts = []
    for item in batch:
        n_frames = item['clip_embeddings'].shape[0]
        texts.extend([item['text']] * n_frames)

    clip_embeddings = torch.cat([item['clip_embeddings'] for item in batch], dim=0)
    labels = torch.cat([item['labels'] for item in batch], dim=0)

    return {
        'texts': texts,
        'clip_embeddings': clip_embeddings, # (total_frames, clip_dim)
        'labels': labels # (total_frames,)
    }