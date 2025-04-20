import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class MomentDataset(torch.utils.data.Dataset):
    def __init__(self, video_embeddings_list, queries_list, start_indices_list, end_indices_list):
        """
        video_embeddings_list: list of torch.Tensor (N_i, clip_dim) for each video
        queries_list: list of lists of str — each sublist contains queries for that video
        start_indices_list: list of lists of int — start frame index for each query
        end_indices_list: list of lists of int — end frame index for each query
        """
        self.video_embeddings_list = video_embeddings_list
        self.queries_list = queries_list
        self.start_indices_list = start_indices_list
        self.end_indices_list = end_indices_list

    def __len__(self):
        return len(self.video_embeddings_list)  # number of videos

    def __getitem__(self, idx):
        return {
            'video': self.video_embeddings_list[idx],        # (N, clip_dim)
            'queries': self.queries_list[idx],               # list of str
            'start_indices': self.start_indices_list[idx],   # list of ints
            'end_indices': self.end_indices_list[idx]        # list of ints
        }

def video_query_collate_fn(batch, tokenizer):
    """
    batch: list of dicts from __getitem__
    tokenizer: pre-initialized tokenizer (e.g. BertTokenizer)
    Returns a dict of:
      - input_ids: (M, T)
      - attention_mask: (M, T)
      - video_embeddings: (M, N, clip_dim)
      - start_idx: (M,)
      - end_idx: (M,)
    """
    all_queries = []
    all_videos = []
    all_start = []
    all_end = []

    for item in batch:
        video = item['video']            # (N, D)
        queries = item['queries']
        starts = item['start_indices']
        ends = item['end_indices']

        for q, s, e in zip(queries, starts, ends):
            all_queries.append(q)
            all_videos.append(video)
            all_start.append(s)
            all_end.append(e)

    # Tokenize all queries together
    encodings = tokenizer(
        all_queries,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encodings['input_ids']              # (M, T)
    attention_mask = encodings['attention_mask']    # (M, T)

    # Pad video tensors to max length
    max_len = max(v.shape[0] for v in all_videos)
    clip_dim = all_videos[0].shape[1]
    padded_videos = torch.zeros(len(all_videos), max_len, clip_dim)

    for i, v in enumerate(all_videos):
        padded_videos[i, :v.shape[0]] = v

    return {
        'input_ids': input_ids,                           # (M, T)
        'attention_mask': attention_mask,                 # (M, T)
        'video_clip_embeddings': padded_videos,           # (M, max_N, D)
        'start_frame_idx': torch.tensor(all_start),       # (M,)
        'end_frame_idx': torch.tensor(all_end)            # (M,)
    }


if __name__ == "__main__":
    # Example:
    video_embeddings = torch.randn(10, 256, 512)  # (Batch_size, num_frames, clip_dim)
    text_queries = [['What happens in this scene?', 'Who is in the shot?']] * 10  # (Batch_size, num_queries)
    start_frame_idx = torch.randint(0, 256, (10,))  # (Batch_size,)
    end_frame_idx = torch.randint(0, 256, (10,))  # (Batch_size,)

    # Initialize Dataset and DataLoader
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MomentDataset(video_embeddings, text_queries, start_frame_idx, end_frame_idx)
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # this is number of videos; can be tuned
        collate_fn=lambda batch: video_query_collate_fn(batch, tokenizer),
        shuffle=True
    )
