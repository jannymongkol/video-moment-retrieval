import torch

from bert_based.model import MomentBERT
from bert_based.data_loader import MomentDataset

def train(model, dataset, loss_fn, optimizer, device):
    """
    Training loop for video moment retrieval.

    model: MomentBERT
    dataloader (MomentDataset): yields dict with tokenized text, video embeddings, labels
    loss_fn: VideoMomentLoss
    optimizer: AdamW or similar
    device: CUDA or CPU
    """
    model.train()
    total_loss = 0
    num_batches = len(dataset)

    for idx, data in enumerate(dataset):

        queries = data['sentences']  # List of sentences, length B
        video_clip_embeddings = data['embedding'].to(device)  # Video embeddings (num_frames, clip_dim)
        framestamps = data['framestamps']

        optimizer.zero_grad()
        
        result = model.forward(queries, video_clip_embeddings) # (B, N_frames)

        # predictions: (B, N_frames)
        # gt: [[start, end], [start, end]]
        # Compute loss
        loss = loss_fn(result, framestamps)

        # Backprop + optimize
        loss.backward()
        optimizer.step()

        # Logging
        total_loss += loss.item()

        if (idx + 1) % 10 == 0 or (idx + 1) == num_batches:
            print(f"Batch {idx+1}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | ")
            
    # Epoch-level summary
    avg_loss = total_loss / num_batches
    return avg_loss


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model = MomentBERT().to(device)
    train_dataset = MomentDataset(
        json_file="data/Charades-CD/charades_train.json",
        embedding_dir="data/clip_video_feature_vector",
        frame_rate=4
    )
    
    num_epochs = 10
    
    loss_fn = VideoMomentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for i in range(num_epochs):
        print(f"\nEpoch {i+1}/{num_epochs}")
        # Assuming loss_fn and optimizer are defined
        avg_loss = train(model, train_dataset, loss_fn, optimizer, device)

    

    # for epoch in range(num_epochs):
    #     print(f"\nEpoch {epoch+1}/{num_epochs}")
    #     train(model, dataloader, loss_fn, optimizer, device)