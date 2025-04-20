import torch

from bert_based.model import MomentBERT

def train(model, dataloader, loss_fn, optimizer, device):
    """
    Training loop for video moment retrieval.

    model: VideoTextBERTWithPredictionHead
    dataloader: yields dict with tokenized text, video embeddings, labels
    loss_fn: VideoMomentLoss
    optimizer: AdamW or similar
    device: CUDA or CPU
    """
    model.train()
    total_loss = 0
    total_start_loss = 0
    total_end_loss = 0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)                    # (M, T)
        attention_mask = batch['attention_mask'].to(device)          # (M, T)
        video_clip_embeddings = batch['video_clip_embeddings'].to(device)  # (M, N, D)
        start_frame_idx = batch['start_frame_idx'].to(device)        # (M,)
        end_frame_idx = batch['end_frame_idx'].to(device)            # (M,)

        optimizer.zero_grad()

        # Forward pass
        start_logits, end_logits = model(input_ids, attention_mask, video_clip_embeddings)

        # Compute loss
        total_batch_loss, start_loss, end_loss = loss_fn(start_logits, end_logits, start_frame_idx, end_frame_idx)

        # Backprop + optimize
        total_batch_loss.backward()
        optimizer.step()

        # Logging
        total_loss += total_batch_loss.item()
        total_start_loss += start_loss.item()
        total_end_loss += end_loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            print(f"Batch {batch_idx+1}/{num_batches} | "
                  f"Loss: {total_batch_loss.item():.4f} | "
                  f"Start: {start_loss.item():.4f} | End: {end_loss.item():.4f}")

    # Epoch-level summary
    avg_loss = total_loss / num_batches
    avg_start = total_start_loss / num_batches
    avg_end = total_end_loss / num_batches
    print(f"\nEpoch Summary — Loss: {avg_loss:.4f} | Start: {avg_start:.4f} | End: {avg_end:.4f}")

    return avg_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MomentBERT().to(device)
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # dataset = MomentDataset(video_embeddings, text_queries, start_frame_idx, end_frame_idx)
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=4,  # this is number of videos; can be tuned
    #     collate_fn=lambda batch: video_query_collate_fn(batch, tokenizer),
    #     shuffle=True
    # )

    # loss_fn = VideoMomentLoss() #TODO
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # for epoch in range(num_epochs):
    #     print(f"\nEpoch {epoch+1}/{num_epochs}")
    #     train(model, dataloader, loss_fn, optimizer, device)