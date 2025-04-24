import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from frame_by_frame_bert.dataloader import TemporalGroundingDataset, collate_fn
from frame_by_frame_bert.model import TemporalGroundingModel

if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(device)
    
video_embeddings_dir = 'data/clip_video_feature_vector'
annotations_file = 'data/Charades-CD/charades_train.json'
val_annotations_file = 'data/Charades-CD/charades_val.json'

dataset = TemporalGroundingDataset(video_embeddings_dir, annotations_file)
val_dataset = TemporalGroundingDataset(video_embeddings_dir, val_annotations_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


model = TemporalGroundingModel(clip_dim=512).to(device)
save_dir = 'checkpoints/frame_by_frame_bert'
os.makedirs(save_dir, exist_ok=True)

# === Loss + Optimizer ===
criterion = nn.BCEWithLogitsLoss()  # Since we're using logits
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# === Training Loop ===
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch in progress:
        clip_embeddings = batch['clip_embeddings'].to(device)  # (B, D)
        texts = batch['texts']                                 # list of strings
        labels = batch['labels'].float().to(device).unsqueeze(1)  # (B, 1)

        # === Forward ===
        logits = model(clip_embeddings, texts)

        # === Loss ===
        loss = criterion(logits, labels)

        # === Backward ===
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(dataloader)

    model.eval()
    val_loss = 0.0
    val_batches = len(val_loader)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
            clip_embeddings = batch['clip_embeddings'].to(device)
            texts = batch['texts']
            labels = batch['labels'].float().to(device).unsqueeze(1)

            logits = model(clip_embeddings, texts)
            loss = criterion(logits, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / val_batches

    print(f"Epoch {epoch+1} completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # === Save Checkpoint ===
    save_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
    }, save_path)

    print(f"Model saved to {save_path}")