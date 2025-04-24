import torch

import os
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bert_based.model import MomentBERT
from bert_based.data_loader import MomentDataset
from bert_based.model import DiceLoss, IntervalBCELoss, StartEndBCELoss
from bert_based.predict_utils import kmeans_region_detection, visualize_intervals, visualize_random_subsamples, visualize_baseline
from eval.rn_IoUm import r1_IoUm_sum


def train(model, dataset, loss_fn, optimizer, device, val_dataset=None):
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

    # Create progress bar
    pbar = tqdm(enumerate(dataset), total=num_batches, desc='Training', leave=True)
    num_train_examples = 0
    for idx, data in pbar:
        queries = data['sentences']  # List of sentences, length B
        video_clip_embeddings = torch.from_numpy(data['embedding']).to(device)  # Video embeddings (num_frames, clip_dim)
        if video_clip_embeddings.shape[0] > model.max_video_len:
            continue
        framestamps = data['framestamps']

        optimizer.zero_grad()
        
        result = model.forward(queries, video_clip_embeddings) # (B, N_frames)

        # predictions: (B, N_frames)
        # gt: [[start, end], [start, end]]
        # Compute loss
        loss = loss_fn(result, framestamps)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_train_examples += len(queries)

        if (idx + 1) % 10 == 0 or (idx + 1) == num_batches:
            # Update progress bar
            pbar.set_postfix({
                'batch': f"{idx+1}/{num_batches}",
                'loss': f'{loss.item():.4f}'}
            )

    # Epoch-level summary
    avg_loss = total_loss / num_train_examples
    
    if val_dataset is not None:
        model.eval()
        val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for idx, data in enumerate(val_dataset):
                queries = data['sentences']
                video_clip_embeddings = torch.from_numpy(data['embedding']).to(device)
                framestamps = data['framestamps']
                
                if video_clip_embeddings.shape[0] > model.max_video_len:
                    continue

                num_val_batches += len(queries)
                result = model.forward(queries, video_clip_embeddings)
                loss = loss_fn(result, framestamps)
                val_loss += loss.item()

        avg_val_loss = val_loss / num_val_batches
        print(f"Validation Loss: {avg_val_loss:.4f}")
    
        return avg_loss, avg_val_loss
    else:
        return avg_loss


if __name__ == "__main__":
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)
    
    if sys.argv[1] == "train":
        # Create checkpoint directory
        os.makedirs("checkpoints", exist_ok=True)
        
        # Model setup
        model = MomentBERT(num_hidden=1, prediction_head='start_end').to(device)
        train_dataset = MomentDataset(
            dataset_json_file="data/Charades-CD/charades_train.json",
            embedding_dir="data/clip_video_feature_vector",
            frame_rate=4
        )
        
        val_dataset = MomentDataset(
            dataset_json_file="data/Charades-CD/charades_val.json",
            embedding_dir="data/clip_video_feature_vector",
            frame_rate=4
        )    
        num_epochs = 5

        loss_fn = StartEndBCELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Training tracking
        best_loss = float('inf')
        best_epoch = -1
        start_time = datetime.now()
        
        # Training loop with progress bar for epochs
        for epoch in tqdm(range(num_epochs), desc='Epochs', position=0):
            # Assuming loss_fn and optimizer are defined
            avg_train_loss, avg_val_loss = train(model, train_dataset, loss_fn, optimizer, device, val_dataset)
            
            print(f"Average Train Loss: {avg_train_loss:.4f}")
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            # Save current epoch
            torch.save(checkpoint, f"checkpoints/moment_bert_1_hid_stend_epoch_{epoch+1}.pt")
            
            # Update best model if needed
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch + 1
                torch.save(checkpoint, "checkpoints/moment_bert_1_hid_stend_best.pt")
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        # Print final summary
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Total training time: {training_time}")
        print(f"Best model saved at epoch {best_epoch} with loss: {best_loss:.4f}")
        print(f"Best model checkpoint: checkpoints/moment_bert_best.pt")
        print(f"Final model checkpoint: checkpoints/moment_bert_epoch_{num_epochs}.pt")
        print("="*50)
        
    elif sys.argv[1] == "predict":
        # Load the best model
        checkpoint_fname = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/moment_bert_1_hid_stend_best.pt"
        checkpoint = torch.load(checkpoint_fname)
        model = MomentBERT(
            num_hidden=1, 
            prediction_head='start_end'
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create test dataset
        test_iid_dataset = MomentDataset(
            dataset_json_file="data/Charades-CD/charades_test_iid.json",
            embedding_dir="data/clip_video_feature_vector",
            frame_rate=4
        )
        
        # # Create test dataset
        # test_ood_dataset = MomentDataset(
        #     dataset_json_file="data/Charades-CD/charades_test_ood.json",
        #     embedding_dir="data/clip_video_feature_vector",
        #     frame_rate=4
        # )
        
        # Make predictions
        predictions = predict(model, test_iid_dataset, device, predictor_fn="argmax")
        
        # Save predictions to JSON file
        with open("bert_test_iid.json", "w") as f:
            json.dump(predictions, f)
            
        # predictions = predict(model, test_ood_dataset, device, predictor_fn="kmeans")
        
        # # Save predictions to JSON file
        # with open("bert_test_ood.json", "w") as f:
        #     json.dump(predictions, f)

    elif sys.argv[1] == "analyze":
        pred_fname = sys.argv[2]
        source_fname = sys.argv[3]

        run_label = sys.argv[4]
        split_label = sys.argv[5]

        if run_label == "baseline":
            visualize_baseline(
                source=source_fname,
                predictions=pred_fname,
                run_label=run_label,
                split_label=split_label,
                num_sample=5,
                frame_rate=16
            )
        else:
            visualize_random_subsamples(
                source=source_fname,
                predictions=pred_fname,
                run_label=run_label,
                split_label=split_label,
                num_sample=5,
                frame_rate=4
            )


        
        # Make predictions
        # predictions = predict(model, test_iid_dataset, device, predictor_fn="kmeans")
        
        # # Save predictions to JSON file
        # with open("bert_test_iid.json", "w") as f:
        #     json.dump(predictions, f)
            
        # predictions = predict(model, test_ood_dataset, device, predictor_fn="kmeans")
        
        # # Save predictions to JSON file
        # with open("bert_test_ood.json", "w") as f:
        #     json.dump(predictions, f)