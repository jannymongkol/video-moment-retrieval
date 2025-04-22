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
from bert_based.model import DiceLoss, IntervalBCELoss
from bert_based.predict_utils import kmeans_region_detection, visualize_intervals, visualize_random_subsamples
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
        loss = loss_fn(result, framestamps, reduction='sum')
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
                loss = loss_fn(result, framestamps, reduction='sum')
                val_loss += loss.item()

        avg_val_loss = val_loss / num_val_batches
        print(f"Validation Loss: {avg_val_loss:.4f}")
    
        return avg_loss, avg_val_loss
    else:
        return avg_loss

def predict(model, test_dataset, device, ious=[0.1, 0.3, 0.5, 0.7, 0.9], predictor_fn="argmax"):
    """
    Prediction loop for video moment retrieval.

    model: MomentBERT
    dataloader (MomentDataset): yields dict with tokenized text, video embeddings, labels
    device: CUDA or CPU
    """
    model.eval()
    predictions = {}
    output = {}
    
    results = {}
    total_sum_r1_iou = {iou: 0 for iou in ious}
    total_queries = 0

    with torch.no_grad():
        for i,data in tqdm(enumerate(test_dataset)):
            
            key = data['video_id']
            queries = data['sentences']
            video_clip_embeddings = torch.from_numpy(data['embedding']).to(device)
            
            if video_clip_embeddings.shape[0] > model.max_video_len:
                continue

            result = model.forward(queries, video_clip_embeddings)
            result = torch.sigmoid(result)
            result = result.cpu().numpy()
            
            if predictor_fn == "argmax":
                pred_clip_length = 8 * data['decode_fps'] # placeholder
                max_framestamps = np.argmax(result, axis=1) # (batch_size, 1)
                num_frames = result.shape[1]

                lower_bound = np.maximum(0, max_framestamps - pred_clip_length//2)
                upper_bound = np.minimum(num_frames, max_framestamps + pred_clip_length//2)
                bounds = np.stack([lower_bound, upper_bound], axis=0).T # (2, batch_size)
            elif predictor_fn == "kmeans":
                # Use kmeans to find the regions
                spans = kmeans_region_detection(result)
                bounds = np.stack(spans, axis=1).T # (2, batch_size)
            
            output[key] = bounds.tolist()
            predictions[key] = result.tolist()

            gt_intervals = np.array(data['framestamps'])
            pred_intervals = bounds
            
            total_queries += gt_intervals.shape[0]
            for iou in ious:
                total_sum_r1_iou[iou] += np.sum(r1_IoUm_sum(gt_intervals, pred_intervals, iou))
            
    for iou in ious:
        total_sum_r1_iou[iou] /= total_queries
        results[iou] = {
            'R1 IoU': total_sum_r1_iou[iou],
        }
        print(f"R1 IoU@{iou}: {total_sum_r1_iou[iou]:.4f}")

    with open(f"predictions_{predictor_fn}_iid.json", "w") as f:
        # json.dump(output, f)
        json.dump(predictions, f)
        print(f"Predictions saved to predictions_{predictor_fn}.json")
        
    return results

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
        model = MomentBERT(num_hidden=1).to(device)
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

        loss_fn = IntervalBCELoss()
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
            torch.save(checkpoint, f"checkpoints/moment_bert_1_hidden_epoch_{epoch+1}.pt")
            
            # Update best model if needed
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch + 1
                torch.save(checkpoint, "checkpoints/moment_bert_1_hidden_best.pt")
        
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
        checkpoint = torch.load("checkpoints/moment_bert_best.pt")
        model = MomentBERT(num_hidden=0).to(device)
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
        predictions = predict(model, test_iid_dataset, device, predictor_fn="kmeans")
        
        # Save predictions to JSON file
        with open("bert_test_iid.json", "w") as f:
            json.dump(predictions, f)
            
        # predictions = predict(model, test_ood_dataset, device, predictor_fn="kmeans")
        
        # # Save predictions to JSON file
        # with open("bert_test_ood.json", "w") as f:
        #     json.dump(predictions, f)

    elif sys.argv[1] == "analyze":
        pred_fname = sys.argv[2] if len(sys.argv) > 2 else "predictions_kmeans_ood.json"
        source_fname = sys.argv[3] if len(sys.argv) > 3 else "data/Charades-CD/charades_test_ood.json"

        visualize_random_subsamples(
            source=source_fname,
            predictions=pred_fname,
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