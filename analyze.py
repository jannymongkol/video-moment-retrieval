import torch
import torch.nn.functional as F

import os
import sys
import numpy as np
from scipy.stats import gamma
from tqdm import tqdm
from datetime import datetime
import json

from get_metadata import get_prob_interval_len
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bert_based.model import MomentBERT
from bert_based.data_loader import MomentDataset
from bert_based.model import DiceLoss, IntervalBCELoss, StartEndBCELoss
from bert_based.predict_utils import kmeans_region_detection, visualize_intervals, visualize_random_subsamples, visualize_baseline
from eval.rn_IoUm import r1_IoUm_sum, dr1_IoUm_sum

# Fixed Info
embedding_dir = "data/clip_video_feature_vector"

# Model Info
prediction_head = "in_out"
predictor_fn = "brute_force"
alpha = 0.5
num_hidden = 0
frame_rate = 4

# Run info
split = "iid"
dataset = f"data/Charades-CD/charades_test_{split}.json"

is_baseline = False
checkpoint_fname = "checkpoints/moment_bert_best.pt"
model_nickname = "moment_bert"
folder = "moment_bert/brute_force_ave_0.5_proportion"
if not os.path.exists(folder):
    os.makedirs(folder)

logits_out_fname = f"{folder}/logits_{model_nickname}_{split}.json"
bounds_out_fname = f"{folder}/bounds_{model_nickname}_{split}.json"
metrics_out_fname = f"{folder}/metrics_{model_nickname}_{split}.json"
visualizations_out_folder = f"{folder}/visualizations_{model_nickname}_{split}"
if not os.path.exists(visualizations_out_folder):
    os.makedirs(visualizations_out_folder)

#############################################################

def predict_from_in_out(model, queries, video_clip_embeddings, predictor_fn, len_pdf):
    # print(queries)
    result = model.forward(queries, video_clip_embeddings)
    result = torch.sigmoid(result)
    result = result.cpu().numpy()
    
    if predictor_fn == "argmax":
        pred_clip_length = 8 * frame_rate # placeholder
        max_framestamps = np.argmax(result, axis=1) # (batch_size, 1)
        num_frames = result.shape[1]

        lower_bound = np.maximum(0, max_framestamps - pred_clip_length//2)
        upper_bound = np.minimum(num_frames, max_framestamps + pred_clip_length//2)
        bounds = np.stack([lower_bound, upper_bound], axis=0).T # (2, batch_size)
    elif predictor_fn == "kmeans":
        # Use kmeans to find the regions
        spans = kmeans_region_detection(result)
        bounds = np.stack(spans, axis=1).T # (2, batch_size)
    elif predictor_fn == "brute_force":
        if len_pdf is None:
            raise ValueError("len_pdf must be provided for brute_force prediction.")
        log_result = np.log(result + 1e-6)
        bounds = []
        for batch_index in range(log_result.shape[0]):
            batch_item = log_result[batch_index]

            n = len(batch_item)
            cumsum = np.cumsum(batch_item)
            cumsum = np.insert(cumsum, 0, 0)  # for easier slicing

            best_score = float('-inf')
            best_start, best_end = 0, 0

            for start in range(n):
                for end in range(start, n):
                    span_sum = (cumsum[end + 1] - cumsum[start]) / (end - start + 1)
                    proportion = (end - start) 
                    length_score = np.log(len_pdf(proportion))
                    # print(length_score)
                    total_score = (1 - alpha) * span_sum + (alpha * length_score)
                    if total_score > best_score:
                        best_score = total_score
                        best_start, best_end = start, end
            bounds.append([best_start, best_end])
        bounds = np.array(bounds) # (2, batch_size)

    return result.tolist(), bounds

def predict_from_start_end(model, queries, video_clip_embeddings, predictor_fn, len_pdf):
    start_result, end_result = model.forward(queries, video_clip_embeddings)
    start_result = torch.sigmoid(start_result).cpu().numpy()
    end_result = torch.sigmoid(end_result).cpu().numpy()
    # print(start_result)
    # start_result = start_result.cpu().numpy()
    # end_result = end_result.cpu().numpy()

    if predictor_fn == "start argmax":
        start_framestamps = np.argmax(start_result, axis=1)

        # mask any area where start_framestamps >= end_framestamps
        mask = np.zeros_like(end_result)
        for i in range(end_result.shape[0]):
            mask[i, start_framestamps[i]:] = 1

        end_result = end_result * mask
        end_framestamps = np.argmax(end_result, axis=1)

        bounds = np.stack([start_framestamps, end_framestamps], axis=0).T # (batch_size, 2)
    
    return [start_result.tolist(), end_result.tolist()], bounds

def predict(model, test_dataset, device, ious=[0.1, 0.3, 0.5, 0.7, 0.9], predictor_fn="argmax", len_pdf=None):
    """
    Prediction loop for video moment retrieval.

    model: MomentBERT
    dataloader (MomentDataset): yields dict with tokenized text, video embeddings, labels
    device: CUDA or CPU
    """
    model.eval()
    predictions = {}
    predicted_bounds = {}
    
    total_sum_r1_iou = {iou: 0 for iou in ious}
    total_sum_dr1_iou = {iou: 0 for iou in ious}
    total_queries = 0

    with torch.no_grad():
        for i,data in tqdm(enumerate(test_dataset)):
            
            key = data['video_id']
            queries = data['sentences']
            num_frames = data['video_duration'] * frame_rate

            video_clip_embeddings = torch.from_numpy(data['embedding']).to(device)
            
            if video_clip_embeddings.shape[0] > model.max_video_len:
                continue

            if model.prediction_head == 'in_out':
                preds, bounds = predict_from_in_out(model, queries, video_clip_embeddings, predictor_fn, len_pdf)
            elif model.prediction_head == 'start_end':
                preds, bounds = predict_from_start_end(model, queries, video_clip_embeddings, predictor_fn, len_pdf)
            
            predicted_bounds[key] = bounds.tolist()
            predictions[key] = preds

            gt_intervals = np.array(data['framestamps'])
            pred_intervals = bounds
            
            total_queries += gt_intervals.shape[0]
            for iou in ious:
                total_sum_r1_iou[iou] += np.sum(r1_IoUm_sum(gt_intervals, pred_intervals, iou))
                total_sum_dr1_iou[iou] += np.sum(dr1_IoUm_sum(gt_intervals, pred_intervals, num_frames, iou))
            
    results = {'R1': {}, 'dR1': {}}
    for iou in ious:
        total_sum_r1_iou[iou] /= total_queries
        results['R1'][iou] = {
            'R1 IoU': total_sum_r1_iou[iou],
        }
        print(f"R1 IoU@{iou}: {total_sum_r1_iou[iou]:.4f}")

    for iou in ious:
        total_sum_dr1_iou[iou] /= total_queries
        results['dR1'][iou] = {
            'dR1 IoU': total_sum_dr1_iou[iou],
        }
        print(f"dR1 IoU@{iou}: {total_sum_dr1_iou[iou]:.4f}")


    with open(metrics_out_fname, "w") as f:
        # json.dump(output, f)
        json.dump(results, f)
        print(f"Results saved to {metrics_out_fname}")

    with open(logits_out_fname, "w") as f:
        # json.dump(output, f)
        json.dump(predictions, f)
        print(f"Predictions saved to {logits_out_fname}")

    with open(bounds_out_fname, "w") as f:
        # json.dump(output, f)
        json.dump(predicted_bounds, f)
        print(f"Predictions saved to {bounds_out_fname}")


if __name__ == "__main__":
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    if not os.path.exists(bounds_out_fname):
        # Run the predictions
        model = MomentBERT(
            prediction_head=prediction_head,
            num_hidden=num_hidden,
        ).to(device)
        checkpoint = torch.load(checkpoint_fname)
        model.load_state_dict(checkpoint['model_state_dict'])

        len_pdf_params, proportion_pdf_params = get_prob_interval_len(
            frame_rate=frame_rate,
            make_plot=False
        )

        len_pdf = lambda x: gamma.pdf(x, *len_pdf_params)

        test_dataset = MomentDataset(
            dataset_json_file=dataset,
            embedding_dir=embedding_dir,
            frame_rate=frame_rate
        )

        predict(model, test_dataset, device, predictor_fn=predictor_fn, len_pdf=len_pdf)
    else:
        print(f"Predictions already exist at {bounds_out_fname}. Skipping prediction step.")

    visualizer = visualize_baseline if is_baseline else visualize_random_subsamples
    visualizer(
        source=dataset,
        predictions=logits_out_fname,
        bounds_predictions=bounds_out_fname,
        run_label=model_nickname,
        split_label=split,
        num_sample=5,
        frame_rate=frame_rate,
        out_folder=visualizations_out_folder
    )