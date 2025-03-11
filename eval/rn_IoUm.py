import numpy as np
import json
import sys

def iou(gt_interval, pred_interval):
    intersection = np.minimum(gt_interval[:, 1], pred_interval[:, 1]) - np.maximum(gt_interval[:, 0], pred_interval[:, 0])
    intersection = np.maximum(intersection, 0)
    union = np.maximum(gt_interval[:, 1], pred_interval[:, 1]) - np.minimum(gt_interval[:, 0], pred_interval[:, 0])
    return intersection / union

def r1_IoUm_sum(gt_interval, pred_interval, min_IoU=0.5):
    IoU = iou(gt_interval, pred_interval)
    return IoU >= min_IoU

def r1_IoUm(gt_interval_path, prediction_interval_path, min_IoU=0.5):
    # Load both gt and pred as json files
    with open(gt_interval_path, 'r') as file:
        gt_data = json.load(file)
    with open(prediction_interval_path, 'r') as file:
        pred_data = json.load(file)

    total_queries = 0
    total_sum = 0
    for vid_id in gt_data:
        gt_intervals = np.array(gt_data[vid_id]['framestamps'])
        pred_intervals = np.array(pred_data[vid_id])
        
        total_queries += gt_intervals.shape[0]
        total_sum += np.sum(r1_IoUm_sum(gt_intervals, pred_intervals, min_IoU))
    
    return total_sum / total_queries

def dr1_IoUm(gt_interval_path, prediction_interval_path, min_IoU=0.5):
    # Load both gt and pred as json files
    with open(gt_interval_path, 'r') as file:
        gt_data = json.load(file)
    with open(prediction_interval_path, 'r') as file:
        pred_data = json.load(file)

    total_queries = 0
    total_sum = 0
    for vid_id in gt_data:
        gt_intervals = np.array(gt_data[vid_id]['framestamps'])
        pred_intervals = np.array(pred_data[vid_id])
        
        total_queries += gt_intervals.shape[0]
        r1_IoUm_res = r1_IoUm_sum(gt_intervals, pred_intervals, min_IoU)
        
        vid_size = gt_data[vid_id]['video_duration'] * gt_data[vid_id]['decode_fps']
        alpha_s = np.abs(gt_intervals[:, 0] - pred_intervals[:, 0])/vid_size
        alpha_e = np.abs(gt_intervals[:, 1] - pred_intervals[:, 1])/vid_size
        
        total_sum += np.sum(r1_IoUm_res * (1 - alpha_s) * (1 - alpha_e))
    return total_sum / total_queries

if __name__ == "__main__":
    # Get the ground truth interval path and the prediction interval path from the command line
    gt_interval_path = sys.argv[1]
    prediction_interval_path = sys.argv[2]
    result_file_path = sys.argv[3]
    m = [float(iou_m) for iou_m in sys.argv[4:]]
    
    results_r1_iou_m = {}
    for iou_m in m:
        results_r1_iou_m[iou_m] = r1_IoUm(gt_interval_path, prediction_interval_path, iou_m)
        
    results_dr1_iou_m = {}
    for iou_m in m:
        results_dr1_iou_m[iou_m] = dr1_IoUm(gt_interval_path, prediction_interval_path, iou_m)
    
    results = {'R1_IoUm': results_r1_iou_m, 'DR1_IoUm': results_dr1_iou_m}
    with open(result_file_path, 'w') as file:
        json.dump(results, file)
    
    