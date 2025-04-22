import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_region_detection(logits, n_clusters=2):
    # Handle batched input: logits shape is (batch_size, len)
    batch_size = logits.shape[0]
    all_spans = []
    
    for batch_idx in range(batch_size):
        # Get logits for current batch
        X = logits[batch_idx].reshape(-1, 1)
        
        # Cluster the logits into regions (e.g., high vs low)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        labels = kmeans.labels_
        
        # Find the cluster with highest centroid (the "positive" region)
        positive_cluster = np.argmax(kmeans.cluster_centers_)
        
        # Get continuous regions
        spans = []
        in_span = False
        start_idx = None
        
        for i, label in enumerate(labels):
            if label == positive_cluster and not in_span:
                # Start of a span
                start_idx = i
                in_span = True
            elif label != positive_cluster and in_span:
                # End of a span
                spans.append([start_idx, i-1])
                in_span = False
        
        # Handle case where span extends to the end
        if in_span:
            spans.append([start_idx, len(labels)-1])
            
        max_span = max(spans, key=lambda x:np.mean(logits[batch_idx][x[0]:x[1]+1]))
        all_spans.append(max_span)
    
    return all_spans

def visualize_intervals(true_interval, predicted_logits, video_id, run_label, query_id):
    # Create a figure
    plt.figure(figsize=(10, 5))
    
    # Plot true intervals
    # plt.plot(interval, [1, 1], color='green', linewidth=5, label='True Interval')
    plt.axvspan(true_interval[0], true_interval[1], color='green', alpha=0.5, label='True Interval')
    
    # Plot predicted intervals
    plt.plot(predicted_logits, color='red', label='Predicted Interval')
    
    # Set title and labels
    plt.title(f'Video ID: {video_id}')
    plt.xlabel('Frame Index')
    plt.ylim(0, 1)
    plt.ylabel('Predicted Probability')
    
    # Show legend
    plt.legend()
    
    # save the figure
    plt.savefig(f'visualizations/{run_label}_{video_id}_{query_id}.png')
    plt.close()
    
def visualize_random_subsamples(source, predictions, num_sample=5, frame_rate=16):
    with open(source, 'r') as f:
        source_json = json.load(f)

    with open(predictions, 'r') as f:
        pred_json = json.load(f)

    # get random keys
    keys = list(pred_json.keys())
    random_keys = np.random.choice(keys, num_sample, replace=False)
    for key in random_keys:
        print(key)
        queries = source_json[key]['sentences']
        framestamps = source_json[key]['framestamps']
        print(framestamps)
        if frame_rate < 16:
            downsample_ratio = 16 // frame_rate
            framestamps = [ 
                [
                    round(start / downsample_ratio), 
                    round(end / downsample_ratio)
                ]
                for start, end in framestamps
            ]

        logits = np.array(pred_json[key])

        for i, q in enumerate(queries):
            # get the true interval
            true_interval = framestamps[i]
            print(true_interval)
            # get the predicted logits
            predicted_logits = logits[i]
            print(predicted_logits)

            # get the video id
            video_id = key
            # get the query id
            query_id = i

            # visualize the intervals
            visualize_intervals(true_interval, predicted_logits, video_id, 'random_subsample', query_id)
        
