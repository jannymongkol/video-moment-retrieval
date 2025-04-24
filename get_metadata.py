import json
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import gamma

# Create a histogram

split = "train"

def get_prob_interval_len(frame_rate=16, make_plot=False):
    with open(f"data/Charades-CD/charades_{split}.json", "r") as f:
        data = json.load(f)
    
    all_interval_lens = []
    all_interval_proportions = []

    for key in data:
        framestamps = data[key]['framestamps']
        downsample_rate = 16 // frame_rate

        video_duration = data[key]['video_duration']
        total_frames = (video_duration * 16) // frame_rate

        curr_interval_lens = [
            (f[1] - f[0]) // downsample_rate
            for f in framestamps
        ]

        curr_interval_proportions = [
            min(l / total_frames, 1) for l in curr_interval_lens
        ]

        for l in curr_interval_lens:
            if l > total_frames:
                print(key, framestamps, video_duration)

        all_interval_lens.extend(curr_interval_lens)
        all_interval_proportions.extend(curr_interval_proportions)

    len_counts, len_bin_edges = np.histogram(all_interval_lens, bins='auto', density=True)
    len_bin_centers = (len_bin_edges[:-1] + len_bin_edges[1:]) / 2
    len_params = gamma.fit(all_interval_lens)

    if make_plot:
        plt.plot(len_bin_centers, len_counts)
        plt.plot(np.arange(0, np.max(len_bin_centers), 1), 
                 gamma.pdf(np.arange(0, np.max(len_bin_centers), 1), *len_params), 
                 label='Fitted Gamma', color='red')
        plt.xlabel('Interval Length')
        plt.ylabel('Probability Density')
        plt.title('Histogram of Interval Lengths')
        # plt.show()
        plt.savefig(f'{split}_interval_length.png')
        plt.close()

    proportion_params = gamma.fit(all_interval_proportions)
    proportion_counts, proportion_bin_edges = np.histogram(all_interval_proportions, bins='auto', density=True)
    proportion_bin_centers = (proportion_bin_edges[:-1] + proportion_bin_edges[1:]) / 2
  
    if make_plot:
        plt.plot(proportion_bin_centers, proportion_counts)
        plt.plot(np.arange(0, 1, 0.01), 
                 gamma.pdf(np.arange(0, 1, 0.01), *proportion_params), 
                 label='Fitted Gamma', color='red')
        plt.xlabel('Interval Proportion')
        plt.ylabel('Probability Density')
        plt.title(f'Histogram of Interval Proportions ({split})')
        # plt.show()
        plt.savefig(f'{split}_interval_proportion.png')
        plt.close()
    return len_params, proportion_params

if __name__ == "__main__":
    params = get_prob_interval_len(4, make_plot=True)