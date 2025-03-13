import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "CTRL": {"iid": 12.8, "ood": 13.0},
    "ACRN": {"iid": 14.0, "ood": 13.0},
    "ABLR": {"iid": 25.0, "ood": 12.2},
    "SCDM": {"iid": 32.5, "ood": 23.8},
    "2D-TAN": {"iid": 26.2, "ood": 12.5},
    "DRN": {"iid": 28.3, "ood": 17.0},
    "TSP-PRL": {"iid": 18.0, "ood": 6.6},
    "WSSL": {"iid": 4.9, "ood": 9.5},
    "Our Baseline": {"iid": 20.7, "ood": 10.4}
}

# Extracting data
labels = list(data.keys())
iid_scores = [data[model]["iid"] for model in labels]
ood_scores = [data[model]["ood"] for model in labels]

# Bar graph parameters
x = np.arange(len(labels))  # label locations
width = 0.35  # bar width

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
bars_iid = ax.bar(x - width/2, iid_scores, width, label='IID', color='skyblue')
bars_ood = ax.bar(x + width/2, ood_scores, width, label='OOD', color='green')

# Add labels, title, and legend
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('R@1,IoU@0.7', fontsize=12)
ax.set_title('Model Performance on IID and OOD', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='center')
ax.legend()

for bar in bars_iid:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25, f'{bar.get_height():.1f}', 
            ha='center', va='bottom', fontsize=10)
for bar in bars_ood:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.25, f'{bar.get_height():.1f}', 
            ha='center', va='bottom', fontsize=10)

# Display the bar graph
plt.tight_layout()
plt.savefig('results/model_comparison.png')
plt.show()

