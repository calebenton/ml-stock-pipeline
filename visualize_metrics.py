# visualize_metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set_theme(style='darkgrid')

# Load model metrics
metrics_path = "metrics/model_scores.csv"
df = pd.read_csv(metrics_path)

# Set plot style
#plt.style.use("seaborn-darkgrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ["RMSE", "MAE", "R2"]

for i, metric in enumerate(metrics):
    axes[i].bar(df["Model"], df[metric], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[i].set_title(f"{metric} by Model", fontsize=14)
    axes[i].set_ylabel(metric)
    axes[i].set_xlabel("Model")

plt.tight_layout()
plt.savefig("metrics/model_scores_plot.png")
plt.show()
