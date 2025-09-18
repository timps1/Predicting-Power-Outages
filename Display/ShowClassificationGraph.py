import pandas as pd
import matplotlib.pyplot as plt
import sys
import re

if len(sys.argv) != 2:
    print("Usage: python ShowClassificationGraph.py <path_to_csv>")
    sys.exit(1)

# Load CSV
df = pd.read_csv(sys.argv[1])

# Extract numeric C value from Model string
df["C"] = df["Model"].apply(lambda x: float(re.search(r"C_(\d+(\.\d+)?)", x).group(1)))

# Extract model type (everything before (C_x))
df["ModelType"] = df["Model"].apply(lambda x: re.sub(r"\(C_\d+\)", "", x))

# Get unique model type(s)
model_type = df["ModelType"].unique()[0]  # assuming one model type per file

# Split into the 5 groups
zero_df = df.iloc[0::5].reset_index(drop=True)      # class 0
one_df = df.iloc[1::5].reset_index(drop=True)       # class 1
accuracy_df = df.iloc[2::5].reset_index(drop=True)  # accuracy
macro_df = df.iloc[3::5].reset_index(drop=True)     # macro avg
weighted_df = df.iloc[4::5].reset_index(drop=True)  # weighted avg

# Create figure with 3x2 grid, last cell unused
fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
axes = axes.flatten()

def plot_metrics(ax, df, title):
    ax.plot(df["C"], df["precision"], marker="o", label="Precision")
    ax.plot(df["C"], df["recall"], marker="o", label="Recall")
    ax.plot(df["C"], df["f1-score"], marker="o", label="F1-score")
    ax.set_title(title)
    ax.set_xlabel("C Parameter")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

def plot_metrics_accuracy(ax, df, title):
    ax.plot(df["C"], df["f1-score"], marker="o", label="F1-score")
    ax.set_title(title)
    ax.set_xlabel("C Parameter")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()

# Plot each group
plot_metrics(axes[0], zero_df, "Class 0")
plot_metrics(axes[1], one_df, "Class 1")
plot_metrics(axes[2], accuracy_df, "Accuracy")
plot_metrics(axes[3], macro_df, "Macro Avg")
plot_metrics(axes[4], weighted_df, "Weighted Avg")

# Hide the last empty subplot (bottom-right)
fig.delaxes(axes[5])

# Now set automatic suptitle
fig.suptitle(f"Model: {model_type}", fontsize=16, y=0.99)

plt.tight_layout()
plt.show()
