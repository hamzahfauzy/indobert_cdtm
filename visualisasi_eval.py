import json
import matplotlib.pyplot as plt

# Load eval_result.json
with open("eval_result.json", "r") as f:
    eval_result = json.load(f)

# Pilih metrik yang ingin divisualisasi
metrics = ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]
values = [eval_result[m] for m in metrics]

# Plot bar chart
plt.figure(figsize=(6,4))
plt.bar(metrics, values, color=["skyblue","orange","green","red"])
plt.ylim(0,1)
plt.title(f"Model Evaluation Metrics (Epoch {eval_result['epoch']})")
for i, v in enumerate(values):
    plt.text(i, v+0.01, f"{v:.3f}", ha='center', fontsize=10)
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("eval_metrics.png")
plt.show()
