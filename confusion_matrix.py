import json
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load prediksi
with open("predictions.json", "r") as f:
    data = json.load(f)

y_true = np.array(data["y_true"])
y_pred = np.array(data["y_pred"])

# Confusion Matrix
target_names = ["Negatif", "Netral", "Positif"]
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix IndoBERT")
plt.show()
plt.savefig("confusion_matrix_from_json.png")
