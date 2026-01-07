import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================
# CONFIG
# =============================
MODEL_PATH = "./model_sentiment_indobert"
INPUT_JSON = "data_prediksi_bersih.json"
OUTPUT_JSON = "hasil_prediksi.json"
MAX_LENGTH = 128

label_map = {
    0: "negatif",
    1: "netral",
    2: "positif"
}

# =============================
# LOAD MODEL
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =============================
# LOAD DATA JSON
# =============================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# =============================
# PREDICTION LOOP
# =============================
for item in tqdm(data, desc="Predicting"):
    text = item.get("text", "").strip()

    if not text:
        item["sentiment"] = None
        item["confidence"] = 0.0
        results.append(item)
        continue

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    label_id = torch.argmax(probs).item()

    # simpan hasil
    item["sentiment"] = label_map[label_id]
    item["sentiment_id"] = label_id
    item["confidence"] = round(probs[label_id].item(), 4)
    item["probabilities"] = {
        "negatif": round(probs[0].item(), 4),
        "netral": round(probs[1].item(), 4),
        "positif": round(probs[2].item(), 4)
    }

    results.append(item)

# =============================
# SAVE OUTPUT JSON
# =============================
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nHasil prediksi disimpan ke: {OUTPUT_JSON}")
