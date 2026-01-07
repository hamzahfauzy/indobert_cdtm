import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# =============================
# CONFIG
# =============================
MODEL_NAME = "indobenchmark/indobert-base-p1"
DATASET_PATH = "dataset_labeled.json"
NUM_LABELS = 3
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =============================
# LOAD DATASET JSON
# =============================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

# train-test split (WAJIB buat evaluasi)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# =============================
# TOKENIZER
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

train_encodings = tokenize(X_train)
test_encodings = tokenize(X_test)

# =============================
# DATASET CLASS
# =============================
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, y_train)
test_dataset = SentimentDataset(test_encodings, y_test)

# =============================
# MODEL
# =============================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# =============================
# METRIC FUNCTION
# =============================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# =============================
# TRAINING ARGUMENTS
# =============================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available()  # 🔥 optional, percepat training
)

# =============================
# TRAINER
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# =============================
# TRAIN
# =============================
trainer.train()

# =============================
# EVALUATION
# =============================
eval_result = trainer.evaluate()
print("\nEvaluation Result:")
for k, v in eval_result.items():
    print(f"{k}: {v:.4f}")

# =============================
# SAVE MODEL
# =============================
model.save_pretrained("./model_sentiment_indobert")
tokenizer.save_pretrained("./model_sentiment_indobert")

print("\nModel saved to ./model_sentiment_indobert")
