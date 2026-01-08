# =========================================
# FULL cDTM PIPELINE FROM JSON FILE
# =========================================

import json
import re
import pandas as pd
import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models import LdaSeqModel
from gensim.models import LdaModel
import warnings
from datetime import datetime

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

start_time = datetime.now()
print("Script started at:", start_time)

factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())

DOMAIN_STOPWORDS = set([
    "min", "admin", "mimin", "kak", "pak", "bu", "bapak", "ibu",
    "tolong", "mohon", "izin", "nanya", "tanya", "jawab",
    "yg", "aja", "dong", "sih", "deh", "nih", "tuh",
    "ga", "gak", "nggak", "tidak", "nya", "lah", "kah", "pun", "the",
    "semoga", "selalu", "udah", "jadi", "mau", "tahun", "lolos", "anak",
    "lulus", "batch", "gimana", "kasih", "email", "banyak", "kerja", "sukses", "semangat",
    "seleksi", "program", "jam", "tim", "kurang", "apa", "gram", "surat", "cross", "mana",
    "salah","selamat","sama","lebih","bulan","hari","tersebut","satu","sekali","kapan",
    "kok","bagaimana","siang","cara","siap","berapa","semua","dulu","benar","piring"
])

def remove_domain_stopwords(tokens):
    return [
        t for t in tokens
        if t not in DOMAIN_STOPWORDS and len(t) > 2
    ]

STOPWORDS_FINAL = set(stopwords).union(DOMAIN_STOPWORDS)

def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()

def assign_topic(tokens, topic_labels, time_slice):
    labels_this_slice = topic_labels.get(time_slice, {})
    scores = {}

    for topic_id, label in labels_this_slice.items():
        keywords = label.split("_")
        score = sum(1 for k in keywords if k in tokens)
        scores[topic_id] = score

    # jika semua skor 0 → noise
    if max(scores.values(), default=0) == 0:
        return -1  # unknown / noise

    return max(scores, key=scores.get)


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="gensim.models.ldaseqmodel"
)

# =========================================
# 1. LOAD DATA JSON
# =========================================

JSON_FILE = "hasil_prediksi_pakai.json"   # GANTI dengan nama file kamu

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# =========================================
# 2. KONVERSI TIMESTAMP → DATETIME
# =========================================
# Jika timestamp MILIDETIK (umumnya IG, API)
df["created_at_dt"] = pd.to_datetime(df["created_at"], unit="s")

# Urutkan waktu (WAJIB UNTUK cDTM)
df = df.sort_values("created_at_dt").reset_index(drop=True)

# =========================================
# 3. TIME SLICING (BULANAN)
# =========================================
df["time_slice"] = df["created_at_dt"].dt.to_period("M").astype(str)

time_map = {p: i for i, p in enumerate(sorted(df["time_slice"].unique()))}
df["time_slice_id"] = df["time_slice"].map(time_map)

df["clean_text"] = df["text"]

# =========================================
# 5. TOKENISASI
# =========================================
texts = [t.split() for t in df["clean_text"]]

texts_cleaned = []
valid_index = []

for i, text in enumerate(df["clean_text"]):
    tokens = simple_tokenize(text)
    tokens = [
        w for w in tokens
        if w not in STOPWORDS_FINAL
        and len(w) > 2
        and w.isalpha()
    ]

    if len(tokens) > 0:
        texts_cleaned.append(tokens)
        valid_index.append(i)

# FILTER df AGAR SINKRON
df = df.loc[valid_index].reset_index(drop=True)

# =========================================
# 6. DICTIONARY & CORPUS (BoW)
# =========================================
dictionary = Dictionary(texts_cleaned)

dictionary.filter_extremes(
    no_below=5,   # kata muncul minimal 10 dok
    no_above=0.5  # kata terlalu umum dibuang
)

corpus = [dictionary.doc2bow(text) for text in texts_cleaned]

print("Jumlah dokumen :", len(corpus))
print("Jumlah kata    :", len(dictionary))

# =========================================
# 7. TIME SLICES UNTUK cDTM
# =========================================
time_slices = df.groupby("time_slice").size().tolist()

assert sum(time_slices) == len(corpus)

print("Jumlah time slice :", len(time_slices))
print("Time slices       :", time_slices)

# =========================================
# 8. TRAINING cDTM
# =========================================
NUM_TOPICS = 7  # rekomendasi tesis: 5–7

base_lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    passes=3,
    iterations=50,
    alpha='auto',
    eta='auto',
    random_state=42
)

ldaseq = LdaSeqModel(
    corpus=corpus,
    time_slice=time_slices,
    num_topics=NUM_TOPICS,
    id2word=dictionary,
    initialize='ldamodel',
    lda_model=base_lda,
    passes=1   # PENTING
)

dominant_topics = []
topic_labels = {}

for doc_id, ts in enumerate(df["time_slice_id"].values):
    topic_dist = ldaseq.doc_topics(doc_id)
    dominant_topics.append(int(topic_dist.argmax()))

df["dominant_topic"] = dominant_topics

topic_labels = {}

for t in range(len(time_slices)):
    topic_labels[t] = {}

    for topic_id in range(NUM_TOPICS):
        topic_terms = ldaseq.print_topic(
            topic=topic_id,
            time=t,
            top_terms=5
        )

        # Jika bentuknya list of tuples → AMAN
        if isinstance(topic_terms, list):
            words = [w for w, _ in topic_terms[:2]]
        else:
            words = []

        label = "_".join(words) if words else "unknown"
        topic_labels[t][topic_id] = label


df["dominant_topic_label"] = [
    topic_labels[ts].get(tp, "unknown")
    for ts, tp in zip(df["time_slice_id"], df["dominant_topic"])
]

OUTPUT_JSON = "hasil_cdtm_dengan_sentimen.json"

# pilih kolom yang mau disimpan
output_columns = [
    "username",
    "text",
    "created_at",
    "created_at_dt",
    "post_url",
    "sentiment",
    "sentiment_id",
    "confidence",
    "probabilities",
    "time_slice",
    "time_slice_id",
    "dominant_topic"
]

output_data = df[output_columns].to_dict(orient="records")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

end_time = datetime.now()
duration = end_time - start_time
print(f"Script finished at : {end_time}")
print(f"Durasi eksekusi : {duration}")