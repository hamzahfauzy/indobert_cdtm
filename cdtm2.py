# =========================================
# FULL cDTM PIPELINE FROM JSON FILE
# =========================================

import json
import re
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models import LdaSeqModel
from gensim.models import LdaModel
# from wordcloud import WordCloud
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
    "seleksi", "program", "jam", "tim", "kurang", "apa", "gram"
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
# df["time_slice"] = df["created_at_dt"].dt.to_period("2W-MON")
df["time_slice"] = df["created_at_dt"].dt.to_period("M")

df["clean_text"] = df["text"]

# =========================================
# 5. TOKENISASI
# =========================================
texts = [t.split() for t in df["clean_text"]]

texts_cleaned = [
    [
        w for w in doc
        if w not in STOPWORDS_FINAL
        and len(w) > 2
        and w.isalpha()
    ]
    for doc in texts
]

# Safety: buang dokumen kosong
texts_cleaned = [doc for doc in texts_cleaned if len(doc) > 0]

# =========================================
# 6. DICTIONARY & CORPUS (BoW)
# =========================================
dictionary = Dictionary(texts_cleaned)

dictionary.filter_extremes(
    no_below=5,   # kata muncul minimal 10 dok
    no_above=0.5  # kata terlalu umum dibuang
)

corpus = [dictionary.doc2bow(text) for text in texts]

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

# =========================================
# 9. CETAK TOPIK PER WAKTU
# =========================================
for time in range(len(time_slices)):
    print(f"\n===== TIME SLICE {time} =====")
    topics = ldaseq.print_topics(time=time, top_terms=10)

    for topic in topics:
        topic_id = topic[0]
        topic_terms = topic[1]
        print(f"Topic {topic_id}: {topic_terms}")


def map_topic_labels_dynamic(model, time_slices, top_terms=5):
    """
    Label topik per time slice untuk melihat pergeseran narasi
    """
    labels = {}

    for t in range(len(time_slices)):
        labels[t] = {}
        for topic_id in range(model.num_topics):
            terms = model.print_topic(topic_id, time=t, top_terms=top_terms)
            words = [w for w, _ in terms]
            labels[t][topic_id] = "_".join(words[:2])

    return labels

def track_words(model, words, time_slices):
    """
    Track probabilitas kata per time slice di semua topik.
    words: list of words yang ingin di-track
    return: dict {word: [list probabilitas per slice]}
    """
    trend_dict = {w: [] for w in words}

    for t in range(len(time_slices)):
        topics = model.print_topics(time=t, top_terms=50)

        # Buat dict kata -> total probabilitas di semua topik
        word_prob = {w:0.0 for w in words}

        for top in topics:
            # top = ((kata1, prob1), (kata2, prob2), ...)
            for pair in top:
                w, p = pair
                if w in words:
                    word_prob[w] += p

        # Append hasil tiap slice
        for w in words:
            trend_dict[w].append(word_prob[w])

    return trend_dict

def plot_trends(trend_dict):
    """
    Plot tren kata per time slice
    """
    plt.figure(figsize=(10,6))
    for word, trend in trend_dict.items():
        plt.plot(trend, marker='o', label=word)
    plt.xlabel("Time Slice")
    plt.ylabel("Probabilitas")
    plt.title("Tren Kata per Time Slice di cDTM")
    plt.legend()
    plt.show()


# 1. Buat label topik otomatis (opsional, untuk interpretasi)
topic_labels = map_topic_labels_dynamic(ldaseq, time_slices)
print(topic_labels)

end_time = datetime.now()
duration = end_time - start_time
print(f"Script finished at : {end_time}")
print(f"Durasi eksekusi : {duration}")
# contoh output: {0: 'di_tidak', 1: 'min_tidak', 2: 'kami_untuk'}

# 2. Track kata yang ingin dianalisis
# words_to_track = ["makan", "program", "mbg", "dapur", "gizi", "sppi", "lulus", "lolos", "bgn"]
# trend_dict = track_words(ldaseq, words_to_track, time_slices)

# 3. Plot hasil
# plot_trends(trend_dict)