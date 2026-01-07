# =========================================
# FULL cDTM PIPELINE FROM JSON FILE
# =========================================

import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models import LdaSeqModel
from gensim.models import LdaModel
from wordcloud import WordCloud

import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="gensim.models.ldaseqmodel"
)

# =========================================
# 1. LOAD DATA JSON
# =========================================
# Contoh struktur JSON:
# [
#   {"text": "harga mahal banget", "created_at": 1704067200000},
#   ...
# ]

JSON_FILE = "hasil_prediksi_1.json"   # GANTI dengan nama file kamu

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
df["time_slice"] = df["created_at_dt"].dt.to_period("M")

df["clean_text"] = df["text"]

# =========================================
# 5. TOKENISASI
# =========================================
texts = [t.split() for t in df["clean_text"]]

# =========================================
# 6. DICTIONARY & CORPUS (BoW)
# =========================================
dictionary = Dictionary(texts)

dictionary.filter_extremes(
    no_below=10,   # kata muncul minimal 10 dok
    no_above=0.4  # kata terlalu umum dibuang
)

corpus = [dictionary.doc2bow(text) for text in texts]

print("Jumlah dokumen :", len(corpus))
print("Jumlah kata    :", len(dictionary))

# =========================================
# 7. TIME SLICES UNTUK cDTM
# =========================================
time_slices = df.groupby("time_slice").size().tolist()

assert sum(time_slices) == len(corpus)

MIN_DOCS = 100

new_time_slice = []
buffer = 0

for ts in time_slices:
    if ts < MIN_DOCS:
        buffer += ts
    else:
        if buffer > 0:
            new_time_slice.append(buffer + ts)
            buffer = 0
        else:
            new_time_slice.append(ts)

if buffer > 0:
    new_time_slice[-1] += buffer

time_slices = new_time_slice

print("Jumlah time slice :", len(time_slices))
print("Time slices       :", time_slices)

# =========================================
# 8. TRAINING cDTM
# =========================================
NUM_TOPICS = 7  # rekomendasi tesis: 5–7

base_lda = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=3,
    passes=3,
    iterations=50,
    alpha='auto',
    eta='auto',
    random_state=42
)

ldaseq = LdaSeqModel(
    corpus=corpus,
    time_slice=time_slices,
    num_topics=3,
    id2word=dictionary,
    initialize='ldamodel',
    lda_model=base_lda,
    passes=1   # PENTING
)

# for t in range(ldaseq.num_topics):
#     print(f"\nTopic {t}")
#     for time in range(len(time_slices)):
#         print(ldaseq.print_topic(topic=t, time=time, top_terms=5))

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

# =========================================
# 10. TRACKING PERGESERAN KATA
# =========================================
def track_word(model, word, topic_id, time_slices):
    trend = []

    for t in range(len(time_slices)):
        topic_str = model.print_topic(topic_id, time=t, top_terms=50)

        # Ambil pasangan (prob, word)
        pairs = re.findall(r'([\d\.]+)\*"([^"]+)"', topic_str)

        word_prob = 0.0
        for prob, w in pairs:
            if w == word:
                word_prob = float(prob)
                break

        trend.append(word_prob)

    return trend

# Contoh tracking kata
WORD_TO_TRACK = "harga"
TOPIC_ID = 0

trend = track_word(ldaseq, WORD_TO_TRACK, TOPIC_ID, time_slices)

# =========================================
# 11. VISUALISASI LINE CHART
# =========================================
plt.figure()
plt.plot(trend, marker="o")
plt.xlabel("Time Slice")
plt.ylabel("Bobot Kata")
plt.title(f'Pergeseran Kata "{WORD_TO_TRACK}" pada Topik {TOPIC_ID}')
plt.grid(True)
plt.show()

# =========================================
# 12. WORDCLOUD TOPIK PER WAKTU
# =========================================
def plot_wordcloud(model, dictionary, topic_id, time):
    words = {
        dictionary[word_id]: weight
        for word_id, weight in model.get_topic_terms(
            topic_id, time=time, topn=30
        )
    }

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(words)

    plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    plt.title(f"Topik {topic_id} - Time Slice {time}")
    plt.show()

# WordCloud awal & akhir
plot_wordcloud(ldaseq, dictionary, topic_id=0, time=0)
plot_wordcloud(ldaseq, dictionary, topic_id=0, time=len(time_slices) - 1)

# =========================================
# 13. HEATMAP TOPIK vs WAKTU
# =========================================
topic_strength = np.zeros((NUM_TOPICS, len(time_slices)))

for topic_id in range(NUM_TOPICS):
    for t in range(len(time_slices)):
        topic_strength[topic_id, t] = sum(
            weight for _, weight in
            ldaseq.get_topic_terms(topic_id, time=t)
        )

plt.figure()
plt.imshow(topic_strength)
plt.colorbar()
plt.xlabel("Time Slice")
plt.ylabel("Topik")
plt.title("Distribusi Topik terhadap Waktu")
plt.show()

# =========================================
# END OF SCRIPT
# =========================================
