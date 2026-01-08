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

cdtm_result = {}

for time in range(len(time_slices)):
    cdtm_result[time] = {}

    topics = ldaseq.print_topics(time=time, top_terms=10)

    for idx, topic in enumerate(topics):
        terms = []

        # topic = (('sppi', weight), ('info', weight))
        for word, weight in topic:
            terms.append({
                "word": word,
                "weight": float(weight)
            })

        # pakai index sebagai topic_id
        cdtm_result[time][idx] = {
            "terms": terms
        }

# simpan
with open("cdtm_topics.json", "w", encoding="utf-8") as f:
    json.dump(cdtm_result, f, ensure_ascii=False, indent=2)

end_time = datetime.now()
duration = end_time - start_time
print(f"Script finished at : {end_time}")
print(f"Durasi eksekusi : {duration}")