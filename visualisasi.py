import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

with open("hasil_cdtm_dengan_sentimen.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# pastikan time_slice urut
df["time_slice"] = pd.to_datetime(df["time_slice"])
df = df.sort_values("time_slice")

topic_trend = (
    df
    .groupby(["time_slice", "dominant_topic_label"])
    .size()
    .unstack(fill_value=0)
)

topics = topic_trend.columns.tolist()

colors = cm.get_cmap("tab20", len(topics))
topic_colors = {
    topic: colors(i)
    for i, topic in enumerate(topics)
}

plt.figure(figsize=(14, 7))
for topic in topic_trend.columns:
    plt.plot(topic_trend.index, topic_trend[topic], marker="o", label=topic, color=topic_colors[topic])

plt.title("Evolusi Topik dari Waktu ke Waktu")
plt.xlabel("Waktu")
plt.ylabel("Jumlah Dokumen")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sent_neg_trend = (
    df
    .groupby(["time_slice", "dominant_topic_label"])["probabilities"]
    .apply(lambda x: sum(p["negatif"] for p in x) / len(x))
    .unstack(fill_value=0)
)

plt.figure(figsize=(14, 7))
for topic in sent_neg_trend.columns:
    plt.plot(sent_neg_trend.index, sent_neg_trend[topic], marker="o", label=topic, color=topic_colors[topic])

plt.title("Perubahan Sentimen Negatif per Topik dari Waktu ke Waktu")
plt.xlabel("Waktu")
plt.ylabel("Rata-rata Probabilitas Sentimen Negatif")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sent_positif_trend = (
    df
    .groupby(["time_slice", "dominant_topic_label"])["probabilities"]
    .apply(lambda x: sum(p["positif"] for p in x) / len(x))
    .unstack(fill_value=0)
)

plt.figure(figsize=(14, 7))
for topic in sent_positif_trend.columns:
    plt.plot(sent_positif_trend.index, sent_positif_trend[topic], marker="o", label=topic, color=topic_colors[topic])

plt.title("Perubahan Sentimen Positif per Topik dari Waktu ke Waktu")
plt.xlabel("Waktu")
plt.ylabel("Rata-rata Probabilitas Sentimen Positif")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

topic_prop = topic_trend.div(topic_trend.sum(axis=1), axis=0)

TOP_N = topic_trend.columns.size # ideal: 5–7 untuk tesis

top_topics = (
    topic_prop.mean()
    .sort_values(ascending=False)
    .head(TOP_N)
    .index
)
topic_prop_top = topic_prop[top_topics]

# ================================
# 3. WARNA KONSISTEN
# ================================
colors = [topic_colors[topic] for topic in top_topics]

# ================================
# 4. PLOT AREA (SERAGAM)
# ================================
fig, ax = plt.subplots(figsize=(14, 7))

ax.stackplot(
    topic_prop_top.index,
    topic_prop_top.T.values,
    labels=top_topics,
    colors=colors,
    alpha=0.85
)

# ================================
# 5. STYLING (BIAR SAMA DENGAN YANG LAIN)
# ================================
ax.set_title("Pergeseran Proporsi Topik dari Waktu ke Waktu", fontsize=14)
ax.set_xlabel("Waktu")
ax.set_ylabel("Proporsi Topik")

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    frameon=False
)

plt.tight_layout()
plt.show()