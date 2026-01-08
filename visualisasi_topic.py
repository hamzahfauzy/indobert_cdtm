import json
import matplotlib.pyplot as plt

with open("cdtm_topics_old.json", "r", encoding="utf-8") as f:
    data = json.load(f)

time_slices = sorted(data.keys(), key=int)

words_to_track = ["makan", "gizi", "mbg", "sppi", "bgn"]

word_trend = {w: [] for w in words_to_track}

for t in time_slices:
    all_terms = []
    for topic_data in data[t].values():
        all_terms.extend(topic_data["terms"])

    for word in words_to_track:
        weight = sum(
            term["weight"] for term in all_terms if term["word"] == word
        )
        word_trend[word].append(weight)

    print(f"\n=== Time Slice {t} ===")
    for topic_id, topic_data in data[t].items():
        words = [term["word"] for term in topic_data["terms"][:5]]
        print(f"Topic {topic_id}: {', '.join(words)}")

plt.figure(figsize=(10, 6))
for word, weights in word_trend.items():
    plt.plot(time_slices, weights, marker="o", label=word)

plt.xlabel("Time Slice")
plt.ylabel("Word Weight")
plt.title("Keyword Trend Over Time")
plt.legend()
plt.grid(True)
plt.show()
