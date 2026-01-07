import json
import re
import emoji
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

INPUT_FILE = "data_prediksi.json"
OUTPUT_FILE = "data_prediksi_bersih.json"

def is_contextual(text):
    if not text:
        return False

    text = text.lower().strip()

    # 1. Terlalu pendek
    if len(text.split()) < 3:
        return False

    # 2. Hanya emoji
    if emoji.replace_emoji(text, replace='') == '':
        return False

    # 3. Hanya mention / hashtag
    if re.fullmatch(r'(@\w+|\#\w+|\s)+', text):
        return False

    # 4. Spam umum
    spam_keywords = [
        'follow', 'follow me', 'dm', 'promo',
        'cek bio', 'link bio', 'subscribe', 'fresh maggot', 'wiki milky'
    ]
    if any(k in text for k in spam_keywords):
        return False

    # 5. Repetisi karakter (wkwkwkw, hahahaha)
    if re.search(r'(.)\1{3,}', text):
        return False

    # 6. Hanya stopword
    tokens = re.findall(r'\b\w+\b', text)
    meaningful = [w for w in tokens if w not in stop_words]
    if len(meaningful) < 2:
        return False

    return True


# Load JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter

# Filter
clean_data = []
for item in data:
    comment_text = item.get("text", "")
    if is_contextual(comment_text):
        item["is_contextual"] = True
        clean_data.append(item)

# Save output
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, ensure_ascii=False, indent=2)

print(f"Komentar awal   : {len(data)}")
print(f"Komentar bersih : {len(clean_data)}")
