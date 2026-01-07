from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "indolem/indobertweet-base-uncased"
)

token = tokenizer("pelayanan sangat bagus")

print(f"{token}")