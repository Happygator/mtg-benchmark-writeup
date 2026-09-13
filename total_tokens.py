"""
This script downloads Scryfall's bulk data export and uses tiktoken to calculate the number of tokens all of them would require.
To use, run the following wherever you download this script:

pip install tiktoken
python total_tokens.py

"""
import gzip, json, statistics
import requests, tiktoken

HEADERS = {"User-Agent": "mtg-tok/1.0", "Accept": "application/json"}
FIELDS = ("mana_cost", "type_line", "oracle_text", "power", "toughness", "loyalty")
NOT_REAL_CARDS = {"token", "double_faced_token", "emblem", "art_series"}


def is_commander_legal(card):
    return card["legalities"]["commander"] == "legal" and card["layout"] not in NOT_REAL_CARDS


def as_text(card):
    faces = card.get("card_faces") or [card]   # split / adventure / transform cards
    lines = [card["name"]] + [face.get(f) or "" for face in faces for f in FIELDS]
    return "\n".join(line for line in lines if line)


# 1. Ask Scryfall for today's "one entry per unique card" export.
catalog = requests.get("https://api.scryfall.com/bulk-data", headers=HEADERS).json()
export = next(e for e in catalog["data"] if e["type"] == "oracle_cards")

# 2. Download it. It's gzipped JSONL: one card object per line.
compressed = requests.get(export["jsonl_download_uri"], headers=HEADERS).content
cards = [json.loads(line) for line in gzip.decompress(compressed).splitlines()]
cards = [card for card in cards if is_commander_legal(card)]

# 3. Tokenize every card in one call, then report.
encoder = tiktoken.get_encoding("cl100k_base")
counts = [len(t) for t in encoder.encode_ordinary_batch([as_text(c) for c in cards])]

print(f"{len(cards):,} Commander-legal cards")
print(f"{sum(counts):,} tokens total")
print(f"{statistics.mean(counts):.1f} average tokens per card")