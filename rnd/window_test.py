import json
import math
import urllib.request

EMBED_MODEL = "nomic-embed-text"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"


def get_embedding(text: str) -> list:
    payload = json.dumps({
        "model": EMBED_MODEL,
        "prompt": text,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["embedding"]


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def sliding_window_chunk(query: str, window: int = 2, drop_threshold: float = 0.1):
    """
    Split a query into intent chunks using adjacent window similarity.
    
    window=2 means we embed bigrams (pairs of words) for more context.
    drop_threshold=0.1 means a similarity drop of 0.1 signals a boundary.
    """
    words = query.split()

    if len(words) <= window:
        return [query]

    # Build windows (bigrams or trigrams)
    windows = []
    for i in range(len(words) - window + 1):
        chunk = " ".join(words[i:i + window])
        windows.append(chunk)

    print(f"\nWindows: {windows}")

    # Embed all windows
    embeddings = []
    for w in windows:
        embeddings.append(get_embedding(w))

    # Compute similarity between adjacent windows
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    print(f"\nAdjacent similarities:")
    for i, sim in enumerate(similarities):
        print(f"  '{windows[i]}' ↔ '{windows[i+1]}'  =  {sim:.4f}")

    # Find boundaries where similarity drops sharply
    boundaries = []
    for i in range(1, len(similarities)):
        drop = similarities[i - 1] - similarities[i]
        if drop > drop_threshold:
            boundary_word = i + window - 1
            boundaries.append(boundary_word)
            print(f"  → Boundary at word {boundary_word} (drop={drop:.4f})")

    # Split query at boundaries
    chunks = []
    prev = 0
    for b in boundaries:
        chunk = " ".join(words[prev:b])
        if chunk:
            chunks.append(chunk)
        prev = b
    chunks.append(" ".join(words[prev:]))

    return chunks


# Test queries — mix of single and multi intent
queries = [
    "wake me up at 7am",
    "check the weather in London and text Sarah it is cold",
    "play some jazz",
    "set alarm for 7 check weather text Sarah play jazz",
    "remind me to call mum tomorrow morning",
]

for q in queries:
    print("\n" + "=" * 55)
    print(f"QUERY: {q}")
    print("=" * 55)
    chunks = sliding_window_chunk(q, window=2, drop_threshold=0.1)
    print(f"\n→ Chunks: {chunks}")
