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


class ToolSelector:

    def __init__(self, tools: list):
        self.tools = tools
        self.tool_embeddings = []

        print(f"Embedding {len(tools)} tools...")
        for tool in tools:
            text = f"{tool['name']}: {tool['description']}"
            embedding = get_embedding(text)
            self.tool_embeddings.append(embedding)
        print("Tools embedded.\n")

    def _score_all(self, query: str):
        """Score all tools against query and return sorted list."""
        query_embedding = get_embedding(query)
        scores = []
        for i, tool_embedding in enumerate(self.tool_embeddings):
            score = cosine_similarity(query_embedding, tool_embedding)
            scores.append((score, self.tools[i]))
        scores.sort(reverse=True)
        return scores

    def select_top_k(self, query: str, top_k: int = 3) -> list:
        """
        METHOD 1 — Top-K
        Always returns exactly K tools, ranked by similarity.
        Problem: even irrelevant tools get included if K is too high.
        Even if only 1 tool is relevant, you still get K tools.
        """
        scores = self._score_all(query)

        print(f"[Top-K] All scores for: '{query}'")
        for score, tool in scores:
            print(f"  {tool['name']:<20} score={score:.4f}")

        selected = [tool for _, tool in scores[:top_k]]
        print(f"[Top-K] Selected top {top_k}: {[t['name'] for t in selected]}\n")
        return selected

    def select_threshold(self, query: str, threshold: float = 0.6) -> list:
        """
        METHOD 2 — Threshold
        Only returns tools whose similarity score exceeds the threshold.
        More precise — irrelevant tools are excluded entirely.
        Problem: if threshold is too high, you might exclude relevant tools too.
        Falls back to top-1 if nothing passes threshold.
        """
        scores = self._score_all(query)

        print(f"[Threshold] All scores for: '{query}'")
        for score, tool in scores:
            marker = "✓" if score >= threshold else "✗"
            print(f"  {marker} {tool['name']:<20} score={score:.4f}")

        selected = [tool for score, tool in scores if score >= threshold]

        if not selected:
            # fallback — always return at least the best match
            selected = [scores[0][1]]
            print(f"[Threshold] Nothing passed {threshold}, falling back to best match")

        print(f"[Threshold] Selected: {[t['name'] for t in selected]}\n")
        return selected


    def select_gap(self, query: str, threshold: float = 0.6, min_gap: float = 0.05) -> list:
        """
        METHOD 3 — Threshold + Gap Filtering
        Same as threshold but also checks the score gap between consecutive tools.
        If there is a big drop in score between two tools, we stop there.
        
        Example:
          get_weather   0.79  ← include
          send_message  0.62  ← include (above threshold)
          get_calendar  0.61  ← gap from send_message is only 0.01, include
          play_music    0.45  ← gap from get_calendar is 0.16 (> min_gap), STOP
          set_alarm     0.42  ← excluded
        
        The gap acts as a natural cliff detector — when scores suddenly drop,
        we know we've crossed from relevant to irrelevant tools.
        """
        scores = self._score_all(query)

        print(f"[Gap] All scores for: '{query}'")
        selected = []

        for i, (score, tool) in enumerate(scores):
            if score < threshold:
                print(f"  ✗ {tool['name']:<20} score={score:.4f}  (below threshold)")
                continue

            if i > 0:
                prev_score = scores[i - 1][0]
                gap = prev_score - score
                if gap > min_gap:
                    print(f"  ✗ {tool['name']:<20} score={score:.4f}  (gap={gap:.4f} > {min_gap}, stopped)")
                    break

            print(f"  ✓ {tool['name']:<20} score={score:.4f}")
            selected.append(tool)

        if not selected:
            selected = [scores[0][1]]
            print(f"  fallback to best match")

        print(f"[Gap] Selected: {[t['name'] for t in selected]}\n")
        return selected
