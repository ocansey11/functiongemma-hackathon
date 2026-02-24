import sys
sys.path.insert(0, '..')
from tool_rag import get_embedding, cosine_similarity
from window_test import sliding_window_chunk

tools = [
    {'name': 'get_weather', 'description': 'Get current weather for a location'},
    {'name': 'set_alarm', 'description': 'Set an alarm for a given time'},
    {'name': 'send_message', 'description': 'Send a message to a contact'},
    {'name': 'play_music', 'description': 'Play a song or artist'},
    {'name': 'get_calendar', 'description': 'Get upcoming calendar events'},
]

# Pre-embed tools
print("Embedding tools...")
tool_embeddings = []
for tool in tools:
    text = f"{tool['name']}: {tool['description']}"
    tool_embeddings.append(get_embedding(text))
print("Done.\n")

queries = [
    'wake me up at 7am',
    'check the weather in London and text Sarah it is cold',
    'set alarm for 7 check weather text Sarah play jazz',
]

for q in queries:
    print("=" * 55)
    print(f"QUERY: {q}")
    print("=" * 55)

    # Get chunks with window=3
    chunks = sliding_window_chunk(q, window=3, drop_threshold=0.1)
    print(f"Chunks: {chunks}\n")

    # For each chunk, find best matching tool
    selected_tools = set()
    for chunk in chunks:
        chunk_emb = get_embedding(chunk)
        scores = []
        for i, tool_emb in enumerate(tool_embeddings):
            score = cosine_similarity(chunk_emb, tool_emb)
            scores.append((score, tools[i]['name']))
        scores.sort(reverse=True)

        best_score, best_tool = scores[0]
        print(f"  chunk: '{chunk}'")
        for score, name in scores:
            marker = "✓" if score == best_score else " "
            print(f"    {marker} {name:<20} {score:.4f}")
        
        if best_score > 0.55:  # minimum confidence
            selected_tools.add(best_tool)
        print()

    print(f"→ Final selected tools: {list(selected_tools)}\n")
