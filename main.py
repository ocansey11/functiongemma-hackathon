import sys
import json, os, time
import urllib.request
from google import genai
from prompts import BASE_SYSTEM_PROMPT, build_cot_messages
from google.genai import types
from tool_rag import ToolSelector

OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# Global selector — initialised lazily per unique tool set
_selector_cache = {}

def get_selector(tools):
    """Cache selector per tool set so we don't re-embed tools every call."""
    key = tuple(t["name"] for t in tools)
    if key not in _selector_cache:
        _selector_cache[key] = ToolSelector(tools)
    return _selector_cache[key]


def generate_cactus(messages, tools):
    """Run function calling on-device via Ollama (local substitute for Cactus)."""

    # Tool RAG — filter to relevant tools before sending to model
    user_query = next((m["content"] for m in messages if m["role"] == "user"), "")
    selector = get_selector(tools)
    filtered_tools = selector.select_threshold(user_query, threshold=0.6)

    # Fallback — if nothing selected use all tools
    if not filtered_tools:
        filtered_tools = tools

    ollama_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
        }
        for t in filtered_tools
    ]

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": BASE_SYSTEM_PROMPT}] + messages,
        "tools": ollama_tools,
        "stream": False,
    }).encode("utf-8")

    start_time = time.time()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Ollama error: {e}")
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    tool_calls = raw.get("message", {}).get("tool_calls", [])
    for tc in tool_calls:
        function_calls.append({
            "name": tc["function"]["name"],
            "arguments": tc["function"].get("arguments", {}),
        })

    confidence = 1.0 if function_calls else 0.0

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
        "confidence": confidence,
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Hybrid inference: on-device first, cloud fallback."""
    local = generate_cactus(messages, tools)

    if local["confidence"] >= confidence_threshold:
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud


def print_result(label, result):
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
