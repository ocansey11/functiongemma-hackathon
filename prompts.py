BASE_SYSTEM_PROMPT = """You are a precise function-calling assistant.

CRITICAL RULES:
1. Count every distinct action the user is requesting BEFORE you respond.
2. If the user requests multiple actions, you MUST return ALL of them as separate function calls.
3. Never collapse multiple actions into one function call.
4. Match each action to the most specific tool available.
5. Extract argument VALUES exactly as stated by the user — not the schema description.
6. Integer fields must always be integers, never strings.
7. Never invent tools that are not in the provided list.
8. Return function calls in the same order the user requested them."""

COT_SUFFIX = (
    "\n\nBefore calling any tools, think step by step:\n"
    "- How many distinct actions am I being asked to perform?\n"
    "- Which tool matches each action?\n"
    "- What are the exact argument values from the user's message?\n"
    "Then call all required tools."
)


def build_cot_messages(messages):
    result = []
    for i, m in enumerate(messages):
        if m["role"] == "user" and i == len(messages) - 1:
            result.append({"role": "user", "content": m["content"] + COT_SUFFIX})
        else:
            result.append(m)
    return result


__all__ = ["BASE_SYSTEM_PROMPT", "COT_SUFFIX", "build_cot_messages"]
