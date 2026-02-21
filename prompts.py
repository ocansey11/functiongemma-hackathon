"""
prompts.py — Prompt engineering strategies for FunctionGemma hybrid routing.

Strategies implemented:
  1. Base system prompt          — strong instruction-following rules
  2. Chain-of-Thought (CoT)      — ask Gemma to reason before calling tools
  3. Tool pre-selection prompt   — "query doubling" (Desmond's idea)
                                   ask Gemma which tool fits BEFORE the real call
  4. Self-consistency            — run N times, take majority vote on tool+args
  5. Retry with hint             — on low confidence, retry with the suspected
                                   tool name injected into the prompt

All prompt builders return strings ready to drop into the messages list.
"""

import json
import re
from collections import Counter


# ─────────────────────────────────────────────────────────────
#  1. BASE SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are a precise function-calling assistant.

CRITICAL RULES:
1. Count every distinct action the user is requesting BEFORE you respond.
2. If the user requests multiple actions, you MUST return ALL of them as separate function calls.
3. Never collapse multiple actions into one function call.
4. Match each action to the most specific tool available.
5. Extract argument values exactly as stated — do not paraphrase names, times, or places.
6. Integer fields (hour, minute, minutes) must always be integers, never strings.
7. Never invent tools that are not in the provided list.
8. Return function calls in the same order the user requested them."""


# ─────────────────────────────────────────────────────────────
#  2. CHAIN-OF-THOUGHT PROMPT
#     Research basis: Wei et al. 2022 "Chain-of-Thought Prompting"
# ─────────────────────────────────────────────────────────────

COT_SUFFIX = (
    "\n\nBefore calling any tools, think step by step:\n"
    "- How many distinct actions am I being asked to perform?\n"
    "- Which tool matches each action?\n"
    "- What are the exact argument values?\n"
    "Then call all required tools."
)


def build_cot_messages(messages: list) -> list:
    """Append CoT reasoning suffix to the last user message."""
    result = []
    for i, m in enumerate(messages):
        if m["role"] == "user" and i == len(messages) - 1:
            result.append({"role": "user", "content": m["content"] + COT_SUFFIX})
        else:
            result.append(m)
    return result


# ─────────────────────────────────────────────────────────────
#  3. TOOL PRE-SELECTION  (Desmond's "query doubling" idea)
#     Research basis: "Plan-and-Solve" prompting (Wang et al. 2023)
# ─────────────────────────────────────────────────────────────

PRESELECT_SYSTEM = (
    "You are a tool selection assistant. "
    "Given a user request and a list of available tools, "
    "respond with ONLY a JSON array of tool names you would call, in order. "
    "Example: [\"set_alarm\", \"get_weather\"]. No explanation."
)


def build_preselect_messages(messages: list, tools: list) -> list:
    """Build a lightweight message list for the tool pre-selection pass."""
    tool_list = "\n".join(f"- {t['name']}: {t['description']}" for t in tools)
    user_content = messages[-1]["content"]
    return [
        {"role": "system", "content": PRESELECT_SYSTEM},
        {"role": "user", "content": f"Tools:\n{tool_list}\n\nRequest: {user_content}"},
    ]


def parse_preselect_response(raw_response: str) -> list:
    """Extract tool names from Gemma's pre-selection response."""
    try:
        match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return re.findall(r'"([^"]+)"', raw_response)


def build_hinted_system_prompt(selected_tools: list) -> str:
    """Inject pre-selected tool names as a hint into the system prompt."""
    if not selected_tools:
        return BASE_SYSTEM_PROMPT
    hint = ", ".join(f'"{t}"' for t in selected_tools)
    return (
        BASE_SYSTEM_PROMPT
        + f"\n\nHINT: For this request, the relevant tools are: [{hint}]. "
        "Use all of them if the request requires it."
    )


# ─────────────────────────────────────────────────────────────
#  4. SELF-CONSISTENCY  (majority vote across N runs)
#     Research basis: Wang et al. 2023 "Self-Consistency Improves CoT"
# ─────────────────────────────────────────────────────────────

def majority_vote(all_results: list) -> list:
    """Given a list of function_calls lists from N runs, return the most common set."""
    if not all_results:
        return []

    def calls_key(calls):
        return tuple(
            (c["name"], tuple(sorted(c.get("arguments", {}).items())))
            for c in calls
        )

    keys = [calls_key(r) for r in all_results]
    most_common_key = Counter(keys).most_common(1)[0][0]
    return [
        {"name": name, "arguments": dict(args)}
        for name, args in most_common_key
    ]


# ─────────────────────────────────────────────────────────────
#  5. RETRY WITH HINT
#     Research basis: "Self-Refine" (Madaan et al. 2023)
# ─────────────────────────────────────────────────────────────

UNCERTAINTY_LOW  = 0.40
UNCERTAINTY_HIGH = 0.65


def confidence_band(confidence: float) -> str:
    """Classify confidence into: 'trust' | 'retry' | 'cloud'"""
    if confidence >= UNCERTAINTY_HIGH:
        return "trust"
    elif confidence >= UNCERTAINTY_LOW:
        return "retry"
    else:
        return "cloud"


def build_retry_messages(messages: list, first_calls: list) -> list:
    """Build a retry prompt that tells Gemma what it returned last time."""
    if first_calls:
        prev = ", ".join(c["name"] for c in first_calls)
        hint = f"\n\n[Your previous attempt called: {prev}. Double-check this is correct and complete.]"
    else:
        hint = "\n\n[Your previous attempt returned no tool calls. Try again carefully.]"

    result = []
    for i, m in enumerate(messages):
        if m["role"] == "user" and i == len(messages) - 1:
            result.append({"role": "user", "content": m["content"] + hint})
        else:
            result.append(m)
    return result


__all__ = [
    "BASE_SYSTEM_PROMPT",
    "COT_SUFFIX",
    "build_cot_messages",
    "build_preselect_messages",
    "build_hinted_system_prompt",
    "build_retry_messages",
    "parse_preselect_response",
    "majority_vote",
    "confidence_band",
    "UNCERTAINTY_LOW",
    "UNCERTAINTY_HIGH",
]
