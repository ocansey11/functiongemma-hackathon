from main import generate_cactus, print_result

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "set_alarm",
        "description": "Set an alarm for a given time",
        "parameters": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Hour in 24h format"},
                "minute": {"type": "integer", "description": "Minute"}
            },
            "required": ["hour", "minute"]
        }
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Contact name"},
                "message": {"type": "string", "description": "Message content"}
            },
            "required": ["recipient", "message"]
        }
    },
    {
        "name": "play_music",
        "description": "Play a song or artist",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Song or artist name"}
            },
            "required": ["query"]
        }
    }
]

test_cases = [
    {
        "label": "Two intents — alarm + weather",
        "content": "Set an alarm for 7:30 AM and check the weather in London"
    },
    {
        "label": "Three intents — alarm + weather + message",
        "content": "Wake me up at 6 AM, check the weather in Paris, and text Sam that I'm on my way"
    },
    {
        "label": "Ambiguous — could collapse",
        "content": "Play some Drake and also send John a message saying let's link tonight"
    },
    {
        "label": "Hard — four intents",
        "content": "Set an alarm for 8 AM, check weather in Tokyo, text Sarah good morning, and play some jazz"
    },
]

for case in test_cases:
    messages = [{"role": "user", "content": case["content"]}]
    result = generate_cactus(messages, tools)
    print_result(case["label"], result)
    print("-" * 50)
