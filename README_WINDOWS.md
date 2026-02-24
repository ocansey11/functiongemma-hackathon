# Running on Windows (No Mac / No ARM)

This setup is for Windows users who can't build Cactus natively (Cactus requires ARM).
We use Ollama as a local inference substitute.

## Requirements
- Windows 10/11 with WSL2 (Ubuntu) **or** Ollama for Windows
- Python 3.10+
- Ollama: https://ollama.ai

## Setup

**Option 1 — WSL2 (recommended)**
```bash
# Inside WSL Ubuntu
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

**Option 2 — Ollama for Windows**
Download and install from https://ollama.ai, then in PowerShell:
```
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

## Run
```bash
pip install google-genai
export GEMINI_API_KEY="your_key_here"
python benchmark.py
```

## Notes
- `qwen2.5:0.5b` replaces FunctionGemma for local testing
- `nomic-embed-text` powers the Tool RAG selector
- Results will differ from Mac/ARM Cactus benchmarks
- For real benchmark results, run on Mac via Sam
