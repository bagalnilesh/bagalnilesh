# Mistral Chat with Streamlit + Ollama

This simple chat app uses the local Ollama server with the `mistral` model via the `ollama` Python client.

## Prerequisites
- Python 3.9+
- Ollama installed and running

Install Ollama on Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull mistral
```

## Setup

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 -m streamlit run ollama_mistral_app.py --server.headless=true --server.port=8501
```

The app expects the Ollama server at `http://127.0.0.1:11434`. You can override via:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
export MODEL_NAME=mistral
```

Open the provided URL to use the chat.