import os
import time
import requests
import streamlit as st

try:
    import ollama
except Exception:  # noqa: BLE001
    ollama = None


APP_TITLE = "Mistral Chat (Ollama + Streamlit)"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistral")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are Mistral running via Ollama. Be concise and helpful.",
)


def is_ollama_ready(timeout_seconds: int = 2) -> bool:
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout_seconds)
        return response.status_code == 200
    except Exception:
        return False


st.set_page_config(page_title=APP_TITLE, page_icon="🌀", layout="centered")
st.title(APP_TITLE)
st.caption(
    f"Backend: {OLLAMA_HOST}  •  Model: `{MODEL_NAME}`  •  Streamlit session with chat history"
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

with st.sidebar:
    st.subheader("Status")
    if is_ollama_ready():
        st.success("Ollama server is reachable.")
        if ollama is None:
            st.warning(
                "Python package `ollama` is not available. Install it to enable chatting."
            )
        else:
            try:
                tags = ollama.list()
                model_ids = [m.get("model") for m in tags.get("models", [])]
                if MODEL_NAME not in model_ids:
                    st.info(
                        f"Model `{MODEL_NAME}` not found. Run `ollama pull {MODEL_NAME}` in a terminal."
                    )
                else:
                    st.success(f"Model `{MODEL_NAME}` is available.")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not list models: {exc}")
    else:
        st.error(
            "Ollama server is not reachable. Start it (e.g., run `ollama serve`) and/or install Ollama."
        )

    st.markdown("---")
    st.caption("Options")
    MODEL_NAME = st.text_input("Model", value=MODEL_NAME, help="e.g., mistral, mistral:latest")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max Tokens", 64, 4096, 512, 64)
    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        st.experimental_rerun()

# Display prior assistant/user messages
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 

user_input = st.chat_input("Ask anything…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        if ollama is None:
            placeholder.error("Python package `ollama` not installed. `pip install ollama`."
            )
        elif not is_ollama_ready():
            placeholder.error("Ollama server not reachable at OLLAMA_HOST.")
        else:
            try:
                # Stream the response tokens from the model
                stream = ollama.chat(
                    model=MODEL_NAME,
                    messages=st.session_state.messages,
                    options={
                        "temperature": float(temperature),
                        "num_predict": int(max_tokens),
                    },
                    stream=True,
                )
                assistant_text = ""
                for chunk in stream:
                    delta = chunk.get("message", {}).get("content", "")
                    assistant_text += delta
                    placeholder.markdown(assistant_text)
                    time.sleep(0.01)

                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
            except Exception as exc:  # noqa: BLE001
                placeholder.error(f"Chat error: {exc}")