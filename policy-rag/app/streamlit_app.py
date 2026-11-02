"""Streamlit chat UI for the Activ One MAX policy assistant."""

from __future__ import annotations

import logging
import importlib
import sys
from pathlib import Path

try:
    st = importlib.import_module("streamlit")
except ModuleNotFoundError as exc:
    if exc.name != "streamlit":
        raise
    raise SystemExit("Streamlit is required. Install it with: pip install streamlit") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.rag import RAGPipeline

LOGGER = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "What is the free look period?",
    "How do pre-existing disease and specific disease waiting periods interact?",
    "Does an accident have a waiting period?",
    "I'm worried—will the policy cover piles surgery?",
    "What is the weather in Delhi?",
    "What are the permanent partial disability benefits?",
]

st.set_page_config(page_title="Activ One MAX Policy Assistant", layout="wide")
st.title("Activ One MAX Policy Assistant")
st.caption("Answers are grounded only in the policy wording. Check your Policy Schedule for your personal cover.")

with st.sidebar:
    st.header("Settings")
    st.write(f"**Model:** {settings.claude_model}")
    st.write(f"**Top-k:** {settings.top_k}")
    st.write(f"**Reranker:** {'On' if settings.rerank else 'Off'}")
    st.write(f"**Refusal threshold:** {settings.refusal_threshold}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("refused"):
            st.warning("Document could not support an answer")
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("retrieved"):
            with st.expander("Retrieved passages"):
                for passage in message["retrieved"]:
                    st.markdown(
                        f"**{passage['section_id']} · pages {passage['page_start']}-"
                        f"{passage['page_end']} · {passage['chunk_type']} · "
                        f"score {passage['score']:.4f}**\n\n{passage['original_text']}"
                    )

if not st.session_state.messages:
    st.subheader("Example questions")
    columns = st.columns(3)
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 3].button(question, key=f"example-{index}"):
            st.session_state.pending_question = question

prompt = st.chat_input("Ask a question about this policy")
if prompt is None:
    prompt = st.session_state.pop("pending_question", None)
if prompt:
    previous_history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            result = st.session_state.pipeline.ask(prompt, history=previous_history)
            if result["refused"]:
                st.warning("Document could not support an answer")
            st.markdown(result["answer"])
            with st.expander("Retrieved passages"):
                for passage in result["retrieved"]:
                    st.markdown(
                        f"**{passage['section_id']} · pages {passage['page_start']}-"
                        f"{passage['page_end']} · {passage['chunk_type']} · "
                        f"score {passage['score']:.4f}**\n\n{passage['original_text']}"
                    )
            st.session_state.messages.append({
                "role": "assistant", "content": result["answer"],
                "retrieved": result["retrieved"], "refused": result["refused"],
            })
        except Exception as exc:
            LOGGER.exception("Question answering failed")
            st.error(f"Could not answer this question: {exc}")
