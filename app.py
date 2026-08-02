"""Streamlit chat interface for the LangChain chatbot."""

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from chatbot import Chatbot
from schemas import ChatbotResponse


load_dotenv()

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🤖",
    layout="centered",
)


@st.cache_resource
def get_chatbot() -> Chatbot:
    """Create the LangChain pipeline only once per Streamlit process."""

    return Chatbot()


def format_history(messages: list[dict[str, Any]], limit: int = 6) -> str:
    """Convert recent Streamlit messages into a short prompt context."""

    lines: list[str] = []
    for message in messages[-limit:]:
        if message["role"] == "user":
            lines.append(f"User: {message['content']}")
        else:
            response = message["content"]
            lines.append(f"Assistant: {response['answer']}")
    return "\n".join(lines) or "No previous messages."


def display_response(response: dict[str, Any]) -> None:
    """Render every field from the validated structured output."""

    st.markdown(response["answer"])
    st.markdown(f"**Summary:** {response['summary']}")

    category_column, confidence_column = st.columns(2)
    category_column.metric("Category", response["category"])
    confidence_column.metric("Confidence", f"{response['confidence']:.0%}")

    st.markdown(f"**Keywords:** {', '.join(response['keywords'])}")
    with st.expander("View validated Pydantic output"):
        st.json(response)


st.title("🤖 StudyMate AI")
st.caption(
    "A simple LangChain chatbot with branching, parallel pipelines, "
    "and structured output."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "Questions are routed to a **Programming**, **Mathematics**, or "
        "**General** assistant. The answer, summary, and keywords are then "
        "generated in parallel."
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for saved_message in st.session_state.messages:
    with st.chat_message(saved_message["role"]):
        if saved_message["role"] == "user":
            st.markdown(saved_message["content"])
        else:
            display_response(saved_message["content"])

has_api_key = bool(os.getenv("GROQ_API_KEY"))
if not has_api_key:
    st.warning("Add your GROQ_API_KEY to a .env file, then restart Streamlit.")

question = st.chat_input(
    "Ask a programming, math, or general question...",
    disabled=not has_api_key,
)

if question:
    history = format_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                structured_response: ChatbotResponse = get_chatbot().ask(question, history)
                response_data = structured_response.model_dump()
                display_response(response_data)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_data}
                )
            except Exception as error:
                st.error(f"The chatbot could not answer: {error}")
