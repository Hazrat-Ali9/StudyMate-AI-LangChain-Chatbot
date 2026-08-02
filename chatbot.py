"""LangChain chatbot pipeline using branches, parallel work, and Pydantic."""

import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
)
from langchain_groq import ChatGroq

from prompts import (
    GENERAL_PROMPT,
    KEYWORDS_PROMPT,
    MATHEMATICS_PROMPT,
    PROGRAMMING_PROMPT,
    STRUCTURED_RESPONSE_PROMPT,
    SUMMARY_PROMPT,
)
from schemas import ChatbotResponse


PROGRAMMING_KEYWORDS = {
    "algorithm",
    "api",
    "bug",
    "class",
    "code",
    "coding",
    "css",
    "database",
    "function",
    "git",
    "html",
    "java",
    "javascript",
    "programming",
    "python",
    "react",
    "sql",
}

MATHEMATICS_KEYWORDS = {
    "algebra",
    "calculate",
    "calculus",
    "derivative",
    "equation",
    "formula",
    "geometry",
    "integral",
    "math",
    "mathematics",
    "probability",
    "solve",
    "statistics",
}


def _words(text: str) -> set[str]:
    """Return lowercase words used by the simple beginner-friendly router."""

    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def route_question(question: str) -> str:
    """Choose a branch with transparent keyword-based rules."""

    words = _words(question)
    if words & PROGRAMMING_KEYWORDS:
        return "Programming"
    if words & MATHEMATICS_KEYWORDS or re.search(
        r"\d\s*[+\-*/^=]\s*\d", question
    ):
        return "Mathematics"
    return "General"


def _question_is(category: str):
    """Create a RunnableBranch condition for a category."""

    return lambda data: route_question(data["question"]) == category


def build_chatbot_chain() -> Runnable[dict[str, str], ChatbotResponse]:
    """Build the complete LangChain Expression Language pipeline."""

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    model = ChatGroq(
        model=model_name,
        temperature=0.2,
        max_retries=2,
    )
    text_parser = StrOutputParser()

    programming_chain = PROGRAMMING_PROMPT | model | text_parser
    mathematics_chain = MATHEMATICS_PROMPT | model | text_parser
    general_chain = GENERAL_PROMPT | model | text_parser

    # RunnableBranch evaluates conditions in order and uses the last chain as
    # the default. Each route receives the same {question, chat_history} input.
    answer_branch = RunnableBranch(
        (_question_is("Programming"), programming_chain),
        (_question_is("Mathematics"), mathematics_chain),
        general_chain,
    )

    summary_chain = SUMMARY_PROMPT | model | text_parser
    keywords_chain = KEYWORDS_PROMPT | model | text_parser

    # These three model calls run concurrently from one user request.
    parallel_outputs = RunnableParallel(
        answer=answer_branch,
        summary=summary_chain,
        keywords=keywords_chain,
        question=RunnableLambda(lambda data: data["question"]),
        category=RunnableLambda(lambda data: route_question(data["question"])),
    )

    # Groq constrains the final model call to the Pydantic JSON schema. The
    # resulting value is a validated ChatbotResponse instance.
    structured_model = model.with_structured_output(
        ChatbotResponse,
        method="json_schema",
        strict=True,
    )

    return parallel_outputs | STRUCTURED_RESPONSE_PROMPT | structured_model


class Chatbot:
    """Small service class used by the Streamlit UI."""

    def __init__(self) -> None:
        self.chain = build_chatbot_chain()

    def ask(
        self,
        question: str,
        chat_history: str = "No previous messages.",
    ) -> ChatbotResponse:
        """Run the chatbot and guarantee a Pydantic response."""

        result: Any = self.chain.invoke(
            {"question": question, "chat_history": chat_history}
        )
        if isinstance(result, ChatbotResponse):
            return result
        return ChatbotResponse.model_validate(result)
