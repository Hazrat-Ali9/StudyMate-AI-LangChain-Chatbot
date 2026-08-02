"""Pydantic models used by the chatbot."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatbotResponse(BaseModel):
    """The validated response displayed in the Streamlit interface."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="A helpful, complete answer to the user")
    summary: str = Field(description="A short one or two sentence summary")
    confidence: float = Field(
        description="Confidence in the answer, from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    category: Literal["Programming", "Mathematics", "General"] = Field(
        description="The route selected for the question"
    )
    keywords: list[str] = Field(description="Important keywords from the response")
