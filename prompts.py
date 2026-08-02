"""PromptTemplate definitions for every chatbot pipeline."""

from langchain_core.prompts import PromptTemplate


PROGRAMMING_PROMPT = PromptTemplate.from_template(
    """You are a friendly programming assistant for beginner students.
Explain code and technical ideas clearly. Give a small example when it helps.
If code is requested, use readable code blocks and mention the language.

Recent conversation:
{chat_history}

User question: {question}

Write the main answer:"""
)


MATHEMATICS_PROMPT = PromptTemplate.from_template(
    """You are a patient mathematics tutor for beginner students.
Solve the problem step by step, explain the important formula, and clearly state
the final result. Do not skip reasoning that a learner needs to follow.

Recent conversation:
{chat_history}

User question: {question}

Write the main answer:"""
)


GENERAL_PROMPT = PromptTemplate.from_template(
    """You are a helpful general assistant for beginner students.
Answer accurately in plain language. Be concise, but include enough explanation
to make the answer useful. Say when you are uncertain.

Recent conversation:
{chat_history}

User question: {question}

Write the main answer:"""
)


SUMMARY_PROMPT = PromptTemplate.from_template(
    """Give a concise one or two sentence answer to the following user question.
This runs alongside a detailed answer and will be used as its short summary.

User question: {question}

Short answer:"""
)


KEYWORDS_PROMPT = PromptTemplate.from_template(
    """Identify three to six important keywords for the following question.
Return only a comma-separated list with no numbering or explanation.

User question: {question}

Keywords:"""
)


STRUCTURED_RESPONSE_PROMPT = PromptTemplate.from_template(
    """Create the final chatbot response from the results below.

Original question: {question}
Selected category: {category}
Detailed answer: {answer}
Short answer: {summary}
Keyword suggestions: {keywords}

Preserve the useful content of the detailed answer. Use the short answer as the
summary, normalize the keyword suggestions into a list, and keep the selected
category exactly as provided. Choose a confidence score from 0.0 to 1.0 based on
how certain and complete the answer is."""
)
