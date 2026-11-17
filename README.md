# StudyMate AI — LangChain Chatbot

StudyMate AI is a beginner-friendly chatbot built with Streamlit and the latest
LangChain Expression Language APIs. It routes questions to a specialized
assistant, creates several useful outputs concurrently, and returns a validated
Pydantic object.

## Features

- Streamlit chat interface with conversation history
- Groq chat model configured through a `.env` file
- `PromptTemplate` objects with dynamic question and history variables
- Programming, mathematics, and general routes with `RunnableBranch`
- Concurrent answer, summary, and keyword pipelines with `RunnableParallel`
- Strict, validated structured output using Pydantic `BaseModel`
- Optional **Clear chat** button
- Friendly missing-key and API error messages

## How the pipeline work

```text
User question
     |
     v
RunnableBranch
  |-- Programming prompt
  |-- Mathematics prompt
  `-- General prompt (default)
     |
     v
RunnableParallel
  |-- Routed detailed answer
  |-- Short answer / summary
  `-- Keywords
     |
     v
Structured-output model
     |
     v
ChatbotResponse (Pydantic validation)
     |
     v
Streamlit chat UI
```

### RunnableBranch

[`chatbot.py`](chatbot.py) creates three prompt pipelines. Transparent keyword
rules inspect the question, and `RunnableBranch` selects the programming or
mathematics pipeline. If neither condition matches, LangChain runs the general
pipeline as the default.

```python
answer_branch = RunnableBranch(
    (_question_is("Programming"), programming_chain),
    (_question_is("Mathematics"), mathematics_chain),
    general_chain,
)
```

### RunnableParallel

The selected detailed answer, a concise answer, and keywords all use the same
user request. `RunnableParallel` starts these independent pipelines concurrently
and returns their results in one dictionary.

```python
parallel_outputs = RunnableParallel(
    answer=answer_branch,
    summary=summary_chain,
    keywords=keywords_chain,
    question=RunnableLambda(lambda data: data["question"]),
    category=RunnableLambda(lambda data: route_question(data["question"])),
)
```

### Pydantic structured output

[`schemas.py`](schemas.py) defines `ChatbotResponse` with five required fields:
`answer`, `summary`, `confidence`, `category`, and `keywords`. The Groq model is
bound to this schema with `with_structured_output(..., method="json_schema",
strict=True)`. The application receives a Pydantic object and calls
`ChatbotResponse.model_validate` as an additional validation boundary when a
provider returns a dictionary.

The UI displays every field and includes an expandable JSON view so the exact
validated structure is easy to inspect.

## Project structure

```text
chatbot-phitron/
|-- app.py              # Streamlit interface and chat history
|-- chatbot.py          # LangChain runnable pipeline
|-- prompts.py          # All PromptTemplate definitions
|-- schemas.py          # Pydantic response schema
|-- requirements.txt    # Pinned Python dependencies
|-- .env.example        # Safe environment variable template
|-- .gitignore          # Prevents secrets and local files being committed
`-- README.md
```

## Installation

Python 3.10 or newer is recommended.

1. Clone the repository and enter its directory:

   ```bash
   git clone <your-github-repository-url>
   cd chatbot-phitron
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Copy the environment template:

   ```bash
   cp .env.example .env
   ```

5. Create a key at [Groq Console](https://console.groq.com/keys), then replace
   `your_groq_api_key_here` inside `.env`. Never commit the `.env` file.

## Run the application

```bash
streamlit run app.py
```

Open the local address shown by Streamlit, normally
`http://localhost:8501`.

Try these example questions:

- Programming: `Explain Python lists with a small example.`
- Mathematics: `Solve 3x + 6 = 18 step by step.`
- General: `Why is sleep important for learning?`

## Security

API keys are read from environment variables by `python-dotenv`. The real `.env`
file and the local `.venv` directory are ignored by Git. Only `.env.example`
should be committed.
