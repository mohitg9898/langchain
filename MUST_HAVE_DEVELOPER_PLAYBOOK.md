# Must-Have Developer Playbook (Workspace: Work)

This file is your single source of truth for:
- What to learn in what order
- How to remember code without memorizing everything
- Reusable templates with comments
- Fast daily revision workflow

## 1) Workspace Learning Order

Follow this order repeatedly:
1. `pydantic/` -> data modeling basics
2. `fastapi/` and `Insurance-Model-API/` -> API design and request/response flow
3. `llm_engineering/week1-4` -> LLM basics and prompting
4. `llm_engineering/week5+` -> RAG and advanced retrieval
5. `langchain/` -> framework-specific abstractions and chaining

Why this order:
- Pydantic + FastAPI gives backend fundamentals.
- LLM + RAG builds on those fundamentals.
- LangChain becomes easier when core concepts are clear.

## 2) How Developers Actually Remember Code

Remember this 3-layer system:
1. Concepts in head: what each block is for
2. Patterns in notes: small reusable templates
3. Exact syntax in docs/snippets: looked up fast when needed

Do this retention loop:
- Day 1: Learn + run code
- Day 2: Rebuild from memory (without looking)
- Day 4: Rebuild again
- Day 7: Explain in plain English
- Day 14: Build mini project using same pattern

## 3) Universal Build Pattern (Use Everywhere)

Use this sequence for any feature:
1. Input schema
2. Validation
3. Core logic
4. Output schema
5. Tests
6. Logs + error handling

## 4) Must-Know Template: RAG (Index + Query)

### 4.1 Indexing Template (Day 2 style)

```python
# 1) Load docs
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

folders = ["knowledge-base/employees", "knowledge-base/products", "knowledge-base/contracts"]
documents = []
for folder in folders:
    # Load markdown files from each folder
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()
    for d in docs:
        # Keep metadata for debugging and filtering
        d.metadata["doc_type"] = folder.split("/")[-1]
        documents.append(d)

# 2) Split docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3) Embed + store in vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="vector_db"
)
```

### 4.2 Query Template (Day 3 style)

```python
# Reconnect to existing DB using SAME embedding model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="vector_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = AzureChatOpenAI(
    temperature=0,
    model_name="gpt-4o",
    azure_endpoint="...",
    api_key="...",
    api_version="..."
)

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant for Insurellm.
Use only relevant context when available.
If context is missing, say you do not know.
Context:\n{context}
"""

def answer_question(question: str) -> str:
    # Similarity search happens here
    docs = retriever.invoke(question)

    # Build context from retrieved chunks
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build final system prompt
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    # Ask LLM with context-grounded prompt
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])
    return response.content
```

RAG golden rule:
- Same embedding model for indexing and querying.

## 5) Must-Know Template: FastAPI Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Insurance API")

class PredictRequest(BaseModel):
    # Input fields with constraints for validation
    age: int = Field(..., ge=0, le=120)
    bmi: float = Field(..., gt=0)
    smoker: bool

class PredictResponse(BaseModel):
    # Standard response schema
    premium: float
    risk_level: str

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Core business logic
        base = 1000
        premium = base + req.age * 10 + req.bmi * 20 + (500 if req.smoker else 0)
        risk = "high" if premium > 2500 else "medium" if premium > 1700 else "low"
        return PredictResponse(premium=round(premium, 2), risk_level=risk)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {ex}")
```

## 6) Must-Know Template: Pydantic Model

```python
from pydantic import BaseModel, field_validator

class UserInput(BaseModel):
    name: str
    email: str

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        # Avoid silent bad input
        if not value.strip():
            raise ValueError("name cannot be blank")
        return value
```

## 7) Debugging Checklist (Use Every Time)

1. Interpreter check: ensure `.venv` is active
2. Dependency check: import the exact package in same environment
3. Data check: print sample input and output shapes/content
4. Retrieval check (RAG): print retrieved docs count and first chunk snippet
5. Prompt check: print final system prompt before LLM call
6. API check: validate env vars and endpoint settings
7. Error check: isolate failing function with minimal reproducible input

## 8) Daily Developer Routine (30-45 min)

1. 5 min: review this file
2. 10 min: rewrite one template from memory
3. 10 min: run one real notebook/API file in this workspace
4. 10 min: note 3 mistakes and fixes
5. 5 min: commit small improvement

## 9) Personal Snippet Bank (Fill As You Learn)

Add your best snippets below:

### Snippet: env setup
```python
from dotenv import load_dotenv
import os
load_dotenv(override=True)
key = os.getenv("AZURE_OPENAI_API_KEY")
```

### Snippet: retriever debug
```python
docs = retriever.invoke("Who is Avery Lancaster?")
print("docs:", len(docs))
print(docs[0].page_content[:300] if docs else "No docs found")
```

### Snippet: chat history combine
```python
def combine_history(question: str, history: list[dict]) -> str:
    # Collect prior user messages for history-aware retrieval
    prior = "\n".join(
        item["content"][0]["text"]
        for item in history
        if item.get("role") == "user" and item.get("content")
    )
    return (prior + "\n" + question).strip()
```

## 10) One-Line Reminder

Do not memorize code. Memorize architecture + workflow, then reuse templates.
