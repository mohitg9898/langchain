"""
LangChain Production Cheat Sheet (Comprehensive)
================================================

Why this file exists
- You asked for one practical reference to read and practice almost everything.
- This file is intentionally broad, production-oriented, and ordered from basics to advanced.
- Most snippets are templates: copy into your app and adapt.

Important
- LangChain evolves quickly. Keep versions pinned and review release notes.
- Some integrations require extra packages and API keys.
"""

# =============================================================================
# 0) Install and environment setup
# =============================================================================

"""
Baseline install:

pip install -U \
  langchain langchain-core langchain-community langchain-openai \
  langchain-text-splitters python-dotenv pydantic faiss-cpu chromadb

Optional integrations:
- pypdf, unstructured, bs4, lxml
- sentence-transformers
- pgvector, psycopg[binary]
- elasticsearch
- langchain-cohere (reranking)
- tiktoken
"""

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "langchain-prod")


# =============================================================================
# 1) Core mental model
# =============================================================================

# Most modern LangChain code is Runnable-based.
# You compose with LCEL:
#   input -> prompt -> model -> parser
#   a | b | c


# =============================================================================
# 2) Models: chat and embeddings
# =============================================================================

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Notes:
# - For low-latency / cheaper tasks, use smaller model variants.
# - Keep temperature near 0 for deterministic backends (RAG, extraction).


# =============================================================================
# 3) Prompting patterns you should know
# =============================================================================

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.prompts.chat import MessagesPlaceholder

# 3.1 Basic chat prompt
basic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are concise and factual."),
        ("user", "{input}"),
    ]
)

# 3.2 Multi-variable prompt
multi_var_prompt = ChatPromptTemplate.from_template(
    """
Role: {role}
Tone: {tone}
Question: {question}
"""
)

# 3.3 Prompt partials (inject constants once)
style_prompt = PromptTemplate.from_template("Write in {style}: {text}").partial(
    style="plain English"
)

# 3.4 Prompt with conversation history placeholder
history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("user", "{input}"),
    ]
)

# 3.5 Few-shot prompt pattern
example_prompt = ChatPromptTemplate.from_messages(
    [("human", "Input: {input}"), ("ai", "Output: {output}")]
)
few_shot = FewShotChatMessagePromptTemplate(
    examples=[
        {"input": "Hi", "output": "Hello"},
        {"input": "Bye", "output": "Goodbye"},
    ],
    example_prompt=example_prompt,
)
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate short phrases politely."),
        few_shot,
        ("human", "Input: {input}"),
    ]
)


# =============================================================================
# 4) Output parsing and structured outputs
# =============================================================================

from pydantic import BaseModel, Field
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)

str_parser = StrOutputParser()
json_parser = JsonOutputParser()


class IncidentReport(BaseModel):
    service: str = Field(description="Affected service")
    severity: str = Field(description="low|medium|high|critical")
    summary: str = Field(description="Short summary")


pydantic_parser = PydanticOutputParser(pydantic_object=IncidentReport)
structured_llm = llm.with_structured_output(IncidentReport)


# =============================================================================
# 5) LCEL essentials and composition
# =============================================================================

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

normalize_input = RunnableLambda(lambda d: {"input": d["input"].strip()})
basic_chain = normalize_input | basic_prompt | llm | str_parser

# Keep original input while generating more fields.
fanout_chain = RunnableParallel(
    original=RunnablePassthrough(),
    question=RunnableLambda(lambda d: d["input"]),
)

# Retry wrapper for flaky providers.
retry_chain = basic_chain.with_retry(stop_after_attempt=3)


# =============================================================================
# 6) Document loaders: all common families
# =============================================================================

"""
Use the loader for your source type. Key idea: everything becomes List[Document].

Common loaders in langchain-community:
- WebBaseLoader: websites
- TextLoader: plain text files
- PyPDFLoader / PDFPlumberLoader: PDF files
- CSVLoader: CSV files
- JSONLoader: JSON files
- DirectoryLoader: recursive directory loading
- UnstructuredMarkdownLoader / UnstructuredFileLoader: rich documents
- Docx2txtLoader: DOCX files
"""

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)

# Web
web_docs = WebBaseLoader("https://docs.smith.langchain.com/").load()

# Local text
txt_docs = TextLoader("notes.txt", encoding="utf-8").load()

# PDF
pdf_docs = PyPDFLoader("manual.pdf").load()

# CSV
csv_docs = CSVLoader(file_path="data.csv").load()

# JSON (jq schema may vary based on your JSON shape)
json_docs = JSONLoader(file_path="events.json", jq_schema=".[]", text_content=False).load()

# Markdown
md_docs = UnstructuredMarkdownLoader("README.md").load()

# Directory recursive loader with glob
dir_docs = DirectoryLoader("./docs", glob="**/*.md").load()


# =============================================================================
# 7) Text splitters: choose by content type
# =============================================================================

from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

# General purpose (best default)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

# Simple fixed char split
char_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Token-aware split (helpful for token budgeting)
token_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=40)

# Markdown-aware split by headings
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])

# Python code-aware splitting
py_splitter = PythonCodeTextSplitter(chunk_size=1200, chunk_overlap=100)

chunks = recursive_splitter.split_documents(web_docs)


# =============================================================================
# 8) Vector stores: FAISS, Chroma, and PGVector patterns
# =============================================================================

from langchain_community.vectorstores import FAISS

faiss_db = FAISS.from_documents(chunks, embeddings)
faiss_db.save_local("faiss_index")

# If loading FAISS from disk in development, understand the deserialization risk.
faiss_db_reloaded = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# Chroma pattern (local persistence)
try:
    from langchain_community.vectorstores import Chroma

    chroma_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="knowledge_base",
    )
except Exception:
    chroma_db = None

# PGVector pattern (production-friendly for PostgreSQL shops)
try:
    from langchain_community.vectorstores import PGVector

    pg_db = PGVector.from_documents(
        embedding=embeddings,
        documents=chunks,
        collection_name="kb_docs",
        connection_string=os.getenv("PGVECTOR_CONNECTION", ""),
    )
except Exception:
    pg_db = None


# =============================================================================
# 9) Retriever variants you should practice
# =============================================================================

# Base retriever
retriever_similarity = faiss_db.as_retriever(search_kwargs={"k": 4})

# MMR retriever for diversity
retriever_mmr = faiss_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
)

# Score-threshold retrieval
retriever_threshold = faiss_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.2, "k": 6},
)

# Multi-query retriever (LLM expands query into alternatives)
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever

    retriever_multi_query = MultiQueryRetriever.from_llm(
        retriever=retriever_similarity,
        llm=llm,
    )
except Exception:
    retriever_multi_query = retriever_similarity

# Contextual compression retriever with reranker/compressor
try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor

    compressor = LLMChainExtractor.from_llm(llm)
    retriever_compressed = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever_mmr,
    )
except Exception:
    retriever_compressed = retriever_mmr


# =============================================================================
# 10) RAG document chains: stuff, map_reduce, refine
# =============================================================================

from langchain.chains.combine_documents import (
    create_map_reduce_documents_chain,
    create_refine_documents_chain,
    create_stuff_documents_chain,
)

rag_prompt = ChatPromptTemplate.from_template(
    """
Answer using only the provided context.
If context is insufficient, say "I do not know".

Context:
{context}

Question:
{input}
"""
)

# Most common and fastest for small context windows.
stuff_chain = create_stuff_documents_chain(llm, rag_prompt)

# Map-reduce: summarize chunks independently then combine.
map_reduce_chain = create_map_reduce_documents_chain(llm, rag_prompt)

# Refine: iteratively improve answer with additional chunks.
refine_chain = create_refine_documents_chain(llm, rag_prompt, rag_prompt)


# =============================================================================
# 11) Retrieval chain (full RAG pipeline)
# =============================================================================

from langchain.chains import create_retrieval_chain

retrieval_chain = create_retrieval_chain(retriever_similarity, stuff_chain)
rag_out = retrieval_chain.invoke({"input": "What is LangSmith and why use it?"})

# Typical keys in output dict:
# - answer
# - context


# =============================================================================
# 12) Advanced RAG patterns
# =============================================================================

"""
Patterns to learn after base RAG:
- Parent document retrieval
- Self-query retriever (metadata-aware filtering)
- Ensemble retrieval (hybrid BM25 + vector)
- Reranking with cross-encoders / provider rerank APIs
- Query rewriting before retrieval
"""

# Example: metadata filtering pattern (vector-store dependent)
filtered_retriever = faiss_db.as_retriever(search_kwargs={"k": 4})
# Some stores support filter keyword in search_kwargs.


# =============================================================================
# 13) Message history and session memory
# =============================================================================

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class InMemoryHistoryStore:
    def __init__(self):
        self._store = {}

    def get(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]


history_store = InMemoryHistoryStore()
chat_chain = history_prompt | llm | str_parser

chat_chain_with_history = RunnableWithMessageHistory(
    runnable=chat_chain,
    get_session_history=history_store.get,
    input_messages_key="input",
    history_messages_key="history",
)

# Call pattern:
# chat_chain_with_history.invoke(
#     {"input": "Summarize our last decision"},
#     config={"configurable": {"session_id": "user-123"}},
# )


# =============================================================================
# 14) Tools and function calling
# =============================================================================

from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def get_exchange_rate(base: str, quote: str) -> str:
    """Return exchange rate as text. Replace with real API in production."""
    fake_rates = {("USD", "INR"): "83.10", ("EUR", "USD"): "1.07"}
    rate = fake_rates.get((base.upper(), quote.upper()), "unknown")
    return f"{base}/{quote}={rate}"


tools = [multiply, get_exchange_rate]

# Bind tools directly for model-native tool calling.
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# 15) Agent patterns (use only when needed)
# =============================================================================

"""
Use agents when the workflow needs dynamic tool choice.
For deterministic workflows (most APIs), prefer explicit chains.
"""

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a tool-using assistant."),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
except Exception:
    agent_executor = None

# In production, prefer LangGraph for complex stateful agents.


# =============================================================================
# 16) Routing patterns
# =============================================================================

# Route question to specialized chain based on custom logic.
def route_topic(d: dict) -> str:
    q = d["input"].lower()
    if "billing" in q or "invoice" in q:
        return "billing"
    return "general"


router = RunnableLambda(route_topic)

# You can then map route label to dedicated chains in app code.


# =============================================================================
# 17) Streaming and async patterns
# =============================================================================

# Token streaming pattern
def stream_answer(question: str):
    for chunk in llm.stream(question):
        yield chunk.content


# Async pattern examples (for FastAPI/async backends)
# result = await basic_chain.ainvoke({"input": "..."})
# async for event in chain.astream_events({...}, version="v1"):
#     ...


# =============================================================================
# 18) Callbacks, observability, and tracing
# =============================================================================

from langchain_core.callbacks import BaseCallbackHandler


class SimpleLoggerCallback(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("CHAIN START", serialized.get("name"), inputs)

    def on_chain_end(self, outputs, **kwargs):
        print("CHAIN END", outputs)


callback_logger = SimpleLoggerCallback()

# Call with callbacks:
# basic_chain.invoke({"input": "hello"}, config={"callbacks": [callback_logger]})


# =============================================================================
# 19) Guardrails and safety patterns
# =============================================================================

"""
Practical safety controls:
- System prompt constraints and refusal policy
- Output schema validation
- Retrieval grounding and citation requirement
- Blocklists / regex checks for outputs
- Human approval for high-risk actions
"""


# =============================================================================
# 20) Error handling, retries, and fallbacks
# =============================================================================

primary_llm = ChatOpenAI(model="gpt-4o", temperature=0)
fallback_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def answer_with_fallback(question: str) -> str:
    try:
        chain = basic_prompt | primary_llm | str_parser
        return chain.invoke({"input": question})
    except Exception:
        chain = basic_prompt | fallback_llm | str_parser
        return chain.invoke({"input": question})


# =============================================================================
# 21) Caching and performance notes
# =============================================================================

"""
Performance checklist:
- Cache embeddings for unchanged documents.
- Store chunk hashes and only re-index changed chunks.
- Use top-k and chunk-size tuning together.
- Use MMR or reranking for better relevance.
- Use async/streaming for user-perceived latency.
"""


# =============================================================================
# 22) Evaluation and regression testing
# =============================================================================

"""
Build eval datasets with:
- query
- expected answer style
- required facts/citations

Track over time:
- correctness
- groundedness
- latency
- cost
"""


# =============================================================================
# 23) Minimal FastAPI integration pattern
# =============================================================================

"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
chain = create_retrieval_chain(retriever_similarity, stuff_chain)

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: AskRequest):
    out = chain.invoke({"input": req.question})
    return {"answer": out.get("answer", ""), "context": out.get("context", [])}
"""


# =============================================================================
# 24) Common mistakes and direct fixes
# =============================================================================

"""
1) Prompt has {context} but no {input}
   Fix: include both placeholders when user query is required.

2) Using agent for a deterministic flow
   Fix: use explicit chains first; add agents only for dynamic tool routing.

3) No metadata in documents
   Fix: add source/title/chunk_id for better filtering and citations.

4) No chunk overlap
   Fix: add overlap to preserve meaning across chunk boundaries.

5) Returning raw AIMessage in API response
   Fix: parse to string/JSON schema before returning.

6) Not versioning prompts
   Fix: store prompt versions and compare evals before deployment.
"""


# =============================================================================
# 25) Copy-paste builders (quick reuse)
# =============================================================================

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant.
Use only the context below. If insufficient, say "I do not know".

Context:
{context}

Question:
{input}
"""


def build_basic_rag_chain(vectorstore, model_name: str = "gpt-4o", k: int = 4):
    """Basic, production-friendly retrieval chain builder."""
    model = ChatOpenAI(model=model_name, temperature=0)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    doc_chain = create_stuff_documents_chain(model, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return create_retrieval_chain(retriever, doc_chain)


def build_compressed_rag_chain(vectorstore, model_name: str = "gpt-4o", k: int = 8):
    """RAG with MMR + contextual compression for better precision."""
    model = ChatOpenAI(model=model_name, temperature=0)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    doc_chain = create_stuff_documents_chain(model, prompt)

    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 30},
    )

    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor

        compressor = LLMChainExtractor.from_llm(model)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    except Exception:
        retriever = base_retriever

    return create_retrieval_chain(retriever, doc_chain)


# =============================================================================
# 26) Practice roadmap (in order)
# =============================================================================

"""
Phase 1: Prompt -> LLM -> Parser (no retrieval)
Phase 2: Loaders + Splitters + Embeddings + Vector store
Phase 3: Retriever + Stuff documents chain + Retrieval chain
Phase 4: MMR / threshold / compression retrievers
Phase 5: Structured output + session history
Phase 6: Tools + simple agent
Phase 7: Evaluation, tracing, and production hardening
"""


if __name__ == "__main__":
    print("Comprehensive LangChain production cheat sheet ready.")
