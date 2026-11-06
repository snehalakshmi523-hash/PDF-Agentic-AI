# PDF Agent: chat with your documents (LangChain + Streamlit)

An AI agent that answers questions about uploaded PDFs, remembers the conversation, and can call
external tools (calculator, live weather).

## Quick start (100% free, runs locally, no API keys)

1. Install [Ollama](https://ollama.com) and download a model that supports tool calling:
   ```bash
   ollama pull llama3.1:8b
   ```
   Keep Ollama running. It starts automatically on most systems.
2. Set up the project:
   ```bash
   python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env                                    # Windows: copy .env.example .env
   streamlit run app.py
   ```
3. Open http://localhost:8501, upload a PDF in the sidebar, and start asking questions.

The first PDF upload downloads a small embedding model (about 130 MB, once). After that everything runs offline,
except the weather tool.

**Other providers** (edit `.env`): `LLM_PROVIDER=groq` (free hosted tier, needs `GROQ_API_KEY`),
`openai` or `anthropic` (paid). `EMBEDDING_PROVIDER=openai` is also supported.
Each embedding model gets its own vector collection, so switching never mixes incompatible vectors.

Try: "Summarise section 2", then "What does it say about penalties?" (follow-up), then
"Add 18% to the total price in the document" (RAG plus calculator), or "Weather in Kochi?".

Run the offline tests (no model or network needed): `pytest -q`

## Requirements checklist

| Requirement | Where |
|---|---|
| Answer questions about a PDF | `agent/ingest.py` (PDF -> chunks -> Chroma) and the `search_document` tool |
| Remember conversation context | LangGraph checkpointer keyed by `thread_id` plus summarisation middleware (`agent/agent.py`) |
| At least one external tool | `calculator` and `get_weather` (Open-Meteo, no key needed) in `agent/tools.py` |
| Error handling | See below |
| Frontend with PDF upload, saved to RAG, chat | `app.py` (Streamlit, custom-styled; theme in `.streamlit/config.toml`) |

## Architecture

```
Streamlit UI ──upload──> ingest_pdf ──> pypdf -> text splitter -> local embeddings (fastembed) -> Chroma (./data/chroma)
     │
     └─chat──> agent (create_agent, tool-calling loop, InMemorySaver memory)
                  ├─ search_document  -> Chroma similarity search (filtered to checked docs)
                  ├─ calculator       -> AST-whitelisted evaluator
                  └─ get_weather      -> Open-Meteo HTTP API
```

## Design choices

- **RAG as a tool (agentic RAG), not a fixed pipeline.** The model decides when to search. A greeting,
  a calculation or a weather question doesn't trigger a pointless retrieval, and the model can
  rewrite a follow-up like "and the penalties?" into a standalone query before searching.
  The cost is one extra LLM round-trip compared with always-retrieve.
- **Chroma with persistence.** Uploaded documents are embedded once and survive restarts
  ("saved to RAG"). Documents are identified by a content hash, so re-uploading the same file is a
  no-op, and each chunk carries `filename` and `page` metadata, which gives page-level citations.
- **Chunking:** 1000 characters with 150 overlap, page by page. This is small enough for precise
  retrieval and large enough to keep a paragraph's meaning together. Both values are in `config.py`.
- **Memory:** a LangGraph checkpointer stores the message history per `thread_id` (one per browser
  session; "New conversation" starts a fresh one). `SummarizationMiddleware` compresses older turns
  once the history gets long, so chats don't overflow the context window.
- **Tools:** the calculator uses an AST whitelist rather than `eval()`, so it is safe against code
  injection, and it exists because LLMs make arithmetic mistakes. The weather tool uses the free Open-Meteo
  API, so it needs no API key.
- **Grounding:** the system prompt requires facts about the document to come only from retrieved passages,
  with `(filename, p.N)` citations, and to say "not found" when the passages don't contain the answer.
  One deliberate exception: if the PDF contains a question (e.g. an exam paper) and the user asks for
  its answer, the agent finds the question in the PDF and answers it from general knowledge, saying so.
- **Loop protection:** small models sometimes search repeatedly. A `ToolCallLimitMiddleware` allows at most
  3 `search_document` calls per question and then forces the model to answer with what it has.
- **Thread-safe document selection:** tools run in worker threads where Streamlit's `session_state` is not
  readable. The UI passes the checked documents into `ask()`, which stores them in a `ContextVar` that
  the search tool reads, so each question searches exactly the documents ticked in the sidebar.
- **UI:** Streamlit with custom CSS (hero header, card-style chat bubbles, clickable example questions)
  and a theme in `.streamlit/config.toml`. The UI holds no agent logic, so it can be swapped for another frontend.
- **Free by default, provider-agnostic:** local fastembed embeddings plus an Ollama chat model, switchable in `.env`. The rest of the code only sees a LangChain chat model.

## Error handling

- **Upload:** empty file, corrupt or non-PDF file, password-protected PDF, oversized file, scanned PDF
  with no extractable text (with a clear "run OCR" message), and embedding or API failures, all
  reported per file without crashing the app.
- **Tools:** every tool catches its own exceptions and returns an `Error: ...` string. The model
  reads it and explains the problem, instead of the whole run failing. HTTP calls have timeouts.
- **LLM calls:** client-level timeout and retries, a search-call limit and a `recursion_limit` to stop
  runaway tool loops (with a friendly fallback message if a limit is hit), and a UI layer that maps auth,
  rate-limit, timeout and Ollama-connection errors to actionable messages.
- **Config:** missing API keys are detected at startup with a clear message.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Can't reach Ollama" | Start Ollama and run `ollama pull llama3.1:8b` |
| 401 / key rejected | Check the key in `.env` (the file must be named `.env`, next to `app.py`), then restart Streamlit |
| Slow answers | Local models on CPU are slow; use `LLM_PROVIDER=groq` (free key) for speed |
| "I couldn't find a clear answer" | Ask with the exact wording, number or section used in the PDF; click **New conversation** |
| "No text could be extracted" | The PDF is a scan; run OCR on it first |
| Fresh start | Delete the `data/` folder to remove all stored documents |

## Limitations

- Memory is in-process (`InMemorySaver`), so it is lost when the server restarts.
- Scanned PDFs need OCR first. Tables and figures are extracted as plain text only.
- All users of one server share the document store (the selection checkboxes are per session).
- Small local models are less reliable at tool use than larger hosted ones.

## How I would extend it for more complex tasks

1. **Better retrieval:** hybrid search (BM25 + vectors), a cross-encoder reranker, and a
   query-rewriting step. Add multimodal or OCR ingestion (Unstructured, Azure Document Intelligence) so tables, scans and
   charts are covered. Evaluate with RAGAS or LangSmith datasets.
2. **Durable, multi-user state:** swap `InMemorySaver` for the Postgres or SQLite checkpointer, add
   authentication, and give each user their own Chroma collection (or move to pgvector or Qdrant).
3. **More tools:** web search, SQL over structured data, code execution in a sandbox, email or
   calendar actions, with human-in-the-loop approval (`interrupt_before`) on anything with side effects.
4. **Multi-agent workflows:** a supervisor that routes to specialists (e.g. a "contract reviewer" and a
   "data analyst") built as LangGraph sub-graphs, or plan-and-execute for long multi-step tasks.
5. **Long-term memory:** a LangGraph `Store` for user preferences and facts that persist across threads.
6. **Streaming and observability:** stream tokens to the UI, show tool calls as they happen, and trace
   everything with LangSmith. Add guardrails (prompt-injection filtering of retrieved text, PII redaction),
   caching and cost limits.
7. **Deployment:** containerise it, put FastAPI in front of the agent, and keep Streamlit (or a React app)
   as a thin client.