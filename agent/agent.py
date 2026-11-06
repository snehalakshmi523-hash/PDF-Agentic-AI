"""Agent construction (LangChain 1.x `create_agent`, which runs on LangGraph)."""
from __future__ import annotations

from contextvars import ContextVar
from typing import Callable, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ToolCallLimitMiddleware
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from . import config
from .tools import calculator, get_weather, make_search_tool

SYSTEM_PROMPT = """You are a helpful, beginner-friendly PDF research assistant.

How to work:
1. For anything about the uploaded PDF(s), call `search_document` with a short, focused query.
   - If the user asks for a numbered item ("first question", "question 3", "section 2"), search for
     that wording, e.g. "Question 1" or "1." plus the topic.
   - You may search again with a DIFFERENT query only if the first result clearly lacks what you need.
     Never repeat the same query, and never search more than 3 times for one question.
2. As soon as you have passages, write the final answer. Do not keep searching.
3. Base facts about the document only on the returned passages and cite them as (filename, p.N).
   If the passages don't contain the answer, say so plainly. Never invent document content.
4. If the document contains a question or exercise (e.g. an exam paper) and the user asks you to
   answer it: first quote or restate the question briefly, then answer it using your general knowledge,
   and say that the answer comes from general knowledge, not from the PDF.
5. Only when asked to summarise or give an overview of the whole document, structure the answer as:
   overview, main topics, key points, conclusion. For specific questions, answer directly and briefly.
6. Use `calculator` only for calculations and `get_weather` only for weather questions.
7. If a tool returns an error, explain it to the user instead of pretending it worked.
8. Use the conversation history to understand follow-up questions."""


# Which documents to search for the CURRENT question. Tools run in worker threads where Streamlit's
# session_state is not readable, so the UI passes the selection through ask() into this variable
# (LangChain copies the context into its worker threads).
_selected_docs: ContextVar[tuple] = ContextVar("selected_docs", default=())


def build_llm():
    config.require_keys()
    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.LLM_MODEL, temperature=0, timeout=60, max_retries=2)
    if config.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=config.LLM_MODEL, temperature=0, timeout=60, max_retries=2)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.LLM_MODEL, temperature=0, timeout=60, max_retries=2)


def build_agent(store: Chroma, get_selected_docs: Callable[[], Sequence[str]] | None = None, checkpointer=None):
    llm = build_llm()
    getter = get_selected_docs or (lambda: _selected_docs.get())
    tools = [make_search_tool(store, getter), calculator, get_weather]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        # Short-term memory: the checkpointer persists the message history per thread_id.
        checkpointer=checkpointer or InMemorySaver(),
        # Keeps long chats inside the context window by summarising old turns.
        middleware=[
            # Hard stop for search loops: after 3 searches per question the extra call is refused
            # and the model is forced to answer with what it already has.
            ToolCallLimitMiddleware(tool_name="search_document", run_limit=3, exit_behavior="continue"),
            SummarizationMiddleware(model=llm, trigger=("tokens", 6000), keep=("messages", 12)),
        ],
    )


def ask(agent, question: str, thread_id: str, selected_docs: Sequence[str] | None = None) -> str:
    """Run one turn. thread_id selects the conversation memory; selected_docs the PDFs to search."""
    if selected_docs is not None:
        _selected_docs.set(tuple(selected_docs))
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 25},
        )
    except GraphRecursionError:
        return ("I couldn't find a clear answer in the document for that. "
                "Try asking with the exact wording, page, or section you mean.")

    content = result["messages"][-1].content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content.strip() or "I couldn't generate an answer. Please try rephrasing your question."