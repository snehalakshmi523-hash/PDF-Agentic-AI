"""Streamlit frontend: upload PDFs -> indexed into Chroma (RAG) -> chat with the agent."""
import logging
import uuid

import streamlit as st

from agent import config
from agent.agent import ask, build_agent
from agent.ingest import IngestError, delete_document, get_store, ingest_pdf, list_documents

logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="PDF Agent", page_icon="📄", layout="wide")

st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, .stApp, button, input, textarea {font-family: 'Inter', system-ui, sans-serif !important;}
    .stApp {background: linear-gradient(180deg, #EEF0FF 0%, #F7F8FC 260px);}
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden; height: 0;}
    header[data-testid="stHeader"] {background: transparent;}
    .block-container {max-width: 860px; padding-top: 1.5rem; padding-bottom: 6rem;}

    /* hero */
    .hero {padding: 1.2rem 0 0.4rem;}
    .hero h1 {font-size: 2.3rem; font-weight: 700; letter-spacing: -0.03em; margin: 0;
              background: linear-gradient(90deg, #4F46E5, #0EA5E9); -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;}
    .hero p {color: #5B6478; font-size: 1.02rem; margin: 0.35rem 0 0;}
    .chips span {display: inline-block; margin: 0.7rem 0.4rem 0 0; padding: 0.25rem 0.7rem;
                 border-radius: 999px; background: #fff; border: 1px solid #E3E6F3;
                 color: #4F46E5; font-size: 0.8rem; font-weight: 500;}

    /* sidebar */
    [data-testid="stSidebar"] {background: #fff; border-right: 1px solid #E8EAF4;}
    [data-testid="stSidebar"] h2 {font-size: 1.05rem; font-weight: 600;}
    [data-testid="stFileUploaderDropzone"] {background: #F5F6FF; border: 1.5px dashed #B9BFFF; border-radius: 14px;}

    /* chat bubbles */
    [data-testid="stChatMessage"] {background: #fff; border: 1px solid #E8EAF4; border-radius: 16px;
                                   padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
                                   box-shadow: 0 1px 2px rgba(20,24,60,.04);}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {background: #EEF0FF; border-color: #DADFFF;}
    [data-testid="stChatInput"] {border-radius: 16px; box-shadow: 0 6px 24px rgba(79,70,229,.12);}

    /* buttons */
    .stButton > button {border-radius: 10px; border: 1px solid #DADFFF; background: #fff; font-weight: 500;}
    .stButton > button:hover {border-color: #4F46E5; color: #4F46E5;}
    </style>""",
    unsafe_allow_html=True,
)


def friendly_error(e: Exception) -> str:
    """Translate provider/network exceptions into something a user can act on."""
    name, text = type(e).__name__, str(e).lower()
    if isinstance(e, config.ConfigError):
        return str(e)
    if config.LLM_PROVIDER == "ollama" and ("connect" in name.lower() or "refused" in text or "11434" in text):
        return ("Can't reach Ollama. Install it from ollama.com, start it, and run "
                f"`ollama pull {config.LLM_MODEL}`.")
    if "not found" in text and "model" in text:
        return f"The model isn't available. If you use Ollama, run `ollama pull {config.LLM_MODEL}`."
    if "authentication" in name.lower() or "api key" in text or "401" in text:
        return "The API key was rejected. Check the keys in your .env file."
    if "ratelimit" in name.lower() or "429" in text or "quota" in text:
        return "The model provider is rate-limiting requests or your quota is used up. Try again in a moment."
    if "timeout" in name.lower() or "connection" in name.lower():
        return "Couldn't reach the model provider. Check your connection and try again."
    if "recursion" in name.lower():
        return "The agent took too many steps on that question. Try asking something more specific."
    return f"Something went wrong ({name}). Try again, or rephrase the question."


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner=False)
def get_vector_store():
    config.require_keys()
    return get_store()


try:
    store = get_vector_store()
except Exception as e:
    st.error(friendly_error(e))
    st.stop()

ss = st.session_state
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("messages", [])
ss.setdefault("selected", None)  # None = "use everything currently indexed"
ss.setdefault("uploader_key", 0)

docs = list_documents(store)
if ss.selected is None:
    ss.selected = list(docs)
ss.selected = [d for d in ss.selected if d in docs]


@st.cache_resource(show_spinner=False)
def get_agent_and_memory():
    # One agent per server process; per-user memory is separated by thread_id.
    # The selected documents are passed in on every ask() call.
    return build_agent(store)


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True, key=f"up{ss.uploader_key}",
        help="Text is extracted, split into chunks, embedded and stored in the vector database.",
    )
    if uploads:
        for f in uploads:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    r = ingest_pdf(f.getvalue(), f.name, store)
                except IngestError as e:
                    st.error(f"{f.name}: {e}")
                    continue
                except Exception as e:
                    st.error(f"{f.name}: {friendly_error(e)}")
                    continue
            if r.already_indexed:
                st.info(f"{f.name} is already indexed.")
            else:
                st.success(f"{f.name}: {r.pages} pages, {r.chunks} chunks indexed.")
            if r.doc_id not in ss.selected:
                ss.selected.append(r.doc_id)
        ss.uploader_key += 1  # clear the widget so files aren't re-processed on each rerun
        docs = list_documents(store)

    if docs:
        st.caption("Questions are answered from the checked documents.")
        chosen = []
        for doc_id, name in docs.items():
            c1, c2 = st.columns([5, 1])
            if c1.checkbox(name, value=doc_id in ss.selected, key=f"sel-{doc_id}"):
                chosen.append(doc_id)
            if c2.button("✕", key=f"del-{doc_id}", help=f"Remove {name}"):
                delete_document(store, doc_id)
                ss.selected = [d for d in ss.selected if d != doc_id]
                st.rerun()
        ss.selected = chosen
    else:
        st.caption("No documents yet. Upload a PDF to start asking questions.")

    st.divider()
    if st.button("New conversation"):
        ss.thread_id, ss.messages = str(uuid.uuid4()), []
        st.rerun()
    st.caption(f"🤖 {config.LLM_PROVIDER} · {config.LLM_MODEL}")

# ----------------------------------------------------------------------- chat
st.markdown(
    """<div class="hero"><h1>Chat with your PDFs</h1>
    <p>Upload a document, then ask questions, get cited answers, run calculations or check the weather.</p>
    <div class="chips"><span>📄 PDF search (RAG)</span><span>🧠 Remembers context</span>
    <span>🧮 Calculator</span><span>⛅ Weather</span></div></div>""",
    unsafe_allow_html=True,
)
st.write("")
if not docs:
    st.info("👈 Upload a PDF in the sidebar to begin. You can also ask for the weather or a calculation.")
elif not ss.messages:
    st.caption("Try one of these:")
    cols = st.columns(3)
    for col, text in zip(cols, ["Summarise this document", "List the main topics", "Write the answer for Question 1"]):
        if col.button(text, use_container_width=True):
            ss.pending = text
            st.rerun()

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

pending = ss.pop("pending", None)
if prompt := (st.chat_input("Ask about your documents…") or pending):
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = ask(get_agent_and_memory(), prompt, ss.thread_id, list(ss.selected))
            except Exception as e:
                logging.exception("Agent call failed")
                answer = f"⚠️ {friendly_error(e)}"
        st.markdown(answer)
    ss.messages.append({"role": "assistant", "content": answer})