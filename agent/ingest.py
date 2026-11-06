"""PDF ingestion: extract text -> chunk -> embed -> persist in Chroma (the RAG store)."""
from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from . import config

log = logging.getLogger(__name__)
# Separate collection per embedding model: vectors of different models/sizes must never be mixed.
COLLECTION = "pdf_" + re.sub(r"[^a-zA-Z0-9]+", "_", f"{config.EMBEDDING_PROVIDER}_{config.EMBEDDING_MODEL}").strip("_")[:60]


class IngestError(Exception):
    """A user-facing problem with the uploaded file."""


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    pages: int
    chunks: int
    already_indexed: bool = False


class FastEmbedEmbeddings(Embeddings):
    """Free, local embeddings (ONNX, CPU only, no API key). Model downloads once on first use."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def get_embeddings() -> Embeddings:
    if config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    return FastEmbedEmbeddings(config.EMBEDDING_MODEL)


def get_store(embeddings: Embeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def list_documents(store: Chroma) -> dict[str, str]:
    """Return {doc_id: filename} for everything in the store."""
    data = store.get(include=["metadatas"])
    return {m["doc_id"]: m["filename"] for m in data["metadatas"] or [] if m}


def _extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            # Many PDFs are "encrypted" with an empty password; try that first.
            if not reader.decrypt(""):
                raise IngestError("This PDF is password-protected. Remove the password and upload it again.")
        pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
    except IngestError:
        raise
    except (PyPdfError, ValueError, OSError) as e:
        raise IngestError(f"Could not read this file as a PDF ({e}).") from e
    return pages


def ingest_pdf(file_bytes: bytes, filename: str, store: Chroma) -> IngestResult:
    if not file_bytes:
        raise IngestError("The uploaded file is empty.")
    if len(file_bytes) > config.MAX_PDF_MB * 1024 * 1024:
        raise IngestError(f"File is larger than {config.MAX_PDF_MB} MB.")

    doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]

    # Idempotent: re-uploading the same file doesn't create duplicate chunks.
    existing = store.get(where={"doc_id": doc_id}, limit=1)
    if existing["ids"]:
        pages = existing["metadatas"][0].get("total_pages", 0)
        return IngestResult(doc_id, filename, pages, 0, already_indexed=True)

    pages = _extract_pages(file_bytes)
    if not any(text for _, text in pages):
        raise IngestError(
            "No text could be extracted. This looks like a scanned PDF; run OCR on it first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    docs: list[Document] = []
    for page_no, text in pages:
        if not text:
            continue
        for chunk in splitter.split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page_no,
                        "total_pages": len(pages),
                    },
                )
            )

    try:
        store.add_documents(docs, ids=[f"{doc_id}-{i}" for i in range(len(docs))])
    except Exception as e:  # embedding API / network / vector-store failures
        log.exception("Indexing failed")
        raise IngestError(f"Indexing failed: {e}") from e

    return IngestResult(doc_id, filename, len(pages), len(docs))


def delete_document(store: Chroma, doc_id: str) -> None:
    store.delete(where={"doc_id": doc_id})
