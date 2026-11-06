"""Offline tests: no API keys or network needed (fake embeddings, no LLM)."""
import io

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from pypdf import PdfWriter

from agent import ingest
from agent.tools import calculator, make_search_tool, safe_eval


def make_pdf(text: str) -> bytes:
    # Minimal one-page PDF with real extractable text.
    stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, x)
    return out


@pytest.fixture
def store(tmp_path):
    return Chroma(collection_name="test_docs", embedding_function=DeterministicFakeEmbedding(size=32),
                  persist_directory=str(tmp_path))


def test_calculator():
    assert safe_eval("(1250*0.18)/12") == pytest.approx(18.75)
    assert calculator.invoke({"expression": "sqrt(16)+2**3"}) == "12.0"
    assert calculator.invoke({"expression": "1/0"}).startswith("Error")
    assert calculator.invoke({"expression": "__import__('os').system('ls')"}).startswith("Error")


def test_ingest_and_search(store):
    r = ingest.ingest_pdf(make_pdf("The warranty period is 24 months."), "a.pdf", store)
    assert r.chunks >= 1 and r.pages == 1
    assert ingest.ingest_pdf(make_pdf("The warranty period is 24 months."), "a.pdf", store).already_indexed
    assert ingest.list_documents(store) == {r.doc_id: "a.pdf"}

    search = make_search_tool(store, lambda: [r.doc_id])
    assert "warranty" in search.invoke({"query": "warranty"})
    assert "upload a PDF" in make_search_tool(store, lambda: []).invoke({"query": "x"})

    ingest.delete_document(store, r.doc_id)
    assert ingest.list_documents(store) == {}


def test_bad_files(store):
    with pytest.raises(ingest.IngestError):
        ingest.ingest_pdf(b"", "e.pdf", store)
    with pytest.raises(ingest.IngestError):
        ingest.ingest_pdf(b"not a pdf at all", "x.pdf", store)
    blank = io.BytesIO()
    w = PdfWriter(); w.add_blank_page(100, 100); w.write(blank)
    with pytest.raises(ingest.IngestError, match="scanned"):
        ingest.ingest_pdf(blank.getvalue(), "blank.pdf", store)
