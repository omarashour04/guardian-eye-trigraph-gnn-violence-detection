import pytest

from rag_service.legal_scraper import (
    RawLegalDocument,
    scrape_legal_source,
    scrape_legal_sources,
)
from rag_service.legal_sources import LEGAL_SOURCE_REGISTRY


MOCK_HTML_SOURCE = {
    "country": "UK",
    "source_url": "https://example.test/legal-html",
    "law_title": "Mock HTML Law",
    "source_language": "en",
    "source_type": "html",
    "official_source": True,
}

MOCK_PDF_SOURCE = {
    "country": "Canada",
    "source_url": "https://example.test/legal.pdf",
    "law_title": "Mock PDF Law",
    "source_language": "en",
    "source_type": "pdf",
    "official_source": True,
}


def _tiny_pdf_bytes(text):
    return f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 72 720 Td ({text}) Tj ET
endstream
endobj
trailer
<< /Root 1 0 R >>
%%EOF
""".encode("latin-1")


class FailingSession:
    def get(self, url, timeout):
        raise RuntimeError("network disabled")


def test_html_extraction_works_from_mocked_html():
    document = scrape_legal_source(
        MOCK_HTML_SOURCE,
        content="""
        <html>
          <head><style>.hidden { display: none; }</style></head>
          <body>
            <main>
              <h1>Mock Legal Title</h1>
              <p>Violent conduct may have legal consequences.</p>
            </main>
            <script>ignored()</script>
          </body>
        </html>
        """,
    )

    assert document.extraction_status == "success"
    assert "Mock Legal Title" in document.raw_text
    assert "Violent conduct may have legal consequences." in document.raw_text
    assert "ignored" not in document.raw_text
    assert document.extraction_method.startswith("html_")


def test_pdf_extraction_works_from_mocked_pdf_bytes():
    document = scrape_legal_source(
        MOCK_PDF_SOURCE,
        content=_tiny_pdf_bytes("Mock PDF legal text"),
    )

    assert document.extraction_status == "success"
    assert "Mock PDF legal text" in document.raw_text
    assert document.extraction_method.startswith("pdf_")


def test_failed_download_returns_failed_status_and_error_message():
    document = scrape_legal_source(
        MOCK_HTML_SOURCE,
        session=FailingSession(),
        retries=1,
        timeout=1,
    )

    assert document.extraction_status == "failed"
    assert document.raw_text == ""
    assert document.error_message


def test_scraper_preserves_metadata_from_legal_source_registry():
    source = LEGAL_SOURCE_REGISTRY["UAE"][0]
    document = scrape_legal_source(source, content="<html><body>UAE legal text</body></html>")

    assert document.country == source["country"]
    assert document.source_url == source["source_url"]
    assert document.law_title == source["law_title"]
    assert document.source_language == source["source_language"]
    assert document.source_type == source["source_type"]
    assert document.official_source == source["official_source"]
    assert isinstance(document, RawLegalDocument)


def test_no_network_calls_are_required_for_tests():
    documents = scrape_legal_sources(
        sources=[MOCK_HTML_SOURCE],
        content="<html><body>local fixture text</body></html>",
    )

    assert len(documents) == 1
    assert documents[0].extraction_status == "success"
    assert "local fixture text" in documents[0].raw_text
