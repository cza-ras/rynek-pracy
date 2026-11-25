"""Unit tests for the job_listing_count workflow: parsing, activities, and registration."""

import httpx
import httpx2
import pytest
from support.http_clients import patch_http_send
from worker.workflows import job_listing_count as jlc

PAGE_HTML = """
<html><head><title>Praca bi developer – Pracuj.pl - 33 oferty pracy</title></head>
<body><h1>Praca: bi developer</h1><span>33 oferty pracy</span></body></html>
"""


def test_extract_listing_count_from_title_tag() -> None:
    assert jlc.extract_listing_count("<title>Praca bi developer – Pracuj.pl - 33 oferty pracy</title>") == 33


def test_extract_listing_count_from_body_phrase() -> None:
    assert jlc.extract_listing_count("<span>33 oferty pracy</span>") == 33


def test_extract_listing_count_singular_and_plural_and_nbsp() -> None:
    assert jlc.extract_listing_count("1 oferta pracy") == 1
    assert jlc.extract_listing_count("24 oferty pracy") == 24
    assert jlc.extract_listing_count("5 ofert pracy") == 5
    assert jlc.extract_listing_count("1 234 oferty pracy") == 1234
    assert jlc.extract_listing_count("1 234 oferty pracy") == 1234


def test_extract_listing_count_ignores_salary_numbers() -> None:
    html = "12 500 zł netto / mies. <span>33 oferty pracy</span>"
    assert jlc.extract_listing_count(html) == 33


def test_extract_listing_count_missing() -> None:
    assert jlc.extract_listing_count("<html><body>Nic nie znaleziono</body></html>") is None


def test_keyword_from_url() -> None:
    assert jlc.keyword_from_url("https://www.pracuj.pl/praca/bi%20developer") == "bi developer"
    assert jlc.keyword_from_url("https://www.pracuj.pl/praca/data%20engineer?rd=30") == "data engineer"


def test_record_document_name_and_markdown() -> None:
    name = jlc.record_document_name("https://www.pracuj.pl/praca/bi%20developer", "2026-11-25")
    assert name == "liczba-ogloszen-bi-developer-2026-11-25.md"
    text = jlc.record_markdown("https://www.pracuj.pl/praca/bi%20developer", 33, "2026-11-25")
    assert "Liczba ofert: 33" in text
    assert "Data: 2026-11-25" in text
    assert "bi developer" in text


def test_library_auth_headers_from_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_LIBRARY_API_KEY", "user-key-test")

    headers = jlc.library_auth_headers()

    assert headers["Authorization"] == "Bearer user-key-test"


def test_library_auth_headers_fall_back_to_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_LIBRARY_API_KEY", raising=False)
    monkeypatch.setenv("MISTRAL_API_KEY", "fallback-key")

    headers = jlc.library_auth_headers()

    assert headers["Authorization"] == "Bearer fallback-key"


def test_library_auth_headers_require_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_LIBRARY_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MISTRAL_LIBRARY_API_KEY"):
        jlc.library_auth_headers()


async def test_fetch_listing_count_parses_page(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, httpx.URL] = {}

    async def send(self: object, request: httpx2.Request, **kwargs: object) -> httpx2.Response:
        captured["url"] = request.url
        return httpx2.Response(200, text=PAGE_HTML, request=request)

    patch_http_send(monkeypatch, send)

    result = await jlc.fetch_listing_count("https://www.pracuj.pl/praca/bi%20developer")

    assert result.count == 33
    assert result.url == "https://www.pracuj.pl/praca/bi%20developer"
    assert str(captured["url"]) == "https://www.pracuj.pl/praca/bi%20developer"


async def test_fetch_listing_count_raises_when_count_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def send(self: object, request: object, **kwargs: object) -> httpx2.Response:
        return httpx2.Response(200, text="<html><body>brak wyników</body></html>", request=request)

    patch_http_send(monkeypatch, send)

    with pytest.raises(ValueError, match="No job-listing count"):
        await jlc.fetch_listing_count("https://www.pracuj.pl/praca/bi%20developer")


def _library_api_transport(monkeypatch: pytest.MonkeyPatch, existing: list[dict]) -> list[dict]:
    """Stand in for the Mistral Libraries API: every request is answered in-process."""
    calls: list[dict] = []

    async def send(self: object, request: httpx2.Request, **kwargs: object) -> httpx2.Response:
        calls.append({"method": request.method, "url": str(request.url), "request": request})
        if request.method == "GET":
            return httpx2.Response(200, json={"data": existing}, request=request)
        if request.method == "POST" and "/documents" in str(request.url):
            return httpx2.Response(200, json={"id": "doc_1"}, request=request)
        return httpx2.Response(200, json={"id": "lib_new", "name": "Rynek pracy"}, request=request)

    patch_http_send(monkeypatch, send)
    return calls


async def test_store_listing_count_reuses_library_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_LIBRARY_API_KEY", "user-key-test")
    calls = _library_api_transport(monkeypatch, existing=[{"id": "lib_existing", "name": "Rynek pracy"}])

    stored = await jlc.store_listing_count(
        "https://www.pracuj.pl/praca/bi%20developer", 33, "2026-11-25", "Rynek pracy"
    )

    assert stored.library_id == "lib_existing"
    assert stored.document_id == "doc_1"
    assert [call["method"] for call in calls] == ["GET", "POST"]
    assert calls[0]["request"].headers["Authorization"] == "Bearer user-key-test"


async def test_store_listing_count_creates_library_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_LIBRARY_API_KEY", "user-key-test")
    _library_api_transport(monkeypatch, existing=[])

    stored = await jlc.store_listing_count(
        "https://www.pracuj.pl/praca/bi%20developer", 33, "2026-11-25", "Rynek pracy"
    )

    assert stored.library_id == "lib_new"
    assert stored.document_name == "liczba-ogloszen-bi-developer-2026-11-25.md"


def test_workflow_class_registered() -> None:
    import mistralai.workflows as workflows

    definition = workflows.get_workflow_definition(jlc.JobListingCount)
    assert definition.name == "job_listing_count"
