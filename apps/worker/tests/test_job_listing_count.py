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


def _install_fake_mistral_client(
    monkeypatch: pytest.MonkeyPatch,
    libraries: dict,
) -> dict:
    from types import SimpleNamespace

    fake = SimpleNamespace(
        beta=SimpleNamespace(
            libraries=SimpleNamespace(
                list_async=libraries["list_async"],
                create_async=libraries["create_async"],
                documents=SimpleNamespace(upload_async=libraries["upload_async"]),
            )
        )
    )
    monkeypatch.setattr(
        jlc.workflows_mistralai, "get_mistral_client", lambda *args, **kwargs: fake
    )
    return libraries


def test_extract_listing_count_from_title_tag() -> None:
    assert (
        jlc.extract_listing_count(
            "<title>Praca bi developer – Pracuj.pl - 33 oferty pracy</title>"
        )
        == 33
    )


def test_extract_listing_count_from_body_phrase() -> None:
    assert jlc.extract_listing_count("<span>33 oferty pracy</span>") == 33


def test_extract_listing_count_singular_and_plural_and_nbsp() -> None:
    assert jlc.extract_listing_count("1 oferta pracy") == 1
    assert jlc.extract_listing_count("24 oferty pracy") == 24
    assert jlc.extract_listing_count("5 ofert pracy") == 5
    assert jlc.extract_listing_count("1\u00a0234 oferty pracy") == 1234
    assert jlc.extract_listing_count("1\u202f234 oferty pracy") == 1234


def test_extract_listing_count_ignores_salary_numbers() -> None:
    html = "12\u00a0500 zł netto / mies. <span>33 oferty pracy</span>"
    assert jlc.extract_listing_count(html) == 33


def test_extract_listing_count_missing() -> None:
    assert (
        jlc.extract_listing_count("<html><body>Nic nie znaleziono</body></html>")
        is None
    )


def test_keyword_from_url() -> None:
    assert (
        jlc.keyword_from_url("https://www.pracuj.pl/praca/bi%20developer")
        == "bi developer"
    )
    assert (
        jlc.keyword_from_url("https://www.pracuj.pl/praca/data%20engineer?rd=30")
        == "data engineer"
    )


def test_record_document_name_and_markdown() -> None:
    name = jlc.record_document_name(
        "https://www.pracuj.pl/praca/bi%20developer", "2026-11-25"
    )
    assert name == "liczba-ogloszen-bi-developer-2026-11-25.md"
    text = jlc.record_markdown(
        "https://www.pracuj.pl/praca/bi%20developer", 33, "2026-11-25"
    )
    assert "Liczba ofert: 33" in text
    assert "Data: 2026-11-25" in text
    assert "bi developer" in text


async def test_fetch_listing_count_parses_page(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, httpx.URL] = {}

    async def send(
        self: httpx2.AsyncClient, request: httpx2.Request, **kwargs: object
    ) -> httpx2.Response:
        captured["url"] = request.url
        return httpx2.Response(200, text=PAGE_HTML, request=request)

    patch_http_send(monkeypatch, send)

    result = await jlc.fetch_listing_count("https://www.pracuj.pl/praca/bi%20developer")

    assert result.count == 33
    assert result.url == "https://www.pracuj.pl/praca/bi%20developer"
    assert captured["url"] == "https://www.pracuj.pl/praca/bi%20developer"


async def test_fetch_listing_count_raises_when_count_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def send(self: object, request: object, **kwargs: object) -> httpx2.Response:
        return httpx2.Response(
            200, text="<html><body>brak wyników</body></html>", request=request
        )

    patch_http_send(monkeypatch, send)

    with pytest.raises(ValueError, match="No job-listing count"):
        await jlc.fetch_listing_count("https://www.pracuj.pl/praca/bi%20developer")


async def test_store_listing_count_reuses_library_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    calls: dict = {"created": [], "uploads": []}

    async def list_async(**kwargs: object) -> object:
        return SimpleNamespace(
            data=[SimpleNamespace(id="lib_existing", name="Rynek pracy")]
        )

    async def create_async(**kwargs: object) -> object:
        calls["created"].append(kwargs)
        return SimpleNamespace(id="lib_new")

    async def upload_async(**kwargs: object) -> object:
        calls["uploads"].append(kwargs)
        return SimpleNamespace(id="doc_1")

    _install_fake_mistral_client(
        monkeypatch,
        {
            "list_async": list_async,
            "create_async": create_async,
            "upload_async": upload_async,
        },
    )

    stored = await jlc.store_listing_count(
        "https://www.pracuj.pl/praca/bi%20developer", 33, "2026-11-25", "Rynek pracy"
    )

    assert stored.library_id == "lib_existing"
    assert calls["created"] == []
    assert len(calls["uploads"]) == 1
    upload = calls["uploads"][0]
    assert upload["library_id"] == "lib_existing"
    assert upload["file"].file_name == "liczba-ogloszen-bi-developer-2026-11-25.md"
    assert "Liczba ofert: 33" in upload["file"].content.decode()


async def test_store_listing_count_creates_library_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    calls: dict = {"created": [], "uploads": []}

    async def list_async(**kwargs: object) -> object:
        return SimpleNamespace(data=[])

    async def create_async(**kwargs: object) -> object:
        calls["created"].append(kwargs)
        return SimpleNamespace(id="lib_new")

    async def upload_async(**kwargs: object) -> object:
        calls["uploads"].append(kwargs)
        return SimpleNamespace(id="doc_1")

    _install_fake_mistral_client(
        monkeypatch,
        {
            "list_async": list_async,
            "create_async": create_async,
            "upload_async": upload_async,
        },
    )

    stored = await jlc.store_listing_count(
        "https://www.pracuj.pl/praca/bi%20developer", 33, "2026-11-25", "Rynek pracy"
    )

    assert stored.library_id == "lib_new"
    assert calls["created"][0]["name"] == "Rynek pracy"


def test_workflow_class_registered() -> None:
    import mistralai.workflows as workflows

    definition = workflows.get_workflow_definition(jlc.JobListingCount)
    assert definition.name == "job_listing_count"
