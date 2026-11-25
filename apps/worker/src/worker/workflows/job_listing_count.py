"""Count job listings on a keyword search page and store the daily result in a Mistral Library.

Input is a search-results URL (for example https://www.pracuj.pl/praca/bi%20developer for the
keyword "bi developer"). The page shows the number of listings in phrases like
"Praca bi developer – Pracuj.pl - 33 oferty pracy". The workflow extracts that number, then
saves one document per run — `{keyword}-{date}.md` with the count and the date — into the
Mistral Library given by `library_name`.
"""

import re
from datetime import timedelta
from urllib.parse import unquote, urlparse

import mistralai.workflows as workflows
import mistralai.workflows.plugins.mistralai as workflows_mistralai
from pydantic import BaseModel, Field

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
}

_COUNT_SEPARATORS = "\u00a0\u202f "

_COUNT_GROUPS = r"\d+(?:[\u00a0\u202f ]\d+)*"

_COUNT_PATTERNS = (
    re.compile(
        rf"Pracuj\.pl\s*-\s*({_COUNT_GROUPS})\s*ofert(?:y|a|ów)?\s+pracy", re.IGNORECASE
    ),
    re.compile(rf"({_COUNT_GROUPS})\s+ofert(?:y|a|ów)?\s+pracy", re.IGNORECASE),
)


class JobListingCountInput(BaseModel):
    url: str = "https://www.pracuj.pl/praca/bi%20developer"
    library_name: str = "Rynek pracy"
    dry_run: bool = False


class PageListingCount(BaseModel):
    count: int
    page_title: str | None
    url: str


class StoredListingCount(BaseModel):
    library_id: str
    library_name: str
    document_id: str
    document_name: str


class JobListingCountOutput(BaseModel):
    count: int
    date: str
    url: str
    stored: StoredListingCount | None = Field(
        default=None, description="Saved library record, absent on dry_run."
    )


def keyword_from_url(url: str) -> str:
    """Human-readable search keyword from the URL path, e.g. "bi%20developer" -> "bi developer"."""
    path_segment = unquote(urlparse(url).path.rstrip("/").rpartition("/")[2])
    return path_segment or url


def extract_listing_count(html: str) -> int | None:
    """The number of job listings announced on the page, e.g. 33 from "- 33 oferty pracy"."""
    for pattern in _COUNT_PATTERNS:
        match = pattern.search(html)
        if match:
            return int(re.sub(rf"[{_COUNT_SEPARATORS}]", "", match.group(1)))
    return None


def record_document_name(url: str, count_date: str) -> str:
    keyword = keyword_from_url(url).strip().lower().replace(" ", "-")
    return f"liczba-ogloszen-{keyword}-{count_date}.md"


def record_markdown(url: str, count: int, count_date: str) -> str:
    keyword = keyword_from_url(url)
    return (
        f"# Liczba ogłoszeń: {keyword}\n\n"
        f"- Data: {count_date}\n"
        f"- Liczba ofert: {count}\n"
        f"- URL: {url}\n"
    )


@workflows.activity(start_to_close_timeout=timedelta(seconds=30))
async def fetch_listing_count(url: str) -> PageListingCount:
    import httpx2

    async with httpx2.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url, headers=BROWSER_HEADERS)
        response.raise_for_status()
        html = response.text

    count = extract_listing_count(html)
    if count is None:
        raise ValueError(f"No job-listing count found on {url}")

    title_match = re.search(r"<title>([^<]+)</title>", html)
    return PageListingCount(
        count=count,
        page_title=title_match.group(1).strip() if title_match else None,
        url=url,
    )


@workflows.activity(
    start_to_close_timeout=timedelta(seconds=60), retry_policy_max_attempts=5
)
async def store_listing_count(
    url: str, count: int, count_date: str, library_name: str
) -> StoredListingCount:
    from mistralai.client import models as mistralai_models

    client = workflows_mistralai.get_mistral_client()

    library_id = None
    libraries = await client.beta.libraries.list_async(page_size=100)
    for library in libraries.data:
        if library.name == library_name:
            library_id = library.id
            break
    if library_id is None:
        created = await client.beta.libraries.create_async(
            name=library_name,
            description="Dzienne wyniki pomiaru liczby ogłoszeń pracy dla wybranych słów kluczowych.",
            owner_type="Workspace",
        )
        library_id = created.id

    document = await client.beta.libraries.documents.upload_async(
        library_id=library_id,
        file=mistralai_models.File(
            file_name=record_document_name(url, count_date),
            content=record_markdown(url, count, count_date).encode(),
            content_type="text/markdown",
        ),
    )
    return StoredListingCount(
        library_id=library_id,
        library_name=library_name,
        document_id=document.id,
        document_name=record_document_name(url, count_date),
    )


@workflows.workflow.define(name="job_listing_count")
class JobListingCount:
    """Fetch a keyword search page, extract the number of listings, store it with its date."""

    @workflows.workflow.entrypoint
    async def run(self, params: JobListingCountInput) -> JobListingCountOutput:
        fetched = await fetch_listing_count(params.url)
        count_date = workflows.workflow.now().date().isoformat()

        stored = None
        if not params.dry_run:
            stored = await store_listing_count(
                url=params.url,
                count=fetched.count,
                count_date=count_date,
                library_name=params.library_name,
            )

        return JobListingCountOutput(
            count=fetched.count, date=count_date, url=params.url, stored=stored
        )
