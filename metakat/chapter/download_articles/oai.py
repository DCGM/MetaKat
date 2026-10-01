from __future__ import annotations

import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

USER_AGENT = "MetaKat-article-sampler/0.1"

_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


@dataclass
class OaiRecord:
    identifier: str
    datestamp: str | None
    sets: list[str]
    # Dublin Core element name -> values, in document order, stripped, empty values dropped.
    dc: dict[str, list[str]]
    # The record's metadata element, for formats richer than oai_dc.
    metadata: ET.Element | None = None


def http_get(url: str, timeout: float = 120, retries: int = 5, backoff: float = 10) -> bytes:
    """GET with retries. Requests are sequential and callers pause between them; a server that asks
    to slow down (429/503) is waited for as long as its Retry-After says, at least ``backoff``."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:
            if attempt == retries or (isinstance(error, urllib.error.HTTPError) and error.code in (403, 404, 410)):
                raise
            wait = backoff * attempt
            if isinstance(error, urllib.error.HTTPError) and error.code in (429, 503):
                retry_after = error.headers.get("Retry-After", "")
                wait = max(wait, float(retry_after) if retry_after.isdigit() else 0)
            logger.warning(f"GET {url} failed ({error}), waiting {wait:.0f} s, attempt {attempt}/{retries}")
            time.sleep(wait)
    raise AssertionError("unreachable")


def parse_records(xml: bytes) -> tuple[list[OaiRecord], str | None]:
    """Parse one ListRecords response into its records and the resumption token."""
    root = ET.fromstring(xml)
    error = root.find("oai:error", _NS)
    if error is not None:
        if error.get("code") == "noRecordsMatch":
            return [], None
        raise RuntimeError(f"OAI-PMH error {error.get('code')}: {error.text}")

    records = []
    for record in root.iterfind("oai:ListRecords/oai:record", _NS):
        header = record.find("oai:header", _NS)
        if header.get("status") == "deleted":
            continue
        dc: dict[str, list[str]] = {}
        metadata = record.find("oai:metadata", _NS)
        if metadata is not None:
            for element in metadata.iter():
                if not element.tag.startswith("{" + _NS["dc"] + "}"):
                    continue
                value = (element.text or "").strip()
                if value:
                    dc.setdefault(element.tag.split("}", 1)[1], []).append(value)
        records.append(OaiRecord(
            identifier=header.findtext("oai:identifier", namespaces=_NS),
            datestamp=header.findtext("oai:datestamp", namespaces=_NS),
            sets=[s.text for s in header.iterfind("oai:setSpec", _NS)],
            dc=dc,
            metadata=next(iter(metadata), None) if metadata is not None else None,
        ))

    token = root.findtext("oai:ListRecords/oai:resumptionToken", namespaces=_NS)
    return records, (token.strip() if token and token.strip() else None)


def list_sets(base_url: str) -> list[tuple[str, str]]:
    """Return the (setSpec, setName) pairs of an OAI-PMH repository."""
    sets = []
    url = f"{base_url}?verb=ListSets"
    while url:
        root = ET.fromstring(http_get(url))
        for element in root.iterfind("oai:ListSets/oai:set", _NS):
            sets.append((element.findtext("oai:setSpec", namespaces=_NS), element.findtext("oai:setName", namespaces=_NS)))
        token = root.findtext("oai:ListSets/oai:resumptionToken", namespaces=_NS)
        url = f"{base_url}?{urllib.parse.urlencode({'verb': 'ListSets', 'resumptionToken': token.strip()})}" \
            if token and token.strip() else None
    return sets


def iter_records(base_url: str, metadata_prefix: str = "oai_dc", set_spec: str | None = None,
                 delay: float = 1.0) -> Iterator[OaiRecord]:
    """Yield every non-deleted record of an OAI-PMH repository, following resumption tokens."""
    params = {"verb": "ListRecords", "metadataPrefix": metadata_prefix}
    if set_spec:
        params["set"] = set_spec
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    page = 0
    while url:
        records, token = parse_records(http_get(url))
        page += 1
        logger.info(f"OAI page {page}: {len(records)} records")
        yield from records
        if not token:
            break
        url = f"{base_url}?{urllib.parse.urlencode({'verb': 'ListRecords', 'resumptionToken': token})}"
        time.sleep(delay)
