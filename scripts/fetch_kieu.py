"""Download Truyện Kiều from Vietnamese Wikisource and write the cleaned corpus.

Usage::

    python -I scripts/fetch_kieu.py            # download + clean
    python -I scripts/fetch_kieu.py --offline  # re-clean the cached download

The revision is pinned (``SOURCE_REVISION``) so the corpus is reproducible.
The raw API response is cached in ``.cache/downloads/`` (not committed); only
the cleaned text in ``data/truyen-kieu.txt`` is committed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lyra.train.corpus import (  # noqa: E402
    CORRECTIONS,
    EXPECTED_LINES,
    SOURCE_PAGE,
    SOURCE_REVISION,
    CorpusError,
    extract_poem_lines,
)

API = "https://vi.wikisource.org/w/api.php"
USER_AGENT = "lyra-neural-verse-corpus/1.0 (https://github.com/TIENDUNG21/lyra-neural-verse)"


def download(cache: Path) -> None:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "revisions",
            "revids": SOURCE_REVISION,
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "format": "json",
            "formatversion": "2",
        }
    )
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        cache.write_bytes(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="use the cached download")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "truyen-kieu.txt")
    args = parser.parse_args()

    cache = ROOT / ".cache" / "downloads" / f"kieu-{SOURCE_REVISION}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not args.offline or not cache.exists():
        download(cache)

    payload = json.loads(cache.read_text(encoding="utf-8"))
    page = payload["query"]["pages"][0]
    revision = page["revisions"][0]
    if page["title"] != SOURCE_PAGE or revision["revid"] != SOURCE_REVISION:
        raise CorpusError(f"unexpected page/revision: {page['title']} {revision['revid']}")
    lines = extract_poem_lines(revision["slots"]["main"]["content"], CORRECTIONS)
    if len(lines) != EXPECTED_LINES:
        raise CorpusError(f"expected {EXPECTED_LINES} lines, found {len(lines)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} lines from revision {SOURCE_REVISION} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
