# Corpus source

`truyen-kieu.txt` is *Truyện Kiều* (*Đoạn trường tân thanh*) by Nguyễn Du (1766-1820),
3,254 lines of lục bát verse, one verse per line.

- Source: Vietnamese Wikisource, page "Truyện Kiều", revision 206607 (2026-11-07):
  <https://vi.wikisource.org/w/index.php?title=Truy%E1%BB%87n_Ki%E1%BB%81u&oldid=206607>
- Status: the poem is in the public domain (the author died in 1820); Wikisource marks the
  page `{{PD-old}}`. Wikisource's own page content is available under CC BY-SA 4.0.
- Produced by `python -I scripts/fetch_kieu.py`, which downloads the pinned revision through
  the MediaWiki API into `.cache/downloads/` (not committed) and cleans it with
  `lyra.train.corpus.extract_poem_lines`.

## Cleaning

1. Keep only the text inside `<poem>` blocks; drop line-number templates (`{{số|5}}`),
   other templates, tags and wiki markup.
2. Normalize to Unicode NFC, remove zero-width characters, collapse whitespace.
3. Fix eight digitization errors, listed in `lyra.train.corpus.CORRECTIONS`. In seven lines
   the final punctuation had been merged into the last vowel as a tone mark (a full stop
   became a dot below, a question mark a hook above, e.g. `làm đầụ` -> `làm đầu.`), and
   line 676 had `nưóc` for `nước`.

Capitalization and punctuation are kept so the file stays readable. The tokenizer removes
punctuation and lowercases.
