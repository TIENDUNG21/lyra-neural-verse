# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-17

First release.

### Added

- Reverse-mode autograd on NumPy with broadcasting-aware gradients and fused kernels for
  softmax, log-softmax, GELU, layer norm, embedding, cross-entropy, dropout and masked
  fill; every kernel is verified against finite differences.
- Decoder-only transformer (`LyraLM`) with tied embeddings, an output bias, phonology-aware
  factored embeddings and line-slot embeddings; AdamW, warmup plus cosine schedule,
  gradient clipping, `.npz` checkpoints with resume.
- Vietnamese text layer: NFC/NFD-independent normalization, an orthographic syllable
  parser (onset, medial, nucleus, coda, tone), exact and near rhyme classes, a strict
  lục bát checker, and documented exceptions for all 3,254 lines of *Truyện Kiều*.
- Corpus pipeline pinned to Wikisource revision 206607, with eight digitization errors
  corrected by line number.
- Constrained decoder that masks the vocabulary by tone class and rhyme class at every
  step, with temperature, top-k and top-p sampling over a mulberry32 generator.
- CLI: `lyra train`, `sample`, `check`, `export`, `evaluate`, `plot`.
- int8 model file (1.2 MB) shared by the CLI and the browser.
- Browser playground in TypeScript with the same engine, checker and decoder, tested for
  parity against a fixture written by the Python implementation; deployed to GitHub Pages.
- Release training run, three ablations, loss curve and held-out evaluation, reproducible
  with `scripts/reproduce.sh`.
