# ADR 0004: Structure-aware embeddings, folded into one int8 file

Status: accepted

## Context

A plain GPT recipe on this corpus (the `plain` ablation below) reaches a held-out
syllable perplexity of 574, slightly worse than an interpolated bigram (567). Two
problems show in its log. Line markers are 12.5% of the validation tokens, and from the
logged all-token loss (5.698) and syllable-only loss (ln 574 = 6.353) the model still
spends about 1.1 nats on each marker: it does not reliably know where a line ends, and
so not which positions carry the tone and rhyme rules. And most of the 2,374 syllables
occur a handful of times, too few to learn an embedding from.

The browser must run the same model, so whatever is added should not complicate the
TypeScript engine.

## Decision

Three additions to the plain GPT recipe:

1. A line-slot embedding: 17 learned vectors indexed by line type and position in the
   line (0 before the first marker or past a line's end). Slots are a deterministic
   function of the token stream, computed for the whole stream so a training window that
   starts mid-line still knows its slots, and tracked incrementally by both inference
   engines.
2. Factored token embeddings: the vector of a syllable is its own vector plus the vectors
   of its onset, rhyme (nucleus + coda) and tone. The output layer is tied to the summed
   table.
3. An output bias initialized to the centered log-unigram distribution of the training
   tokens; with Adam at a learning rate of 1e-3 a zero-initialized bias moves too slowly
   to matter.

`lyra export` sums the factored table into a single embedding matrix. The released file
holds a plain transformer plus the slot table, and the browser engine only adds one
lookup per token.

## Consequences

Measured with the release configuration, changing only the two switches
(`runs/summary.md`):

| Phonology | Slots | Held-out syllable perplexity | Best step |
|---|---|---|---|
| no | no | 574 | 500 |
| yes | no | 671 | 225 |
| no | yes | 476 | 500 |
| yes | yes | 467 | 275 |

Line slots account for nearly all of the gain; with them the marker loss falls to about
0.02 nats (all-token loss 5.380 against syllable-only 6.146). Factored embeddings add a little on top of
slots and hurt without them: they give the model more ways to fit rare syllables, and
without the structural signal it uses that capacity to memorize sooner. The decision keeps
both because the combination is best, but the factored embeddings are the weaker part of
this ADR and the first thing to revisit with a larger corpus.

Exporting the folded table means the released model cannot be trained further with the
factored parametrization; continuing training starts from `best.npz`, which keeps the
factors. Int8 per-channel quantization of the folded model costs nothing measurable
(466.9 against 467.0).
