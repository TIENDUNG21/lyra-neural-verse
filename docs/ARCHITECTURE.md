# Architecture

Lyra is two programs that share one model file. The Python package trains a small
decoder-only transformer on *Truyện Kiều* with its own autograd, then exports an int8
file. The Python CLI and the TypeScript playground both load that file and run the same
inference engine and the same constrained decoder. A fixture written by Python pins the
two implementations together.

```mermaid
flowchart LR
  subgraph data [Data]
    W[vi.wikisource.org rev 206607] -->|scripts/fetch_kieu.py| K[data/truyen-kieu.txt]
  end
  subgraph text [lyra.text]
    N[normalize] --> P[phonology] --> R[rhyme]
    P --> C[lucbat checker]
    R --> C
    P --> F[features]
    T[tokenizer] --> S[slots]
  end
  subgraph train [lyra.autograd / nn / optim / train]
    A[Tensor + backward] --> M[LyraLM]
    M --> L[training loop] --> B[best.npz]
  end
  subgraph release [release]
    B -->|lyra export: fold + int8| Q[models/lyra-kieu-q8.bin]
  end
  K --> T
  F --> M
  S --> M
  Q --> E1[Python InferenceEngine] --> D1[Python decoder] --> CLI[lyra sample]
  Q --> E2[TS Engine] --> D2[TS decoder] --> UI[web playground]
  D1 -. parity.json .-> D2
```

## Text layer (`lyra/text`)

Everything starts from Unicode. `normalize.py` maps input to NFC, removes zero-width
characters and replaces the Icelandic eth (U+00D0/U+00F0), which some Vietnamese text
uses for Đ/đ. NFD input therefore produces the same tokens as NFC input; a property
test checks this over arbitrary strings.

`phonology.py` parses a written syllable into onset, medial, nucleus, coda and tone.
The parser works on orthography, so it has to undo the spelling conventions: `gi` and
`qu` are onsets (`già` = gi + a, `quốc` = qu + medial u + ô + c), `ia/ya/iê/yê` spell
one diphthong, `ay` and `au` carry a short ă, and `oa/oe/uy/uâ/uê/uơ` begin with a
glide. Tones come from the combining marks after NFD decomposition. The same module
re-places tone marks in the traditional position (`hoà` becomes `hòa`), which is the
style of the Wikisource text, so both spellings are one token.

`rhyme.py` defines two rhyme strengths. An exact rhyme (vần chính) shares nucleus and
coda. A near rhyme (vần thông) shares the coda and a vowel class: front (i, ê, e, iê),
back rounded (u, ô, o, uô) or central (a, ă, â, ơ, ư, ươ). Before the glide coda `-i`
any nucleus rhymes. These classes came from measuring every rhyme pair in the poem
(ADR 0003): 1,937 of 3,253 pairs are exact, 1,289 near, 27 neither. Near rhyme is an
equivalence relation, so every syllable gets a `near_key` and the decoder can build a
rhyme mask with one comparison per token.

`lucbat.py` checks the form: line lengths 6/8, bằng/trắc at the even positions,
ngang/huyền contrast at positions 6 and 8 of the 8-line, and the two rhymes per
couplet. It is strict. Real deviations are not absorbed into the rules; they are
declared as waivers (`data/kieu-exceptions.tsv`) with a reason.

`tokenizer.py` maps canonical syllables to ids and prefixes every line with `<l6>` or
`<l8>`. `slots.py` derives, from the token stream alone, where each token sits: line
type and position in the line (17 slots). `features.py` gives each vocabulary entry its
onset, rhyme, near-rhyme class and tone ids.

## Autograd (`lyra/autograd`)

`Tensor` wraps a NumPy array. Every operation on a tensor that requires gradients
records its parents and a closure that maps the output gradient to input gradients.
`backward()` builds a topological order with an explicit stack (no recursion limit on
deep graphs) and accumulates gradients in a dictionary keyed by node identity, so a
tensor used twice receives the sum. Broadcasting is handled in one place,
`unbroadcast`, which sums a gradient back to the shape of the operand that was
broadcast.

Elementwise operations, `matmul`, `reshape`, `transpose` and indexing are primitives.
Softmax, log-softmax, GELU, layer norm, embedding lookup, cross-entropy, dropout and
masked fill are fused: their backward passes are derived by hand, which keeps the graph
small and avoids storing intermediate arrays (ADR 0002). Each one is verified against
central finite differences in float64, and Hypothesis generates random broadcastable
shapes for the binary operations.

## Model (`lyra/nn`)

`LyraLM` is a pre-norm GPT-style decoder: token embedding plus learned position
embedding, four blocks of causal multi-head attention and a GELU MLP, a final layer
norm, and an output layer tied to the token embedding. Two additions adapt it to a
23k-token corpus of fixed-form verse (ADR 0004 has the ablation; the slots do most of
the work):

- Phonology-aware embeddings. The vector of a syllable is the sum of its own vector and
  the vectors of its onset, rhyme and tone. Rare syllables share what is known about
  their sound, and because the output layer is tied to the same table, a hidden state
  can raise every syllable of one rhyme class at once.
- Line-slot embeddings. A learned vector per slot is added to the input, so the model
  does not have to count syllables to know where a line ends or which positions carry
  the tone and rhyme rules. This follows the format embeddings of SongNet (Li et al.,
  2020).

An output bias, initialized to the log-unigram distribution of the training data, lets
the model start from word frequencies instead of learning them through dot products.

`InferenceEngine` is a separate, autograd-free implementation for generation. It keeps
per-layer key and value caches, so each new token costs one row of attention.
Positions are learned, so the cache cannot slide; when it is full the engine re-reads
the most recent half of the context and continues. Tests compare it with `LyraLM` at
every step, including across the overflow.

## Training (`lyra/optim`, `lyra/train`)

The poem is cut into blocks of 40 lines; every tenth block is held out, which spreads
validation across the story. Training samples random 128-token windows from the
concatenated training blocks, with the line slots taken from the whole stream so a
window that starts mid-line still knows its slots. Token dropout replaces 10% of the
input syllables with `<unk>`. AdamW (decoupled weight decay on matrices only), linear
warmup then cosine decay, global gradient-norm clipping at 1.0.

Evaluation uses windows that advance by half a block and score only their second half,
so every validation token is predicted with at least 64 tokens of context. The loop
writes `log.csv` every 25 steps, keeps `best.npz` by validation loss and `last.npz` with
optimizer and RNG state for `--resume`. `lyra plot` turns the log into the SVG in the
README without a plotting library.

## Release file (`lyra/io/modelfile.py`)

`lyra export` folds the factored embeddings into one table, so the released model is
a plain transformer and the browser needs no phonology to run it. Matrices are stored
as int8 with one float32 scale per output channel (symmetric quantization); vectors
stay float32. The header is JSON with the config, the vocabulary and training
metadata. The file is 1.2 MB and is the only model artifact in the repository.

## Decoding (`lyra/decode`)

`LucBatConstraints` is a state machine over the poem. At each step it returns a
boolean mask: the line marker at a line start; any syllable at free positions; only
bằng or only trắc at constrained positions; the opposite of position 6's tone at
position 8 of the 8-line; and at position 6, only syllables that rhyme with the
syllable being answered. The mask is applied to the model's logits before temperature,
top-k and top-p, so the model chooses among valid continuations and every generated
poem passes the checker. If a rhyme mask is empty (a rare rhyme typed by the user with
exact rhymes requested), the rhyme requirement is dropped for that step and the line is
flagged.

Sampling uses mulberry32, a 32-bit generator that is easy to reproduce bit for bit in
TypeScript, and arranges the floating-point work (sort order, running sums, the
comparison against `u * total`) so both languages perform the same operations in the
same order. The same seed gives the same poem in the CLI and in the browser.

## Browser (`web/`)

The TypeScript code mirrors the Python modules file for file: `model/format.ts` and
`model/engine.ts`, `text/*` (normalization, phonology, rhyme, checker, features,
tokenizer) and `decode/*`. `scripts/make_parity_fixture.py` writes
`web/test/fixtures/parity.json` from the committed model: logits after fixed token
sequences (one longer than the context), token sequences from five decoding
configurations, the parse of every vocabulary entry, and the checker's verdict on the
whole poem. Vitest requires logits within 1e-3 and identical tokens; a Python test
rebuilds the fixture and fails if the committed copy is stale.

The page is built by Vite with base `/lyra-neural-verse/`, and the build fails if
`dist/index.html` references an asset outside that base.
