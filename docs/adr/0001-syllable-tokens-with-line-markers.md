# ADR 0001: Syllable tokens with line markers

Status: accepted

## Context

Vietnamese writes one syllable per space-separated unit, and every lục bát rule is
stated per syllable: the tone at positions 2, 4, 6 and 8, and the rhyme at position 6
or 8. The corpus is small (22,778 syllables, 2,374 distinct after normalization).

## Decision

A token is a canonical syllable: lowercase, NFC, tone mark placed on the conventional
vowel, punctuation removed. Four special tokens come first: `<pad>`, `<unk>`, `<l6>`
and `<l8>`. Each line is preceded by the marker of its type. The vocabulary is built
from the whole poem, not only the training blocks.

## Consequences

The decoder can apply the form rules token by token, because a token is exactly the
unit the rules talk about. A character or BPE model would have to finish a syllable
before its tone is known, and the mask would need lookahead.

The cost is a large vocabulary for the data size: 2,378 tokens for 20k training
tokens, many of them seen once. Two mitigations follow in ADR 0004.

Building the vocabulary from the whole poem leaks the *types* of the 61 validation
syllables that never occur in training; they still have untrained embeddings, so their
probability is low, and they are counted in the validation perplexity. The alternative,
mapping them to `<unk>`, would make the playground unable to produce them at all.

Capitalization is lost, including for names such as Kiều and Kim Trọng. The CLI and
the playground capitalize the first syllable of each line only.
