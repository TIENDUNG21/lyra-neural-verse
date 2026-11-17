# ADR 0003: Strict lục bát rules, rhyme classes from data, declared exceptions

Status: accepted

## Context

Textbook lục bát fixes the tones at the even positions and asks for rhymes at two
places per couplet. Real poems deviate. A checker that every line of *Truyện Kiều* must
pass can be built two ways: loosen the rules until the poem fits, or keep the rules and
list the deviations.

## Decision

The rules stay strict, and the deviations of the reference poem are written down in
`data/kieu-exceptions.tsv` (line, rule, position, reason). A test asserts that the strict
checker's violations on the whole poem are exactly that list, and that every entry is
used. Changing a rule therefore changes the test result instead of silently changing
what "valid" means.

Rhyme strength was decided from the poem. All 3,253 rhyme pairs were classified with the
phonological parser:

| Relation | Pairs |
|---|---|
| Same nucleus and coda (exact) | 1,937 |
| Same coda and vowel class (near) | 1,289 |
| Neither | 27 |

The vowel classes (front i/ê/e/iê, back rounded u/ô/o/uô, central a/ă/â/ơ/ư/ươ) follow
articulation; the a of `anh/ach` counts as front because it is pronounced front. Before
the glide coda `-i` the poem pairs rounded and unrounded vowels freely (`lời/hồi`,
`hai/trời`), so with that coda every nucleus rhymes.

## Consequences

The exceptions file has 60 entries. 33 are tone deviations, all in six-syllable lines
read with a 3/3 rhythm (`Mai cốt cách, tuyết tinh thần`), a well-known variant in which
positions 2 and 4 are not bound. 27 are forced rhymes (vần ép), mostly across nasal codas
(`an/ang`, `in/inh`) or archaic spellings (`chiềng`, `nường`). No eight-syllable line
breaks the ngang/huyền contrast at positions 6 and 8.

Eight lines of the Wikisource revision had punctuation merged into the last vowel as a
tone mark (`làm đầụ` for `làm đầu.`). These are corrected in the cleaning step by line
number, and the correction raises if the source text changes.

The decoder uses the strict rules with exact or near rhyme, so generated poems pass the
checker without waivers.
