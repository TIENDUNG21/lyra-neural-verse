// Unicode normalization and syllable splitting (port of lyra/text/normalize.py).

// Zero-width and other invisible format characters, by code point.
const INVISIBLE = new Set([0x200b, 0x200c, 0x200d, 0x2060, 0xfeff, 0x00ad, 0x180e]);
// Icelandic eth (U+00D0 / U+00F0) is sometimes typed for the Vietnamese D with stroke.
const LOOKALIKES = new Map([
  [0x00d0, 0x0110],
  [0x00f0, 0x0111],
]);
const WORD_CHAR = /^[\p{L}\p{M}]$/u;

/** NFC-normalize, drop invisible characters and fix D-with-stroke lookalikes. */
export function normalizeText(text: string): string {
  let out = "";
  for (const ch of text) {
    const code = ch.codePointAt(0) ?? 0;
    if (INVISIBLE.has(code)) continue;
    const replacement = LOOKALIKES.get(code);
    out += replacement === undefined ? ch : String.fromCodePoint(replacement);
  }
  return out.normalize("NFC");
}

/** Split a line into lowercase syllables; punctuation and symbols separate. */
export function splitSyllables(line: string): string[] {
  const text = normalizeText(line).toLowerCase().normalize("NFC");
  let separated = "";
  for (const ch of text) separated += WORD_CHAR.test(ch) ? ch : " ";
  return separated.split(" ").filter((s) => s.length > 0);
}
