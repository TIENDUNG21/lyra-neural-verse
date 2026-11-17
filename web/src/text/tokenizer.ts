// Syllable tokenizer with line markers (port of lyra/text/tokenizer.py).
import { splitSyllables } from "./normalize";
import { canonicalSyllable } from "./phonology";

export const PAD = 0;
export const UNK = 1;
export const LINE6 = 2;
export const LINE8 = 3;
export const SPECIALS = ["<pad>", "<unk>", "<l6>", "<l8>"] as const;

export function lineMarker(index: number): number {
  return index % 2 === 0 ? LINE6 : LINE8;
}

export class Tokenizer {
  readonly vocab: readonly string[];
  private readonly ids: Map<string, number>;

  constructor(vocab: readonly string[]) {
    SPECIALS.forEach((token, i) => {
      if (vocab[i] !== token) throw new Error("vocabulary must start with the special tokens");
    });
    this.vocab = vocab;
    this.ids = new Map(vocab.map((token, i) => [token, i]));
  }

  get size(): number {
    return this.vocab.length;
  }

  tokenId(syllable: string): number {
    return this.ids.get(canonicalSyllable(syllable)) ?? UNK;
  }

  token(id: number): string {
    const token = this.vocab[id];
    if (token === undefined) throw new RangeError(`token id ${id} out of range`);
    return token;
  }

  encodeLine(line: string): number[] {
    return splitSyllables(line).map((s) => this.tokenId(s));
  }
}
