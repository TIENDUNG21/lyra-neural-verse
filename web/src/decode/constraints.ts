// The lục bát constraint state machine (port of lyra/decode/constraints.py).
import type { TokenFeatures } from "../text/features";
import { TONE_PATTERN } from "../text/lucbat";
import { type Syllable, Tone, parseSyllable } from "../text/phonology";
import { exactKey, nearKey } from "../text/rhyme";
import { LINE6, LINE8, UNK, lineMarker } from "../text/tokenizer";

const NGANG = 1 + Tone.Ngang;
const HUYEN = 1 + Tone.Huyen;

export interface DecodeState {
  readonly linesDone: number;
  readonly inLine: boolean;
  readonly position: number;
  readonly sixthOfSix: Syllable | null;
  readonly endOfEight: Syllable | null;
  readonly sixthOfEight: Syllable | null;
}

export const START_STATE: DecodeState = {
  linesDone: 0,
  inLine: false,
  position: 0,
  sixthOfSix: null,
  endOfEight: null,
  sixthOfEight: null,
};

export const lineLength = (s: DecodeState): 6 | 8 => (s.linesDone % 2 === 0 ? 6 : 8);

export interface Step {
  readonly mask: Uint8Array;
  readonly forced: number | null;
  readonly relaxed: boolean;
}

export class LucBatConstraints {
  private readonly size: number;
  private readonly syllables: Uint8Array;
  private readonly exactIds: Map<string, number>;
  private readonly nearIds: Map<string, number>;

  constructor(
    private readonly features: TokenFeatures,
    private readonly vocab: readonly string[],
    private readonly strictRhyme = false,
  ) {
    this.size = vocab.length;
    this.syllables = Uint8Array.from(features.isSyllable);
    this.syllables[UNK] = 0;
    this.exactIds = new Map(features.rhymeNames.map((name, i) => [name, i]));
    this.nearIds = new Map(features.nearNames.map((name, i) => [name, i]));
  }

  rhymeMask(source: Syllable): Uint8Array {
    const out = new Uint8Array(this.size);
    const ids = this.strictRhyme ? this.features.rhyme : this.features.near;
    const key = this.strictRhyme
      ? (this.exactIds.get(exactKey(source)) ?? -1)
      : (this.nearIds.get(nearKey(source)) ?? -1);
    for (let i = 0; i < this.size; i++) out[i] = ids[i] === key ? 1 : 0;
    return out;
  }

  step(state: DecodeState): Step {
    if (!state.inLine) {
      const marker = lineMarker(state.linesDone);
      const mask = new Uint8Array(this.size);
      mask[marker] = 1;
      return { mask, forced: marker, relaxed: false };
    }
    const length = lineLength(state);
    const position = state.position + 1;
    const need = TONE_PATTERN[length][position];
    let wantTone: number | null = null;
    if (length === 8 && position === 8 && state.sixthOfEight !== null) {
      if (state.sixthOfEight.tone === Tone.Huyen) wantTone = NGANG;
      else if (state.sixthOfEight.tone === Tone.Ngang) wantTone = HUYEN;
    }
    const mask = new Uint8Array(this.size);
    for (let i = 0; i < this.size; i++) {
      if (!this.syllables[i]) continue;
      const tone = this.features.tone[i] ?? 0;
      const bang = tone === NGANG || tone === HUYEN;
      if (need === true && !bang) continue;
      if (need === false && bang) continue;
      if (wantTone !== null && tone !== wantTone) continue;
      mask[i] = 1;
    }
    if (position === 6) {
      const source = length === 8 ? state.sixthOfSix : state.endOfEight;
      if (source !== null) {
        const rhyme = this.rhymeMask(source);
        let any = false;
        for (let i = 0; i < this.size; i++) {
          rhyme[i] = mask[i] && rhyme[i] ? 1 : 0;
          if (rhyme[i]) any = true;
        }
        if (any) return { mask: rhyme, forced: null, relaxed: false };
        return { mask, forced: null, relaxed: true };
      }
    }
    return { mask, forced: null, relaxed: false };
  }

  advance(state: DecodeState, token: number, text?: string): DecodeState {
    if (!state.inLine) {
      if (token !== LINE6 && token !== LINE8) {
        throw new Error("a line must start with its marker token");
      }
      return { ...state, inLine: true, position: 0 };
    }
    if (token === LINE6 || token === LINE8) throw new Error("marker token inside a line");
    const position = state.position + 1;
    const syllable = parseSyllable(text ?? this.vocab[token] ?? "");
    const length = lineLength(state);
    let next: DecodeState = { ...state, position };
    if (position === 6 && length === 6) next = { ...next, sixthOfSix: syllable };
    else if (position === 6 && length === 8) next = { ...next, sixthOfEight: syllable };
    if (position === length) {
      next = { ...next, linesDone: state.linesDone + 1, inLine: false, position: 0 };
      if (length === 8) next = { ...next, endOfEight: syllable };
    }
    return next;
  }
}
