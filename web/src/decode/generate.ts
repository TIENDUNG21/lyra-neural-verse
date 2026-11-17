// Constrained poem generation (port of lyra/decode/generate.py).
import type { Engine } from "../model/engine";
import { type TokenFeatures, tokenFeatures } from "../text/features";
import { splitSyllables } from "../text/normalize";
import { Tokenizer, lineMarker } from "../text/tokenizer";
import { LucBatConstraints, START_STATE } from "./constraints";
import { Mulberry32 } from "./rng";
import { DEFAULT_SETTINGS, type SamplingSettings, sampleToken } from "./sampling";

export interface PoemLine {
  readonly syllables: readonly string[];
  readonly fromPrompt: boolean;
  readonly relaxed: boolean;
  /** Probability of each sampled syllable among the allowed candidates (for the UI). */
  readonly confidence: readonly number[];
}

export interface Poem {
  readonly lines: readonly PoemLine[];
  readonly tokens: readonly number[];
}

export interface GenerateOptions {
  lines: number;
  prompt?: readonly string[];
  settings?: SamplingSettings;
  seed?: number;
  strictRhyme?: boolean;
}

export function lineText(line: PoemLine): string {
  const words = line.syllables.join(" ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function allowedProbability(logits: Float32Array, mask: Uint8Array, token: number): number {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < mask.length; i++) if (mask[i] && (logits[i] ?? 0) > max) max = logits[i] ?? 0;
  let total = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) total += Math.exp((logits[i] ?? 0) - max);
  return Math.exp((logits[token] ?? 0) - max) / total;
}

export class Poet {
  readonly tokenizer: Tokenizer;
  readonly features: TokenFeatures;

  constructor(
    readonly engine: Engine,
    vocab: readonly string[],
  ) {
    this.tokenizer = new Tokenizer(vocab);
    this.features = tokenFeatures(vocab);
  }

  generate(options: GenerateOptions): Poem {
    const lines = options.lines;
    const prompt = options.prompt ?? [];
    const settings = options.settings ?? DEFAULT_SETTINGS;
    if (lines < 1) throw new Error("lines must be >= 1");
    if (prompt.length > lines) throw new Error("the prompt has more lines than requested");
    const constraints = new LucBatConstraints(
      this.features,
      this.tokenizer.vocab,
      options.strictRhyme ?? false,
    );
    const rng = new Mulberry32(options.seed ?? 0);
    const engine = this.engine;
    engine.reset();
    let state = START_STATE;
    const tokens: number[] = [];
    const out: PoemLine[] = [];

    prompt.forEach((line, index) => {
      const syllables = splitSyllables(line);
      const expected = index % 2 === 0 ? 6 : 8;
      if (syllables.length !== expected) {
        throw new Error(
          `prompt line ${index + 1} has ${syllables.length} syllables; expected ${expected}`,
        );
      }
      const marker = lineMarker(index);
      state = constraints.advance(state, marker);
      tokens.push(marker);
      for (const s of syllables) {
        const id = this.tokenizer.tokenId(s);
        state = constraints.advance(state, id, s);
        tokens.push(id);
      }
      out.push({ syllables, fromPrompt: true, relaxed: false, confidence: [] });
    });

    let logits: Float32Array | null = tokens.length > 0 ? engine.feed(tokens) : null;
    let current: string[] = [];
    let confidence: number[] = [];
    let relaxed = false;
    while (state.linesDone < lines) {
      const step = constraints.step(state);
      let token: number;
      if (step.forced !== null) {
        token = step.forced;
      } else {
        if (logits === null) throw new Error("no logits before the first sampled token");
        token = sampleToken(logits, step.mask, settings, rng);
        relaxed = relaxed || step.relaxed;
        current.push(this.tokenizer.token(token));
        confidence.push(allowedProbability(logits, step.mask, token));
      }
      tokens.push(token);
      state = constraints.advance(state, token);
      if (!state.inLine && step.forced === null) {
        out.push({ syllables: current, fromPrompt: false, relaxed, confidence });
        current = [];
        confidence = [];
        relaxed = false;
      }
      if (state.linesDone < lines) logits = engine.feed([token]);
    }
    return { lines: out, tokens };
  }
}
