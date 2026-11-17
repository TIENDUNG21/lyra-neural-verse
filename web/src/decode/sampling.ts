// Temperature, top-k and top-p sampling (port of lyra/decode/sampling.py; the
// floating-point operations happen in the same order as in Python).
import type { Mulberry32 } from "./rng";

export interface SamplingSettings {
  temperature: number;
  topK: number;
  topP: number;
}

export const DEFAULT_SETTINGS: SamplingSettings = { temperature: 0.9, topK: 40, topP: 0.95 };

/** Allowed ids sorted by logit (descending), ties by id (ascending). */
export function rankedCandidates(logits: Float32Array, allowed: Uint8Array): number[] {
  const ids: number[] = [];
  for (let i = 0; i < allowed.length; i++) if (allowed[i]) ids.push(i);
  if (ids.length === 0) throw new Error("no token is allowed at this step");
  return ids.sort((a, b) => (logits[b] ?? 0) - (logits[a] ?? 0) || a - b);
}

export function sampleToken(
  logits: Float32Array,
  allowed: Uint8Array,
  settings: SamplingSettings,
  rng: Mulberry32,
): number {
  let ranked = rankedCandidates(logits, allowed);
  if (settings.temperature === 0) return ranked[0] ?? 0;
  if (settings.topK > 0) ranked = ranked.slice(0, settings.topK);
  const top = logits[ranked[0] ?? 0] ?? 0;
  const running: number[] = [];
  let total = 0;
  for (const id of ranked) {
    total += Math.exp(((logits[id] ?? 0) - top) / settings.temperature);
    running.push(total);
  }
  let cutoff = ranked.length;
  if (settings.topP < 1) {
    for (let n = 0; n < running.length; n++) {
      if ((running[n] ?? 0) >= settings.topP * total) {
        cutoff = n + 1;
        break;
      }
    }
  }
  const threshold = rng.random() * (running[cutoff - 1] ?? 0);
  for (let n = 0; n < cutoff; n++) if (threshold < (running[n] ?? 0)) return ranked[n] ?? 0;
  return ranked[cutoff - 1] ?? 0;
}
