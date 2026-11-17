// Per-token phonological features (port of lyra/text/features.py).
import { parseSyllable } from "./phonology";
import { exactKey, nearKey } from "./rhyme";

export interface TokenFeatures {
  readonly onset: Int32Array;
  readonly rhyme: Int32Array;
  readonly near: Int32Array;
  readonly tone: Int32Array; // 0 = none, 1 + Tone otherwise
  readonly isSyllable: Uint8Array;
  readonly rhymeNames: readonly string[];
  readonly nearNames: readonly string[];
}

function indexOf(names: string[], key: string): number {
  let i = names.indexOf(key);
  if (i < 0) {
    names.push(key);
    i = names.length - 1;
  }
  return i;
}

export function tokenFeatures(vocab: readonly string[]): TokenFeatures {
  const n = vocab.length;
  const onsetNames = ["-"];
  const rhymeNames = ["-"];
  const nearNames = ["-"];
  const features = {
    onset: new Int32Array(n),
    rhyme: new Int32Array(n),
    near: new Int32Array(n),
    tone: new Int32Array(n),
    isSyllable: new Uint8Array(n),
    rhymeNames,
    nearNames,
  };
  vocab.forEach((token, i) => {
    const s = token.startsWith("<") ? null : parseSyllable(token);
    if (s === null) return;
    features.onset[i] = indexOf(onsetNames, s.onset || "∅");
    features.rhyme[i] = indexOf(rhymeNames, exactKey(s));
    features.near[i] = indexOf(nearNames, nearKey(s));
    features.tone[i] = 1 + s.tone;
    features.isSyllable[i] = 1;
  });
  return features;
}
