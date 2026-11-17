// Exact and near rhyme (port of lyra/text/rhyme.py).
import type { Syllable } from "./phonology";

export enum RhymeMatch {
  None = 0,
  Near = 1,
  Exact = 2,
}

const VOWEL_CLASS = new Map<string, string>([
  ["i", "front"],
  ["ê", "front"],
  ["e", "front"],
  ["iê", "front"],
  ["u", "rounded"],
  ["ô", "rounded"],
  ["o", "rounded"],
  ["uô", "rounded"],
  ["oo", "rounded"],
  ["ôô", "rounded"],
  ["a", "central"],
  ["ă", "central"],
  ["â", "central"],
  ["ơ", "central"],
  ["ư", "central"],
  ["ươ", "central"],
]);

export function vowelClass(s: Syllable): string {
  if (s.nucleus === "a" && (s.coda === "nh" || s.coda === "ch")) return "front";
  const cls = VOWEL_CLASS.get(s.nucleus);
  if (cls === undefined) throw new Error(`unknown nucleus ${s.nucleus}`);
  return cls;
}

export function exactKey(s: Syllable): string {
  return `${s.nucleus}|${s.coda}`;
}

export function nearKey(s: Syllable): string {
  return s.coda === "i" ? "*|i" : `${vowelClass(s)}|${s.coda}`;
}

export function rhymeMatch(a: Syllable, b: Syllable): RhymeMatch {
  if (a.nucleus === b.nucleus && a.coda === b.coda) return RhymeMatch.Exact;
  if (nearKey(a) === nearKey(b)) return RhymeMatch.Near;
  return RhymeMatch.None;
}
