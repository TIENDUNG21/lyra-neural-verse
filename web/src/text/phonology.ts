// Vietnamese syllable phonology from orthography (port of lyra/text/phonology.py).
// A syllable = onset + medial + nucleus + coda + tone. See the Python module
// for the full explanation; the two are checked against each other in
// test/parity.test.ts over the whole vocabulary.

import { normalizeText } from "./normalize";

export enum Tone {
  Ngang = 0,
  Huyen = 1,
  Sac = 2,
  Hoi = 3,
  Nga = 4,
  Nang = 5,
}

export const TONE_NAMES = ["ngang", "huyền", "sắc", "hỏi", "ngã", "nặng"] as const;

export function isBang(tone: Tone): boolean {
  return tone === Tone.Ngang || tone === Tone.Huyen;
}

const MARK_TO_TONE = new Map<string, Tone>([
  [String.fromCharCode(0x0300), Tone.Huyen],
  [String.fromCharCode(0x0301), Tone.Sac],
  [String.fromCharCode(0x0309), Tone.Hoi],
  [String.fromCharCode(0x0303), Tone.Nga],
  [String.fromCharCode(0x0323), Tone.Nang],
]);
const TONE_MARK: Record<Tone, string> = {
  [Tone.Ngang]: "",
  [Tone.Huyen]: String.fromCharCode(0x0300),
  [Tone.Sac]: String.fromCharCode(0x0301),
  [Tone.Hoi]: String.fromCharCode(0x0309),
  [Tone.Nga]: String.fromCharCode(0x0303),
  [Tone.Nang]: String.fromCharCode(0x0323),
};

const ONSETS = [
  "ngh",
  "ng",
  "nh",
  "ch",
  "tr",
  "th",
  "ph",
  "kh",
  "gh",
  "gi",
  "qu",
  "b",
  "c",
  "d",
  "đ",
  "g",
  "h",
  "k",
  "l",
  "m",
  "n",
  "p",
  "r",
  "s",
  "t",
  "v",
  "x",
];
const CONSONANT_CODAS = ["ng", "nh", "ch", "c", "m", "n", "p", "t"];
const VOWELS = new Set("aăâeêioôơuưy");
const NUCLEI = new Map<string, string>([
  ["a", "a"],
  ["ă", "ă"],
  ["â", "â"],
  ["e", "e"],
  ["ê", "ê"],
  ["i", "i"],
  ["y", "i"],
  ["o", "o"],
  ["ô", "ô"],
  ["ơ", "ơ"],
  ["u", "u"],
  ["ư", "ư"],
  ["iê", "iê"],
  ["yê", "iê"],
  ["ia", "iê"],
  ["ya", "iê"],
  ["uô", "uô"],
  ["ua", "uô"],
  ["ươ", "ươ"],
  ["ưa", "ươ"],
  ["oo", "oo"],
  ["ôô", "ôô"],
]);
const MEDIAL_O = new Set(["a", "ă", "e"]);
const MEDIAL_U = new Set(["â", "ê", "y", "ơ", "yê", "ya"]);

export interface Syllable {
  readonly text: string;
  readonly onset: string;
  readonly medial: string;
  readonly nucleus: string;
  readonly coda: string;
  readonly tone: Tone;
}

/** Bare (toneless, NFC) spelling and tone; null if two different tone marks. */
export function splitTone(text: string): [string, Tone] | null {
  const decomposed = text.normalize("NFD");
  const tones = new Set<Tone>();
  let bare = "";
  for (const ch of decomposed) {
    const tone = MARK_TO_TONE.get(ch);
    if (tone === undefined) bare += ch;
    else tones.add(tone);
  }
  if (tones.size > 1) return null;
  const [tone] = tones;
  return [bare.normalize("NFC"), tone ?? Tone.Ngang];
}

function isLetter(c: string): boolean {
  return (c >= "a" && c <= "z") || VOWELS.has(c) || c === "đ";
}

function splitOnset(bare: string): [string, string] {
  if (bare === "gi") return ["gi", ""];
  for (const onset of ONSETS) {
    if (bare.startsWith(onset) && bare.length > onset.length) {
      return [onset, bare.slice(onset.length)];
    }
  }
  return ["", bare];
}

function splitVowels(vowels: string, onset: string): [string, string] | null {
  if (onset === "qu") return NUCLEI.has(vowels) ? ["u", vowels] : null;
  if (vowels === "ua" || vowels === "uô" || NUCLEI.has(vowels)) return ["", vowels];
  const tail = vowels.slice(1);
  if (vowels.length >= 2 && vowels[0] === "o" && MEDIAL_O.has(tail)) return ["o", tail];
  if (vowels.length >= 2 && vowels[0] === "u" && MEDIAL_U.has(tail)) return ["u", tail];
  return null;
}

const cache = new Map<string, Syllable | null>();

/** Parse one written syllable; null if it is not Vietnamese. */
export function parseSyllable(text: string): Syllable | null {
  const cached = cache.get(text);
  if (cached !== undefined) return cached;
  const result = parseUncached(text);
  cache.set(text, result);
  return result;
}

function parseUncached(text: string): Syllable | null {
  const split = splitTone(normalizeLower(text));
  if (split === null) return null;
  const [bare, tone] = split;
  const chars = [...bare];
  if (chars.length === 0 || !chars.every(isLetter)) return null;

  let [onset, rest] = splitOnset(bare);
  if (onset === "gi" && (rest.startsWith("ê") || ![...rest].some((c) => VOWELS.has(c)))) {
    rest = `i${rest}`;
  }
  let coda = "";
  for (const candidate of CONSONANT_CODAS) {
    if (rest.endsWith(candidate) && rest.length > candidate.length) {
      coda = candidate;
      rest = rest.slice(0, -candidate.length);
      break;
    }
  }
  if (rest.length === 0 || ![...rest].every((c) => VOWELS.has(c))) return null;

  const last = rest[rest.length - 1] ?? "";
  if (coda === "" && rest.length >= 2 && "iyou".includes(last) && rest !== "uy" && rest !== "oo") {
    let body = rest.slice(0, -1);
    coda = "iy".includes(last) ? "i" : "u";
    if (body === "a" && "yu".includes(last)) body = "ă";
    else if (body.endsWith("a") && last === "y") body = `${body.slice(0, -1)}ă`;
    rest = body;
  }
  const vowels = splitVowels(rest, onset);
  if (vowels === null) return null;
  const [medial, spelling] = vowels;
  const nucleus = NUCLEI.get(spelling);
  if (nucleus === undefined) return null;
  return { text: canonicalSpelling(bare, tone), onset, medial, nucleus, coda, tone };
}

function normalizeLower(text: string): string {
  return normalizeText(text).toLowerCase();
}

function toneVowelIndex(bare: string): number {
  const start = (bare.startsWith("qu") || bare.startsWith("gi")) && bare.length > 2 ? 2 : 0;
  let positions: number[] = [];
  for (let i = start; i < bare.length; i++) if (VOWELS.has(bare[i] ?? "")) positions.push(i);
  if (positions.length === 0 && bare.startsWith("gi")) positions = [1];
  if (positions.length === 0) return -1;
  const marked = positions.filter((i) => "ăâêôơư".includes(bare[i] ?? ""));
  if (marked.length > 0) return marked[marked.length - 1] ?? -1;
  const last = positions[positions.length - 1] ?? -1;
  if (last < bare.length - 1 || positions.length === 1) return last;
  return (positions.length >= 3 ? positions[1] : positions[0]) ?? -1;
}

/** Put the tone mark on the conventional vowel of a toneless spelling. */
export function canonicalSpelling(bare: string, tone: Tone): string {
  if (tone === Tone.Ngang) return bare;
  const index = toneVowelIndex(bare);
  if (index < 0) return bare;
  return (bare.slice(0, index + 1) + TONE_MARK[tone] + bare.slice(index + 1)).normalize("NFC");
}

/** Canonical lowercase spelling (tone mark position normalized). */
export function canonicalSyllable(text: string): string {
  const parsed = parseSyllable(text);
  return parsed === null ? normalizeLower(text).normalize("NFC") : parsed.text;
}
