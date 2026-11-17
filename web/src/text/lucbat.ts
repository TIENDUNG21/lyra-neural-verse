// Strict lục bát checker (port of lyra/text/lucbat.py, without waivers).
import { splitSyllables } from "./normalize";
import { type Syllable, TONE_NAMES, Tone, isBang, parseSyllable } from "./phonology";
import { RhymeMatch, rhymeMatch } from "./rhyme";

export const TONE_PATTERN: Record<6 | 8, Record<number, boolean>> = {
  6: { 2: true, 4: false, 6: true },
  8: { 2: true, 4: false, 6: true, 8: true },
};

export type Rule = "length" | "tone" | "contrast" | "rhyme" | "syllable";

export interface Violation {
  readonly line: number;
  readonly rule: Rule;
  readonly position: number | null;
  readonly message: string;
}

export interface LineReport {
  readonly number: number;
  readonly expectedLength: 6 | 8;
  readonly syllables: readonly string[];
  readonly parsed: readonly (Syllable | null)[];
}

export interface RhymeLink {
  readonly first: readonly [number, number];
  readonly second: readonly [number, number];
  readonly match: RhymeMatch;
}

export interface CheckReport {
  readonly lines: readonly LineReport[];
  readonly violations: readonly Violation[];
  readonly rhymes: readonly RhymeLink[];
}

export function splitPoem(text: string): string[] {
  return text
    .split(/[\n/|]+/)
    .map((line) => line.trim())
    .filter((line) => splitSyllables(line).length > 0);
}

export function checkLines(lines: readonly string[]): CheckReport {
  if (lines.length === 0) throw new Error("nothing to check: the poem has no lines");
  const reports: LineReport[] = lines.map((line, index) => {
    const syllables = splitSyllables(line);
    return {
      number: index + 1,
      expectedLength: index % 2 === 0 ? 6 : 8,
      syllables,
      parsed: syllables.map(parseSyllable),
    };
  });
  const violations: Violation[] = [];
  const rhymes: RhymeLink[] = [];
  reports.forEach((report, index) => {
    violations.push(...checkLine(report));
    const previous = reports[index - 1];
    if (previous !== undefined) {
      const [link, violation] = checkRhyme(previous, report);
      if (link) rhymes.push(link);
      if (violation) violations.push(violation);
    }
  });
  return { lines: reports, violations, rhymes };
}

const at = (r: LineReport, position: number): Syllable | null => r.parsed[position - 1] ?? null;
const lengthOk = (r: LineReport): boolean => r.syllables.length === r.expectedLength;

function checkLine(r: LineReport): Violation[] {
  if (!lengthOk(r)) {
    const message = `expected ${r.expectedLength} syllables, found ${r.syllables.length}`;
    return [{ line: r.number, rule: "length", position: null, message }];
  }
  const out: Violation[] = [];
  for (const [key, needBang] of Object.entries(TONE_PATTERN[r.expectedLength])) {
    const position = Number(key);
    const syllable = at(r, position);
    const word = r.syllables[position - 1] ?? "";
    if (syllable === null) {
      out.push({
        line: r.number,
        rule: "syllable",
        position,
        message: `'${word}' is not Vietnamese`,
      });
    } else if (isBang(syllable.tone) !== needBang) {
      const want = needBang ? "bằng" : "trắc";
      const message = `'${word}' (${TONE_NAMES[syllable.tone]}) should be ${want}`;
      out.push({ line: r.number, rule: "tone", position, message });
    }
  }
  if (r.expectedLength === 8) {
    const sixth = at(r, 6);
    const eighth = at(r, 8);
    if (
      sixth &&
      eighth &&
      isBang(sixth.tone) &&
      isBang(eighth.tone) &&
      sixth.tone === eighth.tone
    ) {
      const kind = sixth.tone === Tone.Huyen ? "huyền" : "ngang";
      const message = `syllables 6 and 8 are both ${kind}; one must be ngang, the other huyền`;
      out.push({ line: r.number, rule: "contrast", position: 8, message });
    }
  }
  return out;
}

function checkRhyme(
  previous: LineReport,
  current: LineReport,
): [RhymeLink | null, Violation | null] {
  if (!lengthOk(previous) || !lengthOk(current)) return [null, null];
  const sourcePos = previous.expectedLength;
  const source = at(previous, sourcePos);
  const target = at(current, 6);
  if (source === null || target === null) return [null, null];
  const match = rhymeMatch(source, target);
  const link: RhymeLink = {
    first: [previous.number, sourcePos],
    second: [current.number, 6],
    match,
  };
  if (match !== RhymeMatch.None) return [link, null];
  const message =
    `'${target.text}' does not rhyme with '${source.text}' ` +
    `(line ${previous.number}, syllable ${sourcePos})`;
  return [link, { line: current.number, rule: "rhyme", position: 6, message }];
}
