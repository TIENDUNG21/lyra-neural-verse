// Parity with the Python reference: the browser must run the same model and the
// same constrained decoder. The fixture is written by scripts/make_parity_fixture.py
// from the committed model file; a Python test fails if it is stale.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { Poet } from "../src/decode/generate";
import { Engine } from "../src/model/engine";
import { type ModelFile, parseModelFile } from "../src/model/format";
import { checkLines } from "../src/text/lucbat";
import { parseSyllable } from "../src/text/phonology";

interface Fixture {
  logits: Record<string, { tokens: number[]; logits: number[] }>;
  generations: {
    lines: number;
    seed: number;
    temperature: number;
    top_k: number;
    top_p: number;
    strict_rhyme?: boolean;
    prompt_lines: string[];
    tokens: number[];
  }[];
  vocab_parses: (unknown[] | null)[];
  extra_words: [string, unknown[] | null][];
  kieu: { violations: [number, string, number | null][]; rhymes: number[] };
}

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const fixture = JSON.parse(readFileSync(join(here, "fixtures", "parity.json"), "utf8")) as Fixture;

let model: ModelFile;

beforeAll(() => {
  const bytes = readFileSync(join(root, "models", "lyra-kieu-q8.bin"));
  model = parseModelFile(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length));
});

function record(word: string): unknown[] | null {
  const s = parseSyllable(word);
  return s === null ? null : [s.text, s.onset, s.medial, s.nucleus, s.coda, s.tone];
}

describe("model parity", () => {
  for (const name of ["couplet", "prefix", "overflow"]) {
    it(`logits match Python for the ${name} sequence`, () => {
      const expected = fixture.logits[name];
      if (expected === undefined) throw new Error(`missing case ${name}`);
      const engine = new Engine(model);
      const logits = engine.feed(expected.tokens);
      expect(logits.length).toBe(expected.logits.length);
      let worst = 0;
      let argmax = 0;
      let expectedArgmax = 0;
      expected.logits.forEach((value, i) => {
        worst = Math.max(worst, Math.abs((logits[i] ?? 0) - value));
        if ((logits[i] ?? 0) > (logits[argmax] ?? 0)) argmax = i;
        if (value > (expected.logits[expectedArgmax] ?? 0)) expectedArgmax = i;
      });
      expect(worst).toBeLessThan(1e-3);
      expect(argmax).toBe(expectedArgmax);
    });
  }
});

describe("decoder parity", () => {
  fixture.generations.forEach((g, index) => {
    it(`case ${index}: seed ${g.seed}, ${g.lines} lines gives the same tokens`, () => {
      const poet = new Poet(new Engine(model), model.vocab);
      const poem = poet.generate({
        lines: g.lines,
        prompt: g.prompt_lines,
        seed: g.seed,
        strictRhyme: g.strict_rhyme ?? false,
        settings: { temperature: g.temperature, topK: g.top_k, topP: g.top_p },
      });
      expect(poem.tokens).toEqual(g.tokens);
      const report = checkLines(poem.lines.map((l) => l.syllables.join(" ")));
      expect(report.violations).toEqual([]);
    });
  });
});

describe("phonology and checker parity", () => {
  it("parses every vocabulary entry like Python", () => {
    const parses = model.vocab.map((t) => (t.startsWith("<") ? null : record(t)));
    expect(parses).toEqual(fixture.vocab_parses);
  });

  it("handles the edge-case words like Python", () => {
    for (const [word, expected] of fixture.extra_words) expect(record(word)).toEqual(expected);
  });

  it("finds exactly the same violations and rhymes in all of Truyện Kiều", () => {
    const lines = readFileSync(join(root, "data", "truyen-kieu.txt"), "utf8")
      .split("\n")
      .filter((l) => l.trim().length > 0);
    const report = checkLines(lines);
    expect(report.violations.map((v) => [v.line, v.rule, v.position])).toEqual(
      fixture.kieu.violations,
    );
    expect(report.rhymes.map((r) => r.match)).toEqual(fixture.kieu.rhymes);
  });
});
