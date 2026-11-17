import { describe, expect, it } from "vitest";
import { Mulberry32 } from "../src/decode/rng";
import { rankedCandidates, sampleToken } from "../src/decode/sampling";
import { START_SLOT_STATE, nextSlot } from "../src/model/engine";
import { ModelFormatError, parseModelFile } from "../src/model/format";
import { checkLines, splitPoem } from "../src/text/lucbat";
import { splitSyllables } from "../src/text/normalize";
import { Tone, canonicalSyllable, parseSyllable } from "../src/text/phonology";
import { RhymeMatch, rhymeMatch } from "../src/text/rhyme";
import { LINE6, LINE8 } from "../src/text/tokenizer";

describe("mulberry32", () => {
  it("matches the reference sequence (same values as tests/decode/test_sampling.py)", () => {
    const rng = new Mulberry32(42);
    expect([rng.random(), rng.random(), rng.random()]).toEqual([
      0.6011037519201636, 0.44829055899754167, 0.8524657934904099,
    ]);
    const wrapped = new Mulberry32(4294967295);
    expect(wrapped.random()).toBe(0.8964226141106337);
  });
});

describe("sampling", () => {
  const logits = Float32Array.from([1, 3, 2, 3, -1]);
  const all = Uint8Array.from([1, 1, 1, 1, 1]);
  it("ranks by logit then id", () => {
    expect(rankedCandidates(logits, all)).toEqual([1, 3, 2, 0, 4]);
  });
  it("is greedy at temperature 0 and respects the mask", () => {
    const greedy = { temperature: 0, topK: 0, topP: 1 };
    expect(sampleToken(logits, Uint8Array.from([1, 0, 1, 0, 1]), greedy, new Mulberry32(0))).toBe(
      2,
    );
  });
  it("throws when nothing is allowed", () => {
    expect(() =>
      sampleToken(
        logits,
        new Uint8Array(5),
        { temperature: 1, topK: 0, topP: 1 },
        new Mulberry32(0),
      ),
    ).toThrow(/no token/);
  });
});

describe("text", () => {
  it("splits syllables the same way for NFC, NFD and zero-width input", () => {
    const line = "Trăm năm trong cõi người ta,";
    expect(splitSyllables(line.normalize("NFD"))).toEqual(splitSyllables(line));
    expect(splitSyllables("ng\u200bười t\ufeffa")).toEqual(["người", "ta"]);
    expect(splitSyllables("ta \u{1F600} là")).toEqual(["ta", "là"]);
  });
  it("parses tones and canonicalizes tone placement", () => {
    expect(parseSyllable("nặng")?.tone).toBe(Tone.Nang);
    expect(canonicalSyllable("hoà")).toBe("hòa");
    expect(parseSyllable("photo")).toBeNull();
  });
  it("matches rhymes like the Python module", () => {
    const s = (w: string) => {
      const p = parseSyllable(w);
      if (p === null) throw new Error(w);
      return p;
    };
    expect(rhymeMatch(s("ta"), s("là"))).toBe(RhymeMatch.Exact);
    expect(rhymeMatch(s("nhau"), s("dâu"))).toBe(RhymeMatch.Near);
    expect(rhymeMatch(s("xanh"), s("vàng"))).toBe(RhymeMatch.None);
  });
  it("checks a poem and reports violations", () => {
    expect(
      checkLines(splitPoem("Trăm năm trong cõi người ta / Chữ tài chữ mệnh khéo là ghét nhau"))
        .violations,
    ).toEqual([]);
    const bad = checkLines([
      "Trăm năm trong ngày người ta",
      "Chữ tài chữ mệnh khéo lòng ghét nhau",
    ]);
    expect(bad.violations.map((v) => [v.line, v.rule, v.position])).toEqual([
      [1, "tone", 4],
      [2, "rhyme", 6],
    ]);
    expect(() => checkLines(splitPoem(" / "))).toThrow(/no lines/);
  });
});

describe("line slots and model file", () => {
  it("assigns slots like lyra/text/slots.py", () => {
    let state = START_SLOT_STATE;
    const slots: number[] = [];
    for (const t of [10, LINE6, 5, 5, 5, 5, 5, 5, 5, LINE8, 5]) {
      const [slot, next] = nextSlot(state, t);
      slots.push(slot);
      state = next;
    }
    expect(slots).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 0, 8, 9]);
  });
  it("rejects files that are not Lyra models", () => {
    expect(() => parseModelFile(new ArrayBuffer(4))).toThrow(ModelFormatError);
    const bytes = new Uint8Array(16);
    bytes.set([78, 79, 80, 69]);
    expect(() => parseModelFile(bytes.buffer)).toThrow(/not a Lyra/);
  });
});
