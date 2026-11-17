// Incremental transformer forward pass with a KV cache.
// A line-by-line port of lyra/nn/inference.py; test/parity.test.ts compares
// its logits with the Python engine on fixtures written by
// scripts/make_parity_fixture.py.

import { LINE6, LINE8 } from "../text/tokenizer";
import type { ModelConfig, ModelFile, Tensor } from "./format";

const GELU_C = Math.sqrt(2 / Math.PI);

export interface SlotState {
  readonly lineLength: number;
  readonly position: number;
}

export const START_SLOT_STATE: SlotState = { lineLength: 0, position: 0 };

/** Line slot of a token (see lyra/text/slots.py) and the following state. */
export function nextSlot(state: SlotState, token: number): [number, SlotState] {
  if (token === LINE6) return [1, { lineLength: 6, position: 0 }];
  if (token === LINE8) return [8, { lineLength: 8, position: 0 }];
  if (state.lineLength === 0) return [0, state];
  const position = state.position + 1;
  const next = { lineLength: state.lineLength, position };
  if (position > state.lineLength) return [0, next];
  return [(state.lineLength === 6 ? 1 : 8) + position, next];
}

function layerNorm(x: Float32Array, weight: Float32Array, bias: Float32Array): Float32Array {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i] ?? 0;
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const c = (x[i] ?? 0) - mean;
    variance += c * c;
  }
  variance /= n;
  const inv = 1 / Math.sqrt(variance + 1e-5);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++)
    out[i] = ((x[i] ?? 0) - mean) * inv * (weight[i] ?? 0) + (bias[i] ?? 0);
  return out;
}

/** out = x @ W + b for one row; W is (inDim, outDim) row-major. */
function linear(x: Float32Array, w: Tensor, b: Tensor): Float32Array {
  const inDim = w.shape[0] ?? 0;
  const outDim = w.shape[1] ?? 0;
  const acc = new Float64Array(outDim);
  const wd = w.data;
  for (let k = 0; k < inDim; k++) {
    const a = x[k] ?? 0;
    if (a === 0) continue;
    const row = k * outDim;
    for (let j = 0; j < outDim; j++) acc[j] = (acc[j] ?? 0) + a * (wd[row + j] ?? 0);
  }
  const out = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) out[j] = (acc[j] ?? 0) + (b.data[j] ?? 0);
  return out;
}

export class Engine {
  readonly config: ModelConfig;
  private readonly w: ReadonlyMap<string, Tensor>;
  private readonly headDim: number;
  private keys: Float32Array[] = [];
  private values: Float32Array[] = [];
  tokens: number[] = [];
  slots: number[] = [];
  slotState: SlotState = START_SLOT_STATE;

  constructor(model: ModelFile) {
    this.config = model.config;
    this.w = model.tensors;
    this.headDim = model.config.d_model / model.config.n_heads;
    this.reset();
  }

  get length(): number {
    return this.tokens.length;
  }

  reset(): void {
    const size = this.config.block_size * this.config.d_model;
    this.keys = Array.from({ length: this.config.n_layers }, () => new Float32Array(size));
    this.values = Array.from({ length: this.config.n_layers }, () => new Float32Array(size));
    this.tokens = [];
    this.slots = [];
    this.slotState = START_SLOT_STATE;
  }

  private t(name: string): Tensor {
    const tensor = this.w.get(name);
    if (tensor === undefined) throw new Error(`missing tensor ${name}`);
    return tensor;
  }

  /** Append tokens to the context and return the logits for the next token. */
  feed(tokens: readonly number[]): Float32Array {
    if (tokens.length === 0) throw new Error("feed() needs at least one token");
    const newSlots: number[] = [];
    for (const token of tokens) {
      const [slot, state] = nextSlot(this.slotState, token);
      newSlots.push(slot);
      this.slotState = state;
    }
    let logits: Float32Array | null = null;
    for (let i = 0; i < tokens.length; i++) {
      if (this.length === this.config.block_size) {
        const half = Math.floor(this.config.block_size / 2);
        const keep = this.tokens.slice(-half);
        const keepSlots = this.slots.slice(-half);
        const state = this.slotState;
        this.reset();
        this.slotState = state;
        for (let j = 0; j < keep.length; j++) this.step(keep[j] ?? 0, keepSlots[j] ?? 0, false);
      }
      logits = this.step(tokens[i] ?? 0, newSlots[i] ?? 0, i === tokens.length - 1);
    }
    return logits ?? new Float32Array(0);
  }

  /** Process one token at the next position; returns logits if requested. */
  private step(token: number, slot: number, wantLogits: boolean): Float32Array | null {
    const { d_model: d, n_heads: heads, n_layers: layers, vocab_size: vocab } = this.config;
    if (token < 0 || token >= vocab) throw new RangeError("token id out of range");
    const pos = this.length;
    const D = this.headDim;
    const tok = this.t("tok_emb.weight").data;
    const posEmb = this.t("pos_emb.weight").data;
    const slotEmb = this.w.get("slot_emb.weight")?.data;
    let x = new Float32Array(d);
    for (let k = 0; k < d; k++) {
      let v = (tok[token * d + k] ?? 0) + (posEmb[pos * d + k] ?? 0);
      if (slotEmb) v += slotEmb[slot * d + k] ?? 0;
      x[k] = v;
    }
    const scale = 1 / Math.sqrt(D);
    for (let layer = 0; layer < layers; layer++) {
      const p = `blocks.${layer}.`;
      const h = layerNorm(x, this.t(`${p}ln1.weight`).data, this.t(`${p}ln1.bias`).data);
      const qkv = linear(h, this.t(`${p}attn.qkv.weight`), this.t(`${p}attn.qkv.bias`));
      const K = this.keys[layer] as Float32Array;
      const V = this.values[layer] as Float32Array;
      // Cache layout: [position][head * D + i], i.e. the k and v parts of qkv.
      K.set(qkv.subarray(d, 2 * d), pos * d);
      V.set(qkv.subarray(2 * d, 3 * d), pos * d);
      const y = new Float32Array(d);
      const scores = new Float64Array(pos + 1);
      for (let head = 0; head < heads; head++) {
        const off = head * D;
        let max = Number.NEGATIVE_INFINITY;
        for (let j = 0; j <= pos; j++) {
          let s = 0;
          for (let i = 0; i < D; i++) s += (qkv[off + i] ?? 0) * (K[j * d + off + i] ?? 0);
          s *= scale;
          scores[j] = s;
          if (s > max) max = s;
        }
        let total = 0;
        for (let j = 0; j <= pos; j++) {
          const e = Math.exp((scores[j] ?? 0) - max);
          scores[j] = e;
          total += e;
        }
        for (let i = 0; i < D; i++) {
          let acc = 0;
          for (let j = 0; j <= pos; j++) acc += (scores[j] ?? 0) * (V[j * d + off + i] ?? 0);
          y[off + i] = acc / total;
        }
      }
      const attn = linear(y, this.t(`${p}attn.proj.weight`), this.t(`${p}attn.proj.bias`));
      for (let k = 0; k < d; k++) attn[k] = (x[k] ?? 0) + (attn[k] ?? 0);
      x = attn;
      const h2 = layerNorm(x, this.t(`${p}ln2.weight`).data, this.t(`${p}ln2.bias`).data);
      const fc = linear(h2, this.t(`${p}mlp.fc.weight`), this.t(`${p}mlp.fc.bias`));
      for (let k = 0; k < fc.length; k++) {
        const a = fc[k] ?? 0;
        fc[k] = 0.5 * a * (1 + Math.tanh(GELU_C * a * (1 + 0.044715 * a * a)));
      }
      const mlp = linear(fc, this.t(`${p}mlp.proj.weight`), this.t(`${p}mlp.proj.bias`));
      for (let k = 0; k < d; k++) mlp[k] = (x[k] ?? 0) + (mlp[k] ?? 0);
      x = mlp;
    }
    this.tokens.push(token);
    this.slots.push(slot);
    if (!wantLogits) return null;
    const final = layerNorm(x, this.t("ln_f.weight").data, this.t("ln_f.bias").data);
    const bias = this.t("head_bias").data;
    const logits = new Float32Array(vocab);
    for (let v = 0; v < vocab; v++) {
      let acc = 0;
      const row = v * d;
      for (let k = 0; k < d; k++) acc += (final[k] ?? 0) * (tok[row + k] ?? 0);
      logits[v] = acc + (bias[v] ?? 0);
    }
    return logits;
  }
}
