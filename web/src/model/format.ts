// Reader for the Lyra model file (format documented in lyra/io/modelfile.py).
//
//   "LYRA" | u32 version | u32 header length | header JSON | pad to 16 | tensor data
//
// Matrices are int8 with one float32 scale per output channel; vectors are
// float32. Everything is dequantized to Float32Array at load time.

export interface ModelConfig {
  vocab_size: number;
  block_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  mlp_ratio: number;
}

export interface Tensor {
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

export interface ModelFile {
  readonly config: ModelConfig;
  readonly vocab: readonly string[];
  readonly tensors: ReadonlyMap<string, Tensor>;
  readonly metadata: Readonly<Record<string, unknown>>;
}

interface TensorEntry {
  name: string;
  shape: number[];
  dtype: "int8" | "float32";
  offset: number;
  axis?: number;
  scale_offset?: number;
}

const ALIGN = 16;

export class ModelFormatError extends Error {}

export function parseModelFile(buffer: ArrayBuffer): ModelFile {
  if (buffer.byteLength < 12) throw new ModelFormatError("file too short");
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.subarray(0, 4));
  if (magic !== "LYRA") throw new ModelFormatError("not a Lyra model file");
  const view = new DataView(buffer);
  const version = view.getUint32(4, true);
  if (version !== 1) throw new ModelFormatError(`unsupported version ${version}`);
  const headerLength = view.getUint32(8, true);
  if (12 + headerLength > buffer.byteLength) throw new ModelFormatError("truncated header");
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + headerLength)));
  let base = 12 + headerLength;
  base += (ALIGN - (base % ALIGN)) % ALIGN;

  const tensors = new Map<string, Tensor>();
  for (const entry of header.tensors as TensorEntry[]) {
    const count = entry.shape.reduce((a, b) => a * b, 1);
    const start = base + entry.offset;
    const data = new Float32Array(count);
    if (entry.dtype === "int8") {
      if (start + count > buffer.byteLength) throw new ModelFormatError(`truncated ${entry.name}`);
      const axis = entry.axis ?? 1;
      const [rows = 0, cols = 0] = entry.shape;
      const q = new Int8Array(buffer, start, count);
      const scaleStart = base + (entry.scale_offset ?? 0);
      const channels = axis === 0 ? rows : cols;
      const scales = readFloat32(view, scaleStart, channels);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const i = r * cols + c;
          data[i] = (q[i] ?? 0) * (scales[axis === 0 ? r : c] ?? 0);
        }
      }
    } else if (entry.dtype === "float32") {
      if (start + 4 * count > buffer.byteLength)
        throw new ModelFormatError(`truncated ${entry.name}`);
      data.set(readFloat32(view, start, count));
    } else {
      throw new ModelFormatError(`unknown dtype ${String(entry.dtype)}`);
    }
    tensors.set(entry.name, { shape: entry.shape, data });
  }
  return {
    config: header.config as ModelConfig,
    vocab: header.vocab as string[],
    tensors,
    metadata: (header.metadata ?? {}) as Record<string, unknown>,
  };
}

function readFloat32(view: DataView, offset: number, count: number): Float32Array {
  if (offset + 4 * count > view.byteLength) throw new ModelFormatError("truncated data");
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = view.getFloat32(offset + 4 * i, true);
  return out;
}
