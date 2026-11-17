import "./styles.css";
import { Poet } from "./decode/generate";
import { Engine } from "./model/engine";
import { parseModelFile } from "./model/format";
import { renderCheck, renderPoem } from "./ui/poem-view";

const MODEL_URL = `${import.meta.env.BASE_URL}model/lyra-kieu-q8.bin`;

function byId<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (node === null) throw new Error(`missing #${id}`);
  return node as T;
}

const form = byId<HTMLFormElement>("controls");
const status = byId<HTMLDivElement>("status");
const poemView = byId<HTMLElement>("poem");
const writeButton = byId<HTMLButtonElement>("write");
const checkerInput = byId<HTMLTextAreaElement>("checker-input");
const checkerOutput = byId<HTMLDivElement>("checker-output");

function setStatus(text: string, error = false): void {
  status.textContent = text;
  status.classList.toggle("error", error);
}

function field(name: string): HTMLInputElement {
  const node = form.elements.namedItem(name);
  if (!(node instanceof HTMLInputElement || node instanceof HTMLSelectElement)) {
    throw new Error(`missing field ${name}`);
  }
  return node as HTMLInputElement;
}

function write(poet: Poet): void {
  const prompt = field("prompt").value.trim();
  const started = performance.now();
  try {
    const poem = poet.generate({
      lines: Number(field("lines").value),
      seed: Number(field("seed").value) >>> 0,
      prompt: prompt ? [prompt] : [],
      strictRhyme: field("strict").checked,
      settings: {
        temperature: Number(field("temperature").value),
        topK: Math.max(0, Math.floor(Number(field("topK").value))),
        topP: Math.min(1, Math.max(0.05, Number(field("topP").value))),
      },
    });
    renderPoem(poemView, poem);
    const ms = performance.now() - started;
    const sampled = poem.lines.reduce((n, l) => n + l.confidence.length, 0);
    setStatus(`${sampled} syllables sampled in ${ms.toFixed(0)} ms on this device.`);
  } catch (error) {
    setStatus((error as Error).message, true);
  }
}

async function start(): Promise<void> {
  renderCheck(checkerOutput, checkerInput.value);
  checkerInput.addEventListener("input", () => renderCheck(checkerOutput, checkerInput.value));
  field("temperature").addEventListener("input", () => {
    const out = form.elements.namedItem("tOut");
    if (out instanceof HTMLOutputElement) out.value = field("temperature").value;
  });
  byId<HTMLButtonElement>("shuffle").addEventListener("click", () => {
    field("seed").value = String(Math.floor(Math.random() * 1_000_000));
    form.requestSubmit();
  });

  writeButton.disabled = true;
  const response = await fetch(MODEL_URL);
  if (!response.ok) throw new Error(`could not load the model (${response.status})`);
  const model = parseModelFile(await response.arrayBuffer());
  const poet = new Poet(new Engine(model), model.vocab);
  const meta = model.metadata as Record<string, unknown>;
  const params = typeof meta.parameters === "number" ? meta.parameters.toLocaleString("en") : "?";
  byId("model-info").textContent =
    `Model: ${params} parameters, ${model.config.n_layers} layers, d=${model.config.d_model}, ` +
    `vocabulary of ${model.vocab.length} syllables, int8 weights.`;
  writeButton.disabled = false;
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    setStatus("Writing...");
    window.setTimeout(() => write(poet), 20);
  });
  write(poet);
}

start().catch((error: unknown) => setStatus((error as Error).message, true));
