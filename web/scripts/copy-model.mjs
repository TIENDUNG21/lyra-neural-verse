// Copy the released model file (the single source of truth in ../models) and
// the logo (../assets) into public/ so Vite serves them next to the app.
// public/ is generated and not committed.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "..", "..", "models", "lyra-kieu-q8.bin");
const target = join(here, "..", "public", "model", "lyra-kieu-q8.bin");
mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
copyFileSync(join(here, "..", "..", "assets", "logo.svg"), join(here, "..", "public", "logo.svg"));
console.log(`copied ${source} -> ${target} and the logo`);
