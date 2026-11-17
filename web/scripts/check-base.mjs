// Fail the build if the page would break under the GitHub Pages sub-path:
// every local asset reference in dist/index.html must start with the base.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const BASE = "/lyra-neural-verse/";
const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "..", "dist", "index.html"), "utf8");
const refs = [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map((m) => m[1]);
const local = refs.filter((r) => !/^(https?:)?\/\//.test(r) && !r.startsWith("#"));
const bad = local.filter((r) => !r.startsWith(BASE));
if (local.length === 0 || bad.length > 0) {
  console.error(`base path check failed; offending references: ${JSON.stringify(bad)}`);
  process.exit(1);
}
console.log(`base path check passed (${local.length} references under ${BASE})`);
