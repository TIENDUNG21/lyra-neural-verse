import { defineConfig } from "vitest/config";

// The site is served from https://tiendung21.github.io/lyra-neural-verse/
export default defineConfig({
  base: "/lyra-neural-verse/",
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
  test: { include: ["test/**/*.test.ts"], environment: "node", testTimeout: 30000 },
});
