# Lyra playground

The browser half of Lyra: a TypeScript port of the inference engine, the Vietnamese
syllable parser, the lục bát checker and the constrained decoder. It loads the same
int8 file as the Python CLI (`../models/lyra-kieu-q8.bin`, copied into `public/` before
`dev`, `test` and `build`).

```sh
npm ci
npm run dev        # http://localhost:5110/lyra-neural-verse/
npm test           # unit tests + parity with web/test/fixtures/parity.json
npm run build      # dist/ for GitHub Pages, base /lyra-neural-verse/
```

`test/parity.test.ts` compares logits (tolerance 1e-3) and whole generated token
sequences with the fixture written by `python scripts/make_parity_fixture.py`. Regenerate
the fixture whenever the model file or the decoding code changes; the Python test suite
fails if it is stale.
