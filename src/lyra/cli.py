"""Command-line interface: ``lyra train | sample | check | export | evaluate | plot``.

Heavy modules are imported inside the command functions so that the BLAS
thread count (``--threads``) is set before NumPy is loaded.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_MODEL = Path("models") / "lyra-kieu-q8.bin"
DEFAULT_CORPUS = Path("data") / "truyen-kieu.txt"
DEFAULT_WAIVERS = Path("data") / "kieu-exceptions.tsv"
_THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _set_threads(threads: int) -> None:
    for name in _THREAD_VARS:
        os.environ[name] = str(threads)


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def cmd_train(args: argparse.Namespace) -> int:
    from lyra.train.loop import load_run_config, train

    model_options, config = load_run_config(args.config)
    if args.max_minutes is not None:
        config = type(config).from_dict({**config.to_dict(), "max_minutes": args.max_minutes})
    result = train(
        corpus_path=args.corpus,
        model_options=model_options,
        config=config,
        out_dir=args.out,
        resume=args.resume,
    )
    print(
        f"done: {result.steps} steps in {result.elapsed_s / 60:.1f} min; best val loss "
        f"{result.best_val_loss:.4f} at step {result.best_step} "
        f"(syllable perplexity {result.best_val_syllable_ppl:.1f})"
    )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from lyra.text.lucbat import check_poem, load_waivers

    if args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.poem:
        text = " / ".join(args.poem)
    else:
        text = sys.stdin.read()
    waivers = load_waivers(args.waivers) if args.waivers else []
    try:
        report = check_poem(text, waivers=waivers)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not args.quiet:
        for line in report.lines:
            marks = "".join("?" if s is None else ("B" if s.is_bang else "T") for s in line.parsed)
            print(
                f"{line.number:>5} [{line.expected_length}] {marks:<8} {' '.join(line.syllables)}"
            )
    for violation in report.violations:
        where = f", syllable {violation.position}" if violation.position else ""
        print(f"line {violation.line}{where}: {violation.rule.value}: {violation.message}")
    exact = sum(1 for r in report.rhymes if r.match.name == "EXACT")
    near = sum(1 for r in report.rhymes if r.match.name == "NEAR")
    print(
        f"{len(report.lines)} lines, {len(report.violations)} violations, "
        f"{len(report.waived)} waived; rhymes: {exact} exact, {near} near"
    )
    if report.unused_waivers:
        print(f"warning: {len(report.unused_waivers)} waivers were not needed", file=sys.stderr)
    return 0 if report.ok else 1


def cmd_plot(args: argparse.Namespace) -> int:
    from lyra.train.csvlog import read_log
    from lyra.train.plot import render_loss_svg

    svg = render_loss_svg(read_log(args.log), title=args.title, subtitle=args.subtitle)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(svg, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from lyra.decode.generate import Poet
    from lyra.decode.sampling import SamplingSettings
    from lyra.text.lucbat import check_lines

    poet = Poet.from_file(args.model)
    settings = SamplingSettings(args.temperature, args.top_k, args.top_p)
    prompt = [p for line in args.prompt for p in line.split("/") if p.strip()]
    for index in range(args.count):
        try:
            poem = poet.generate(
                lines=args.lines,
                prompt=prompt,
                settings=settings,
                seed=args.seed + index,
                strict_rhyme=args.strict_rhyme,
            )
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        if index:
            print()
        print(poem.text(), flush=True)
        report = check_lines([line.text() for line in poem.lines])
        exact = sum(1 for r in report.rhymes if r.match.name == "EXACT")
        relaxed = sum(1 for line in poem.lines if line.relaxed)
        print(
            f"-- seed {args.seed + index}: {len(report.violations)} rule violations, "
            f"{exact}/{len(report.rhymes)} exact rhymes, {relaxed} relaxed lines",
            file=sys.stderr,
        )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from lyra.io.modelfile import write_model_file
    from lyra.nn import LyraLM, ModelConfig
    from lyra.optim import load_checkpoint
    from lyra.text.features import token_features
    from lyra.text.tokenizer import Tokenizer

    ckpt = load_checkpoint(args.checkpoint)
    tokenizer = Tokenizer.load(args.tokenizer)
    model = LyraLM(ModelConfig.from_dict(ckpt.config), token_features(tokenizer.vocab))
    model.load_state_dict(ckpt.params)
    config, state = model.folded_state_dict()
    metadata = {"step": ckpt.step, "parameters": model.num_parameters(), **ckpt.extra}
    size = write_model_file(
        args.out, params=state, config=config, vocab=list(tokenizer.vocab), metadata=metadata
    )
    print(f"wrote {args.out} ({size / 1e6:.2f} MB, step {ckpt.step})")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    import json

    from lyra.train.evaluate import evaluate_file

    result = evaluate_file(args.model, args.corpus)
    print(json.dumps({k: round(v, 3) for k, v in result.items()}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lyra", description="A NumPy language model that writes Vietnamese lục bát verse."
    )
    parser.add_argument("--threads", type=int, default=6, help="BLAS threads (default 6)")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a model on the corpus")
    train.add_argument("--config", type=Path, required=True, help="JSON run config")
    train.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    train.add_argument("--out", type=Path, required=True, help="run directory")
    train.add_argument("--resume", action="store_true", help="continue from <out>/last.npz")
    train.add_argument("--max-minutes", type=float, default=None, help="wall-clock budget")
    train.set_defaults(func=cmd_train)

    check = commands.add_parser("check", help="check a poem against the lục bát rules")
    check.add_argument("poem", nargs="*", help="poem lines (or one string with '/' separators)")
    check.add_argument("--file", type=Path, default=None, help="read the poem from a file")
    check.add_argument("--waivers", type=Path, default=None, help="TSV of documented exceptions")
    check.add_argument("--quiet", action="store_true", help="only print problems and a summary")
    check.set_defaults(func=cmd_check)

    sample = commands.add_parser("sample", help="write lục bát with a trained model")
    sample.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    sample.add_argument("--lines", type=int, default=4, help="total number of lines")
    sample.add_argument("--prompt", action="append", default=[], help="first line(s) to continue")
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--count", type=int, default=1, help="poems to write (seeds seed..)")
    sample.add_argument("--temperature", type=float, default=0.9)
    sample.add_argument("--top-k", type=int, default=40)
    sample.add_argument("--top-p", type=float, default=0.95)
    sample.add_argument("--strict-rhyme", action="store_true", help="exact rhymes only")
    sample.set_defaults(func=cmd_sample)

    export = commands.add_parser("export", help="write the int8 model file from a checkpoint")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--tokenizer", type=Path, required=True)
    export.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    export.set_defaults(func=cmd_export)

    evaluate = commands.add_parser("evaluate", help="validation perplexity and n-gram baselines")
    evaluate.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    evaluate.set_defaults(func=cmd_evaluate)

    plot = commands.add_parser("plot", help="render a training log as an SVG loss curve")
    plot.add_argument("--log", type=Path, required=True, help="log.csv written by lyra train")
    plot.add_argument("--out", type=Path, required=True, help="SVG file to write")
    plot.add_argument("--title", default="Lyra training loss")
    plot.add_argument("--subtitle", default="Cross-entropy per token on Truyện Kiều")
    plot.set_defaults(func=cmd_plot)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    _set_threads(args.threads)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
