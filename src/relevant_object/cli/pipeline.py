"""Reserved console entry point for the incremental pipeline migration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from relevant_object import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relevant-object",
        description=(
            "The package scaffold is installed. The Python pipeline runner will "
            "be enabled after the compatibility-preserving stage migration; use "
            "run_full_pipeline.sh in the meantime."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

