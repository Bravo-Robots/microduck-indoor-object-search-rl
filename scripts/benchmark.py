"""Compatibility launcher. Prefer: bash run.sh benchmark --help."""

from duck_search.__main__ import main

if __name__ == "__main__":
    import sys

    main(["benchmark", *sys.argv[1:]])
