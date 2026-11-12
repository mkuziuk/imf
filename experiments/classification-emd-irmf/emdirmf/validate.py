"""CLI: python -m emdirmf.validate NAME [NAME ...] [--json]."""

import argparse
import json
import sys

from .io import validate_prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate prepared datasets against PROTOCOL.md")
    parser.add_argument("names", nargs="+", help="dataset folder names under DATA_ROOT")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--skip-values", action="store_true", help="skip the finite-value scan")
    args = parser.parse_args(argv)
    ok = True
    for name in args.names:
        report = validate_prepared(name, verbose=not args.json, check_values=not args.skip_values)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        ok &= report["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
