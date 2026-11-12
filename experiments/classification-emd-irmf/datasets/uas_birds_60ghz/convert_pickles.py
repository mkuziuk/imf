"""One-time, isolated numeric export of the authorized NumPy-only pickles.

Run with the worktree Python's -I flag from outside the raw directory. This
module never imports or executes downloaded code. --scan-only executes no
pickle instructions and can audit an extracted full DataPort release too.
"""

import argparse
import hashlib
import io
import json
import os
import pickle
import pickletools
from pathlib import Path

import numpy as np
from numpy._core.multiarray import _reconstruct

ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}
DEFAULT_ROOT = Path("/Users/mikhail/Datasets/imf-classification")
UNKNOWN = object()
MARK = object()


def scan_globals(data):
    """Resolve GLOBAL/STACK_GLOBAL symbolically; execute no pickle operations.

    pickletools supplies each opcode's stack effect. Only strings and memo
    references need values; arrays, reductions and other objects stay symbolic.
    Reject extension, persistent-ID, legacy constructor and external-buffer ops.
    """
    stack, memo, found = [], {}, set()
    blocked = {"EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID", "INST", "OBJ",
               "NEWOBJ", "NEWOBJ_EX", "NEXT_BUFFER", "READONLY_BUFFER"}
    for op, arg, _ in pickletools.genops(data):
        name = op.name
        if name in blocked:
            raise pickle.UnpicklingError(f"unsupported opcode {name}")
        if name in {"PROTO", "FRAME"}:
            continue
        if name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}:
            stack.append(arg)
        elif name == "MARK":
            stack.append(MARK)
        elif name in {"PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"}:
            memo[len(memo) if name == "MEMOIZE" else int(arg)] = stack[-1]
        elif name in {"GET", "BINGET", "LONG_BINGET"}:
            stack.append(memo[int(arg)])
        elif name in {"GLOBAL", "STACK_GLOBAL"}:
            if name == "GLOBAL":
                module, symbol = arg.split(" ", 1)
            else:
                symbol, module = stack.pop(), stack.pop()
            if not isinstance(module, str) or not isinstance(symbol, str):
                raise pickle.UnpicklingError("unresolved STACK_GLOBAL")
            pair = (module, symbol)
            if pair not in ALLOWED_GLOBALS:
                raise pickle.UnpicklingError(f"forbidden global {pair}")
            found.add(pair)
            stack.append(UNKNOWN)
        elif name == "DUP":
            stack.append(stack[-1])
        else:
            before = op.stack_before
            if pickletools.markobject in before:
                marker = len(stack) - 1 - stack[::-1].index(MARK)
                n_prefix = before.index(pickletools.markobject)
                del stack[marker - n_prefix:]
            elif before:
                del stack[-len(before):]
            stack.extend([UNKNOWN] * len(op.stack_after))
    return sorted(found)


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Only the three audited NumPy constructors, including the NumPy 2 alias."""

    def find_class(self, module, name):
        try:
            return ALLOWED_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"forbidden global {(module, name)}") from error

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent IDs are forbidden")


def validate_value(value):
    """Accept only numeric/string arrays or plain scalars and flat containers."""
    def check_leaf(leaf):
        if isinstance(leaf, np.ndarray):
            if leaf.dtype.hasobject:
                raise pickle.UnpicklingError("object dtype is forbidden")
        elif type(leaf) not in {bool, int, float, complex, str}:
            raise pickle.UnpicklingError(f"forbidden value type {type(leaf).__name__}")

    if isinstance(value, dict):
        for key, leaf in value.items():
            if type(key) not in {bool, int, float, str}:
                raise pickle.UnpicklingError("non-scalar dict key")
            check_leaf(leaf)
    elif isinstance(value, (tuple, list)):
        for leaf in value:
            check_leaf(leaf)
    else:
        check_leaf(value)


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def convert(source_root, destination, scan_only=False):
    source_root, destination = source_root.resolve(), destination.resolve()
    raw_root = next((parent for parent in [source_root, *source_root.parents]
                     if parent.name == "raw"), source_root)
    if Path.cwd().resolve().is_relative_to(raw_root):
        raise RuntimeError("run from outside the raw source directory")
    if destination.is_relative_to(raw_root):
        raise RuntimeError("converted output must be outside the raw source directory")
    files = sorted(source_root.rglob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no pickles under {source_root}")
    if len({path.stem for path in files}) != len(files):
        raise ValueError("duplicate source stems; use a narrower source root")
    manifest_path = destination / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    old = {row["source_file"]: row for row in previous.get("files", [])}
    records = []
    for path in files:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        globals_used = scan_globals(data)
        relative = str(path.relative_to(source_root))
        if scan_only:
            records.append({"source_file": relative, "source_sha256": digest,
                            "globals": globals_used})
            continue
        prior = old.get(relative)
        if prior and prior["source_sha256"] == digest:
            output = destination / prior["output_file"]
            if output.exists() and sha256_file(output) == prior["output_sha256"]:
                records.append(prior)
                continue
            raise RuntimeError(f"converted output changed or missing: {output}")
        if prior:
            raise RuntimeError(f"source changed after conversion: {relative}")
        value = NumpyOnlyUnpickler(io.BytesIO(data)).load()
        validate_value(value)
        destination.mkdir(parents=True, exist_ok=True)
        if isinstance(value, (dict, list, tuple)):
            leaves = list(value.values()) if isinstance(value, dict) else list(value)
            arrays = {f"array_{i:04d}": np.asarray(v) for i, v in enumerate(leaves)}
            if any(array.dtype.hasobject for array in arrays.values()):
                raise pickle.UnpicklingError("object dtype is forbidden")
            output = destination / f"{path.stem}.npz"
            if output.exists():
                raise FileExistsError(output)
            with output.open("xb") as handle:
                np.savez(handle, allow_pickle=False, **arrays)
            descriptions = {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                            for k, v in arrays.items()}
        else:
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise pickle.UnpicklingError("object dtype is forbidden")
            output = destination / f"{path.stem}.npy"
            if output.exists():
                raise FileExistsError(output)
            with output.open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
            descriptions = {"shape": list(array.shape), "dtype": str(array.dtype)}
        records.append({"source_file": relative, "source_sha256": digest,
                        "source_bytes": len(data), "globals": globals_used,
                        "output_file": output.name, "output_sha256": sha256_file(output),
                        "container": type(value).__name__, "arrays": descriptions})
        if isinstance(value, dict):
            records[-1]["dict_keys"] = list(value)
        # Persist after every file so a completed conversion is never repeated.
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"source_root": str(source_root),
                                        "allowed_globals": sorted(ALLOWED_GLOBALS),
                                        "files": records}, indent=2) + "\n")
        os.replace(temporary, manifest_path)
    summary = {"n_files": len(records), "globals": sorted({tuple(g) for r in records for g in r["globals"]}),
               "scan_only": scan_only}
    print(json.dumps(summary, indent=2))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(os.environ.get("IMF_CLS_DATA", DEFAULT_ROOT)) / "uas_birds_60ghz"
    parser.add_argument("--source-root", type=Path,
                        default=base / "raw/github_sk8053_drone_bird_detection_with_radar")
    parser.add_argument("--destination", type=Path, default=base / "converted")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    if not sys_isolated():
        parser.error("use Python -I for conversion and scanning")
    convert(args.source_root, args.destination, args.scan_only)


def sys_isolated():
    import sys

    return bool(sys.flags.isolated)


if __name__ == "__main__":
    main()
