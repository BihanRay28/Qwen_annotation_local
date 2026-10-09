"""Publish only the generated labels slot while inference mounts all source images read-only."""

import argparse
import os
import tempfile
import time
from pathlib import Path

from prepare_full_dataset import owned_labels


def publish(source, dataset, filename="labels.json"):
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    if filename not in {"labels.json", "labels.py"} or dataset == source or dataset in source.parents:
        raise ValueError("Publication needs an external generated JSON and a fixed labels filename")
    if not source.exists():
        return False
    if not owned_labels(source):
        raise ValueError("Only recognised generated labels JSON can be published")
    destination = dataset / filename
    if destination.is_symlink() or destination.resolve().parent != dataset:
        raise ValueError("Labels path escapes dataset root")
    if destination.exists() and not owned_labels(destination):
        raise ValueError("An existing original dataset file will not be overwritten")
    descriptor, name = tempfile.mkstemp(prefix="." + filename + ".publishing-", dir=dataset)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as outgoing, source.open("rb") as incoming:
            while block := incoming.read(1024 * 1024):
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--filename", default="labels.json")
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    previous = None
    while True:
        current = args.source.stat().st_mtime_ns if args.source.exists() else None
        if current is not None and current != previous:
            publish(args.source, args.dataset, args.filename)
            previous = current
        if not args.watch:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
