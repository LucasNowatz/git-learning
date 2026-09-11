"""Recompute the file inventory in environment/data/data_manifest.json.

Run after any change to the agent-visible dataset.  The manifest never lists
itself.
"""
import hashlib, json, os, sys

ENVD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "environment", "data")
ENVD = os.path.normpath(ENVD)
MANIFEST = os.path.join(ENVD, "data_manifest.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    man = json.load(open(MANIFEST))
    files = {}
    for dirpath, _, names in os.walk(ENVD):
        for nm in sorted(names):
            p = os.path.join(dirpath, nm)
            rel = os.path.relpath(p, ENVD)
            if rel == "data_manifest.json":
                continue
            files[rel] = dict(sha256=sha256(p), bytes=os.path.getsize(p))
    man["files"] = files
    man["file_count"] = len(files)
    with open(MANIFEST, "w") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    print(f"manifest lists {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
