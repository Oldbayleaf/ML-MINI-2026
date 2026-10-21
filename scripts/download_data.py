"""Download the UCI Cleveland heart-disease data and verify it against a known checksum.

Run once, from the project folder:

    .venv/bin/python scripts/download_data.py      # (Windows: .venv\\Scripts\\python scripts\\download_data.py)

It fetches the UCI archive, extracts `processed.cleveland.data`, and checks its SHA256
against the value recorded below. A mismatch aborts loudly, so a corrupted or silently
changed download can never feed the replication. The file is already present in `data/`,
so you only need this if you want to re-fetch from source.
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UCI_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
TARGET = "processed.cleveland.data"

# SHA256 of the extracted Cleveland file (303 rows, 14 columns, '?' for missing).
SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / TARGET
    if dest.exists():
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest == SHA256:
            print(f"OK: {dest} already present and checksum-verified.")
            return 0
        print(f"WARNING: {dest} exists but its checksum differs ({digest[:12]}...). Re-fetching.")

    print(f"Downloading {UCI_URL} ...")
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
        archive = resp.read()

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        member = next(n for n in zf.namelist() if n.endswith(TARGET))
        payload = zf.read(member)

    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        print(f"CHECKSUM MISMATCH\n  expected {SHA256}\n  got      {digest}", file=sys.stderr)
        return 1

    dest.write_bytes(payload)
    print(f"Wrote {dest} ({len(payload)} bytes), checksum verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
