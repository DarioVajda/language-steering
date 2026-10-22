"""Download the raw benchmark files (MGSM, MSVAMP, XNLI) from the Hugging Face Hub,
plus the original cased SVAMP release from GitHub (used to truecase MSVAMP English).

Files are stored unmodified under data/raw_data/<name>/, and the exact repo commit
of every download is recorded in data/raw_data/manifest.json.

Usage: python src/data/download.py
"""

import json
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw_data"

SOURCES = {
    "mgsm": {"repo_id": "juletxara/mgsm", "allow_patterns": ["*/*.parquet", "README.md"]},
    "msvamp": {"repo_id": "Mathoctopus/MSVAMP", "allow_patterns": ["*.json", "README.md"]},
    # XNLI train is machine-translated MultiNLI (~400k rows per language) and is not used.
    "xnli": {"repo_id": "facebook/xnli", "allow_patterns": ["*/validation-*.parquet", "*/test-*.parquet", "README.md"]},
}

# Pinned commit of https://github.com/arkilpatel/SVAMP (Patel et al., 2021).
SVAMP_COMMIT = "78e727689e1c1bebfc4be39c446898e8e10b0518"
SVAMP_URL = f"https://raw.githubusercontent.com/arkilpatel/SVAMP/{SVAMP_COMMIT}/SVAMP.json"


def main():
    api = HfApi()
    manifest = {}
    for name, src in SOURCES.items():
        sha = api.dataset_info(src["repo_id"]).sha
        out = RAW_DIR / name
        snapshot_download(
            repo_id=src["repo_id"],
            repo_type="dataset",
            revision=sha,
            allow_patterns=src["allow_patterns"],
            local_dir=out,
        )
        manifest[name] = {"repo_id": src["repo_id"], "revision": sha}
        print(f"{name}: {src['repo_id']}@{sha[:8]} -> {out.relative_to(ROOT)}")

    out = RAW_DIR / "svamp" / "SVAMP.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SVAMP_URL, out)
    manifest["svamp"] = {"url": SVAMP_URL, "revision": SVAMP_COMMIT}
    print(f"svamp: arkilpatel/SVAMP@{SVAMP_COMMIT[:8]} -> {out.relative_to(ROOT)}")
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
