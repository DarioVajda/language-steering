"""Download the raw benchmark files (MGSM, MSVAMP, XNLI) from the Hugging Face Hub,
plus the original cased SVAMP release from GitHub (used to truecase MSVAMP English),
MKQA from GitHub, and the fastText language-ID model (both for the output-language experiment).

Files are stored unmodified under data/raw_data/<name>/, and the exact repo commit
(or, for fastText, the SHA-256) of every download is recorded in data/raw_data/manifest.json.

Usage: python src/data/download.py [name ...]   (default: everything)
"""

import hashlib
import json
import sys
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
# Pinned commit of https://github.com/apple/ml-mkqa (Longpre et al., 2021).
MKQA_COMMIT = "f964fe1bac4d580cee3d4928572df1f676f8e656"
MKQA_URL = f"https://raw.githubusercontent.com/apple/ml-mkqa/{MKQA_COMMIT}/dataset/mkqa.jsonl.gz"
# fastText language identification, compressed model (Joulin et al., 2016); not versioned, so checked by hash.
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
LID_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
FILES = {  # name -> (url, file name, revision, expected sha256 or None)
    "svamp": (SVAMP_URL, "SVAMP.json", SVAMP_COMMIT, None),
    "mkqa": (MKQA_URL, "mkqa.jsonl.gz", MKQA_COMMIT, None),
    "fasttext": (LID_URL, "lid.176.ftz", f"sha256:{LID_SHA256}", LID_SHA256),
}


def main():
    names = sys.argv[1:] or [*SOURCES, *FILES]
    unknown = set(names) - set(SOURCES) - set(FILES)
    if unknown:
        sys.exit(f"unknown source(s): {sorted(unknown)}; choose from {[*SOURCES, *FILES]}")
    manifest_path = RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    api = HfApi()
    for name in [n for n in SOURCES if n in names]:
        src = SOURCES[name]
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

    for name in [n for n in FILES if n in names]:
        url, fname, revision, sha256 = FILES[name]
        out = RAW_DIR / name / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, out)
        if sha256 and hashlib.sha256(out.read_bytes()).hexdigest() != sha256:
            sys.exit(f"{name}: SHA-256 mismatch for {url}")
        manifest[name] = {"url": url, "revision": revision}
        print(f"{name}: {url} -> {out.relative_to(ROOT)}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
