"""Data and starter-kit access: rows, images and the organisers' code, all at pinned revisions."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path

from . import settings as S


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:      # a line cut short by a killed worker; it is redone
                continue
    return rows


def read_jsonl_many(folder: Path, pattern: str = "worker_*.jsonl") -> list[dict]:
    return [row for path in sorted(Path(folder).glob(pattern)) for row in read_jsonl(path)]


def write_jsonl(path: Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------- official starter kit
def official_repo(workdir: Path) -> Path:
    """Clone the organisers' repository at the pinned commit (baseline, format checker, scorer)."""
    path = Path(workdir) / "mmcultureqa"
    if not (path / ".git").exists():
        subprocess.run(["git", "clone", "-q", S.OFFICIAL_REPO_URL, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", S.OFFICIAL_COMMIT], check=True)
    return path


def import_official_baseline(repo: Path):
    """Import ``baselines/baseline_qwen_omni.py`` as released.

    The released file decorates ``answer`` with ``@torch.no_grad()`` but imports torch only
    inside functions, so importing it raises ``NameError: torch``. Injecting ``torch`` into
    builtins before the import is the smallest possible fix and leaves the file untouched.
    """
    import builtins
    import torch
    builtins.torch = torch
    sys.path.insert(0, str(repo / "baselines"))
    import baseline_qwen_omni
    return baseline_qwen_omni


def import_official_scorer(repo: Path):
    sys.path.insert(0, str(repo / "scorer"))
    import backbone
    return backbone


# ----------------------------------------------------------------------------- rows
def load_rows(split: str, track: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(S.DATASET_REPO, f"{S.TASK}/{S.REGION}/{split}_{track}.jsonl",
                           repo_type="dataset", revision=S.DATASET_REVISION)
    return read_jsonl(Path(path))


def sample_ids(split: str, n: int | None, seed: int = S.SEED) -> list[str] | None:
    """A seeded sample of item ids. Ids are shared by all four tracks, so one sample serves all."""
    if n is None:
        return None
    ids = sorted(r["id"] for r in load_rows(split, "en"))
    return sorted(random.Random(f"{seed}-{split}").sample(ids, n))


def stage_rows(split: str, n: int | None) -> dict[str, list[dict]]:
    keep = sample_ids(split, n)
    out = {}
    for track in S.TRACKS:
        rows = load_rows(split, track)
        if keep is not None:
            wanted = set(keep)
            rows = [r for r in rows if r["id"] in wanted]
        out[track] = rows
    return out


# ----------------------------------------------------------------------------- images
def fetch_images(split: str, media_root: Path, needed: set[str]) -> None:
    """Download the split's image archive and unpack the images the run needs."""
    from huggingface_hub import hf_hub_download
    media_root = Path(media_root)
    todo = [p for p in sorted(needed) if not (media_root / p).is_file()]
    if not todo:
        return
    archive = hf_hub_download(S.DATASET_REPO, f"archives/images_{S.REGION}_{split}.zip",
                              repo_type="dataset", revision=S.DATASET_REVISION)
    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())
        for name in todo:
            if name in members:
                zf.extract(name, media_root)
    missing = [p for p in todo if not (media_root / p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing from images_{S.REGION}_{split}.zip, e.g. {missing[0]}")
    print(f"[data] {split}: {len(todo)} images unpacked under {media_root}", flush=True)
