"""Run the official MMCultureQA Task 2 baseline over every released split.

    python -m mmcqa_baseline.run --workdir results            # everything in settings.STAGES
    python -m mmcqa_baseline.run --workdir results --stages dev --limit 8   # smoke test

Steps: clone the starter kit at the pinned commit, fetch rows and images at the pinned
dataset revision, write the plan, start ``WORKERS_PER_GPU`` worker processes per GPU,
then merge their answers into official submission files and check those with the
organisers' format checker. Scoring and analysis are separate steps (score.py, analysis.py).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from . import data, settings as S


def gpu_names() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return [line.split(":", 1)[1].split("(")[0].strip() for line in out.splitlines() if line.startswith("GPU")]
    except (OSError, subprocess.CalledProcessError):
        return []


def environment() -> dict:
    import importlib.metadata as md
    versions = {}
    for pkg in ("torch", "transformers", "accelerate", "bitsandbytes", "qwen-omni-utils", "huggingface_hub",
                "bert_score", "sacrebleu", "rouge-score", "numpy"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "gpus": gpu_names(),
            "packages": versions}


def prepare(work: Path, stages: list, limit: int | None) -> list[dict]:
    repo = data.official_repo(work)
    plan, gold_dir = [], work / "gold"
    for name, split, n, mode in stages:
        rows = data.stage_rows(split, n)
        if limit:
            rows = {t: r[:limit] for t, r in rows.items()}
        data.fetch_images(split, work / "media", {r["image"] for rs in rows.values() for r in rs})
        for track, rs in rows.items():
            for r in rs:
                plan.append({"key": f"{name}|{track}|{r['id']}", "stage": name, "split": split, "track": track,
                             "id": r["id"], "image": r["image"], "question": r["question"], "mode": mode})
            if rs and "answer" in rs[0]:
                # gold in the shape the leaderboard script reads: {"id", "answer": {track: text}}
                data.write_jsonl(gold_dir / f"{name}_{track}.jsonl",
                                 ({"id": r["id"], "question": r["question"], "answer": {track: r["answer"]}}
                                  for r in rs))
            data.write_jsonl(gold_dir / f"{name}_{track}_ids.jsonl", ({"id": r["id"]} for r in rs))
        print(f"[run] stage {name}: {split}, {sum(map(len, rows.values()))} items ({mode})", flush=True)
    data.write_jsonl(work / "plan.jsonl", plan)

    from huggingface_hub import snapshot_download
    path = snapshot_download(S.MODEL_ID, revision=S.MODEL_REVISION)
    (work / "model_path.json").write_text(json.dumps({"model": S.MODEL_ID, "revision": S.MODEL_REVISION,
                                                     "path": path}))
    official_commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()
    (work / "logs").mkdir(exist_ok=True)
    (work / "logs" / "run_config.json").write_text(json.dumps({
        "official_repo": S.OFFICIAL_REPO_URL, "official_commit": official_commit,
        "dataset": S.DATASET_REPO, "dataset_revision": S.DATASET_REVISION,
        "model": S.MODEL_ID, "model_revision": S.MODEL_REVISION, "quantize_4bit_nf4": S.QUANTIZE,
        "max_new_tokens": S.MAX_NEW_TOKENS, "max_pixels": S.MAX_PIXELS, "do_sample": S.DO_SAMPLE,
        "system_prompt": S.SYSTEM_PROMPT, "instruction": S.INSTRUCTION,
        "stages": [dict(zip(("name", "split", "per_track", "mode"), s)) for s in stages],
        "limit_per_track": limit, "seed": S.SEED, "workers_per_gpu": S.WORKERS_PER_GPU,
        "environment": environment()}, indent=1, ensure_ascii=False))
    return plan


def launch(work: Path, budget_hours: float) -> float:
    gpus = gpu_names()
    if not gpus:
        raise SystemExit("No GPU found. On Kaggle: Settings -> Accelerator -> GPU T4 x2.")
    n_workers = len(gpus) * S.WORKERS_PER_GPU
    deadline = time.time() + budget_hours * 3600
    logs = work / "logs"
    for stale in (work / "raw").glob("ready_*"):
        stale.unlink()
    procs = []
    for k in range(n_workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % len(gpus)), TOKENIZERS_PARALLELISM="false")
        log = (logs / f"worker_{k}.log").open("a")
        procs.append(subprocess.Popen([sys.executable, "-m", "mmcqa_baseline.worker", "--workdir", str(work),
                                       "--worker", str(k), "--workers", str(n_workers),
                                       "--deadline", str(deadline)],
                                      stdout=log, stderr=subprocess.STDOUT, env=env,
                                      cwd=str(Path(__file__).resolve().parents[1])))
        # start the next worker only once this one has loaded, so peak CPU RAM stays at one model
        ready = work / "raw" / f"ready_{k}"
        while not ready.exists() and procs[-1].poll() is None:
            time.sleep(5)
        print(f"[run] worker {k} on GPU {k % len(gpus)} ({gpus[k % len(gpus)]}): "
              f"{'ready' if ready.exists() else 'EXITED, see logs/worker_%d.log' % k}", flush=True)
    t0 = time.time()
    total = sum(1 for _ in open(work / "plan.jsonl"))
    while any(p.poll() is None for p in procs):
        time.sleep(60)
        done = len(data.read_jsonl_many(work / "raw"))
        print(f"[run] {done}/{total} answered, {(time.time() - t0) / 60:.0f} min", flush=True)
    return time.time() - t0


def merge(work: Path) -> dict:
    """Merge worker files into one table and into submission-shaped files per stage and track."""
    plan = data.read_jsonl(work / "plan.jsonl")
    answers = {r["key"]: r for r in data.read_jsonl_many(work / "raw")}
    data.write_jsonl(work / "predictions_all.jsonl", (answers[j["key"]] | {"question": j["question"]}
                                                      for j in plan if j["key"] in answers))
    status, pred_dir = {}, work / "predictions"
    for stage in dict.fromkeys(j["stage"] for j in plan):
        for track in S.TRACKS:
            jobs = [j for j in plan if j["stage"] == stage and j["track"] == track]
            if not jobs:
                continue
            got = [j for j in jobs if j["key"] in answers]
            path = pred_dir / stage / f"prediction_{S.TASK}_{track}.jsonl"
            data.write_jsonl(path, ({"id": j["id"], "answer": answers[j["key"]]["answer"], "lang": track}
                                    for j in got))
            status[f"{stage}|{track}"] = {"planned": len(jobs), "answered": len(got),
                                          "errors": sum(1 for j in got if answers[j["key"]]["error"])}
    return status


def check_and_zip(work: Path, status: dict) -> dict:
    """Official format check on every complete prediction file; zip devtest files for upload."""
    checker = work / "mmcultureqa" / "format_checker" / "check_format.py"
    report = {}
    for key, st in status.items():
        stage, track = key.split("|")
        pred = work / "predictions" / stage / f"prediction_{S.TASK}_{track}.jsonl"
        if st["answered"] != st["planned"]:
            report[key] = f"incomplete ({st['answered']}/{st['planned']}), not checked"
            continue
        res = subprocess.run([sys.executable, str(checker), "--pred", str(pred), "--lang", track,
                              "--ids", str(work / "gold" / f"{stage}_{track}_ids.jsonl")],
                             capture_output=True, text=True)
        detail = " ".join((res.stdout + res.stderr).split())[-300:]
        report[key] = ("OK" if res.returncode == 0 else "FAILED") + " | " + detail
        if stage == "devtest" and res.returncode == 0:
            with zipfile.ZipFile(pred.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(pred, "prediction.jsonl")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, default=Path("results"))
    ap.add_argument("--stages", nargs="*", help=f"subset of {[s[0] for s in S.STAGES]} (default: all)")
    ap.add_argument("--limit", type=int, help="first N items per track and stage, for a smoke test")
    ap.add_argument("--budget-hours", type=float, default=S.TIME_BUDGET_HOURS)
    ap.add_argument("--skip-inference", action="store_true", help="only merge and check existing answers")
    args = ap.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    stages = [s for s in S.STAGES if not args.stages or s[0] in args.stages]

    if not args.skip_inference:
        t0 = time.time()
        plan = prepare(work, stages, args.limit)
        print(f"[run] plan: {len(plan)} generations, prepared in {time.time() - t0:.0f}s", flush=True)
        wall = launch(work, args.budget_hours)
        (work / "logs" / "wall_clock.json").write_text(json.dumps({"inference_wall_seconds": round(wall, 1),
                                                                  "gpus": gpu_names(),
                                                                  "workers": len(gpu_names()) * S.WORKERS_PER_GPU}))
    status = merge(work)
    checks = check_and_zip(work, status)
    (work / "logs" / "completion.json").write_text(json.dumps({"status": status, "format_check": checks},
                                                              indent=1, ensure_ascii=False))
    for key, st in status.items():
        print(f"  {key:22} {st['answered']:5}/{st['planned']:<5} errors={st['errors']}  {checks[key].split('|')[0]}")


if __name__ == "__main__":
    main()
