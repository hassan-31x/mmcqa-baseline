"""One inference worker: loads the official baseline model on one GPU and answers its share of the plan.

Started by ``run.py``; one process per (GPU, slot). Every answer is appended and flushed
straight away, and items already answered by any worker are skipped, so a restarted run
continues where it stopped.

    CUDA_VISIBLE_DEVICES=0 python -m mmcqa_baseline.worker --workdir results --worker 0 --workers 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import data, settings as S


def done_keys(raw_dir: Path) -> set[str]:
    keys = set()
    for path in raw_dir.glob("worker_*.jsonl"):
        keys.update(r["key"] for r in data.read_jsonl(path))
    return keys


def text_only_answer(B, model, processor, question: str, track: str, max_new_tokens: int) -> str:
    """Ablation: the official ``answer`` call with the image removed, everything else identical."""
    import torch
    content = [{"type": "text", "text": question},
               {"type": "text", "text": B.INSTRUCTION.format(language=B.LANGUAGE.get(track, "the language of the question"))}]
    conversation = [{"role": "system", "content": [{"type": "text", "text": B.SYSTEM}]},
                    {"role": "user", "content": content}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    out = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--deadline", type=float, default=0, help="unix time after which no new item is started")
    args = ap.parse_args()

    import torch
    work = args.workdir
    raw_dir = work / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    plan = data.read_jsonl(work / "plan.jsonl")
    mine = [job for k, job in enumerate(plan) if k % args.workers == args.worker]
    skip = done_keys(raw_dir)
    todo = [job for job in mine if job["key"] not in skip]
    print(f"[worker {args.worker}] {len(mine)} jobs, {len(todo)} to do", flush=True)

    B = data.import_official_baseline(work / "mmcultureqa")
    meta = json.loads((work / "model_path.json").read_text())
    t_load = time.time()
    model, processor = B.build_model(meta["path"], S.QUANTIZE, S.MAX_PIXELS)
    load_seconds = time.time() - t_load

    # Count generated tokens without touching the official answer(): wrap generate.
    stats = {}
    original_generate = model.generate

    def generate(*a, **kw):
        out = original_generate(*a, **kw)
        stats["new_tokens"] = int(out.shape[1] - kw["input_ids"].shape[1])
        stats["prompt_tokens"] = int(kw["input_ids"].shape[1])
        return out

    model.generate = generate
    (raw_dir / f"ready_{args.worker}").write_text(json.dumps({"load_seconds": load_seconds}))
    media = work / "media"

    out_path = raw_dir / f"worker_{args.worker}.jsonl"
    with out_path.open("a", encoding="utf-8") as fh:
        if out_path.stat().st_size and not out_path.read_bytes().endswith(b"\n"):
            fh.write("\n")                     # a previous run was killed mid-line
        started = time.time()
        for n, job in enumerate(todo, 1):
            if args.deadline and time.time() > args.deadline:
                print(f"[worker {args.worker}] time budget reached after {n - 1} items", flush=True)
                break
            stats.clear()
            torch.cuda.synchronize()
            t0 = time.time()
            error = None
            try:
                if job["mode"] == "official":
                    text = B.answer(model, processor, media / job["image"], None, job["question"],
                                    job["track"], S.MAX_NEW_TOKENS)
                else:
                    text = text_only_answer(B, model, processor, job["question"], job["track"], S.MAX_NEW_TOKENS)
            except Exception as exc:          # same policy as the official loop: a failed item scores empty
                text, error = "", f"{type(exc).__name__}: {str(exc)[:300]}"
                torch.cuda.empty_cache()
            latency = time.time() - t0
            fh.write(json.dumps({
                "key": job["key"], "stage": job["stage"], "split": job["split"], "track": job["track"],
                "id": job["id"], "mode": job["mode"], "answer": text, "error": error,
                "latency_seconds": round(latency, 3), "new_tokens": stats.get("new_tokens"),
                "prompt_tokens": stats.get("prompt_tokens"),
                "hit_max_new_tokens": stats.get("new_tokens") == S.MAX_NEW_TOKENS,
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "worker": args.worker, "finished_unix": round(time.time(), 1)}, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - started)
                print(f"[worker {args.worker}] {n}/{len(todo)} {rate:.3f} items/s "
                      f"eta {(len(todo) - n) / rate / 60:.0f} min | {job['stage']} {job['track']}", flush=True)
    print(f"[worker {args.worker}] finished", flush=True)


if __name__ == "__main__":
    main()
