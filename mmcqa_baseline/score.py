"""Score every labelled stage with the leaderboard's own evaluation script.

    python -m mmcqa_baseline.score --workdir results

For each stage with references (dev, train_sample, dev_text_only) and each track it calls
``backbone.evaluate`` from the organisers' scorer, unmodified, at the pinned mmBERT revision,
and keeps per-item scores from the same scorer object for the analysis. The per-item mean is
asserted to equal the official F1.

Systems scored next to the baseline, on exactly the same items:
  official_baseline     Qwen2.5-Omni-3B as released (THE baseline)
  baseline_text_only    same model and prompt, image removed (ablation; dev_text_only stage)
  baseline_trimmed      baseline answer cut at the first invented dialogue turn (diagnostic only)
  echo_question         the question copied as the answer (a floor for this metric)
  tfidf_retrieval       answer of the most similar TRAIN question, char n-gram TF-IDF (no image,
                        no neural model: the "traditional approach" reference point)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from . import data, settings as S

# An answer the baseline keeps writing past its own end: "...\nHuman: That's a great answer..."
INVENTED_TURN = re.compile(r"\s*\b(?:Human|User|Assistant|Question|Answer)\s*:.*", re.S)


def trim_turns(text: str) -> str:
    cut = INVENTED_TURN.sub("", text).strip()
    return cut or text


class Retriever:
    """Nearest train question (cosine over char 2-4-gram TF-IDF) -> its reference answer."""

    def __init__(self, track: str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.rows = data.load_rows("train", track)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
        self.matrix = self.vec.fit_transform([r["question"] for r in self.rows])

    def answer(self, questions: list[str], exclude_ids: list[str] | None = None) -> list[str]:
        sims = (self.vec.transform(questions) @ self.matrix.T).toarray()
        out = []
        for k, row in enumerate(sims):
            order = row.argsort()[::-1]
            pick = next(j for j in order if not exclude_ids or self.rows[j]["id"] != exclude_ids[k])
            out.append(self.rows[pick]["answer"])
        return out


def pin_scorer_weights() -> None:
    """The leaderboard scores with a fixed mmBERT revision; bert_score takes a directory, so point it there."""
    if not os.environ.get("MMCQA_MODEL_DIR"):
        from huggingface_hub import snapshot_download
        os.environ["MMCQA_MODEL_DIR"] = snapshot_download(S.SCORER_MODEL, revision=S.SCORER_REVISION)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("results"))
    ap.add_argument("--no-mbert", action="store_true", help="skip the mBERT comparison scores")
    args = ap.parse_args()
    work = args.workdir.resolve()
    out_dir = work / "scores"
    (out_dir / "files").mkdir(parents=True, exist_ok=True)

    pin_scorer_weights()
    backbone = data.import_official_scorer(work / "mmcultureqa")
    import torch
    from bert_score import BERTScorer
    mbert = None if args.no_mbert else BERTScorer(
        model_type=S.COMPARISON_SCORER[0], num_layers=S.COMPARISON_SCORER[1], idf=False, lang=None,
        rescale_with_baseline=False, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=64)

    preds = defaultdict(dict)                  # (stage, track) -> id -> worker row
    for row in data.read_jsonl(work / "predictions_all.jsonl"):
        preds[(row["stage"], row["track"])][row["id"]] = row

    official, items, retrievers = {}, [], {}
    for gold_path in sorted((work / "gold").glob("*_*.jsonl")):
        if gold_path.stem.endswith("_ids"):
            continue
        stage, track = gold_path.stem.rsplit("_", 1)
        answered = preds.get((stage, track), {})
        gold = [g for g in data.read_jsonl(gold_path) if g["id"] in answered]
        if not gold:
            continue
        ids = [g["id"] for g in gold]
        questions = [g["question"] for g in gold]
        # Score only what was answered, so a run cut short by the time budget is still scored fairly;
        # coverage of the plan is reported separately in logs/completion.json.
        scored_gold = out_dir / "files" / f"gold_{stage}_{track}.jsonl"
        data.write_jsonl(scored_gold, ({"id": g["id"], "answer": g["answer"]} for g in gold))

        systems = {}
        model_answers = [answered[i]["answer"] for i in ids]
        if stage.endswith("text_only"):
            systems["baseline_text_only"] = model_answers
        else:
            systems["official_baseline"] = model_answers
            systems["baseline_trimmed"] = [trim_turns(a) for a in model_answers]
            systems["echo_question"] = questions
            if track not in retrievers:
                retrievers[track] = Retriever(track)
            systems["tfidf_retrieval"] = retrievers[track].answer(
                questions, exclude_ids=ids if stage.startswith("train") else None)

        refs = [backbone._clean(g["answer"][track]) for g in gold]
        for system, answers in systems.items():
            pred_path = out_dir / "files" / f"{system}_{stage}_{track}.jsonl"
            data.write_jsonl(pred_path, ({"id": i, "answer": a} for i, a in zip(ids, answers)))
            result = backbone.evaluate(str(scored_gold), str(pred_path), stage,
                                       submission_metadata=[{"name": "Language", "value": track}])
            block = result["submission_result"]
            official[f"{system}|{stage}|{track}"] = {**block, "n_items": result["submission_metadata"]["n_items"],
                                                     "n_empty": result["submission_metadata"]["n_empty"]}
            cands = [backbone._clean(a) for a in answers]
            nz = [k for k, c in enumerate(cands) if c]
            P, R, F, MF = ([0.0] * len(ids) for _ in range(4))
            if nz:
                p, r, f = backbone.get_scorer().score([cands[k] for k in nz], [refs[k] for k in nz],
                                                      verbose=False, batch_size=backbone.BATCH_SIZE)
                for j, k in enumerate(nz):
                    P[k], R[k], F[k] = float(p[j]), float(r[j]), float(f[j])
                if mbert:
                    _, _, mf = mbert.score([cands[k] for k in nz], [refs[k] for k in nz], verbose=False)
                    for j, k in enumerate(nz):
                        MF[k] = float(mf[j])
            mean_f = sum(F) / len(F)
            assert abs(mean_f - block["BERTScore-F1"]) < 1e-4, (system, stage, track, mean_f, block)
            rouge = backbone.get_rouge()
            for k, i in enumerate(ids):
                row = answered[i]
                items.append({"system": system, "stage": stage, "track": track, "id": i,
                              "question": questions[k], "reference": refs[k], "prediction": cands[k],
                              "mm_p": P[k], "mm_r": R[k], "mm_f1": F[k], "mbert_f1": MF[k] if mbert else None,
                              "rougeL": rouge.score(refs[k], cands[k])["rougeL"].fmeasure if cands[k] else 0.0,
                              **({k2: row.get(k2) for k2 in ("latency_seconds", "new_tokens", "prompt_tokens",
                                                              "hit_max_new_tokens", "error")}
                                 if system in ("official_baseline", "baseline_text_only") else {})})
            print(f"  {system:20} {stage:14} {track:4} n={len(ids):5} F1={block['BERTScore-F1']:.4f} "
                  f"P={block['BERTScore-P']:.4f} R={block['BERTScore-R']:.4f} BLEU={block['BLEU']:.2f} "
                  f"ROUGE-L={block['ROUGE-L']:.4f}", flush=True)

    (out_dir / "official_metrics.json").write_text(json.dumps({
        "scorer": {"model": backbone.HUB_MODEL, "revision": backbone.HUB_REVISION, "layer": backbone.NUM_LAYERS,
                   "weights_dir": os.environ.get("MMCQA_MODEL_DIR")},
        "comparison_scorer": None if args.no_mbert else {"model": S.COMPARISON_SCORER[0],
                                                         "layer": S.COMPARISON_SCORER[1]},
        "results": official}, indent=1, ensure_ascii=False))
    data.write_jsonl(out_dir / "per_item_scores.jsonl", items)
    print(f"[score] wrote {out_dir}")


if __name__ == "__main__":
    main()
