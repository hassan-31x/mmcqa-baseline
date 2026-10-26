"""Extra analysis for the report: dataset stats, train overlap, output anatomy, post-processing
probes, metric calibration, dialect ID, vision ablation by question type, statistical power.

    python -m mmcqa_baseline.insights --workdir results

Run after score and analysis. No GPU needed. Re-scoring goes through the official scorer and is
cached in scores/probe_scores.jsonl. Writes report/tables/11-21 and report/charts/12-21.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, settings as S
from .analysis import (COLORS, GRID, INK, INK2, SHORT, SURFACE, TNAME, TR, TRACK_COLORS, boot_ci,
                       qtype_of, save, script_ok, setup_mpl, topic_of, words, write_table)
from .score import INVENTED_TURN, pin_scorer_weights

DID_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus6"
DID_REVISION = "1f6d9643228adb8463a3dee24b369ec41a3235de"
DID_TARGET = {"msa": "MSA", "arz": "CAI", "ajp": "BEI"}       # MADAR-6 label each track should get
DID_NAMES = {"MSA": "MSA", "CAI": "Cairo (Egyptian)", "BEI": "Beirut (Levantine)", "DOH": "Doha (Gulf)",
             "RAB": "Rabat (Maghrebi)", "TUN": "Tunis (Maghrebi)"}
TRUNC_WORDS = (10, 15, 20, 25, 30, 40)
SENT_END = re.compile(r"(?<=[.!?؟])\s+")
FILLER = re.compile(r"^(?:(?:well|oh|so|um|uh|you know|okay|ok|hmm|ah)\b[,!.]?\s*)+", re.I)
QUESTION_BACK = re.compile(r"\?\s*$|؟\s*$")
SEED = S.SEED


# ----------------------------------------------------------------------------- answer surgery
def first_turn(text: str) -> str:
    """The answer before the model starts inventing the next dialogue turn."""
    return INVENTED_TURN.sub("", text).strip()


def sentences(text: str) -> list[str]:
    return [s for s in SENT_END.split(text.strip()) if s.strip()]


def clean(text: str) -> str:
    """First turn, minus spoken-style openers ("Well, you know,") and trailing questions to the user."""
    parts = sentences(first_turn(text))
    while len(parts) > 1 and QUESTION_BACK.search(parts[-1]):
        parts.pop()
    out = FILLER.sub("", " ".join(parts)).strip()
    return out[:1].upper() + out[1:] if out else first_turn(text) or text


def first_words(text: str, k: int) -> str:
    return " ".join(text.split()[:k])


# ----------------------------------------------------------------------------- cached official re-scoring
class Rescorer:
    """Official BERTScorer (mmBERT L17) with a disk cache keyed by (candidate, reference)."""

    def __init__(self, work: Path):
        pin_scorer_weights()
        self.backbone = data.import_official_scorer(data.official_repo(work))
        self.path = work / "scores" / "probe_scores.jsonl"
        self.cache = {r["k"]: r["f"] for r in data.read_jsonl(self.path)}

    @staticmethod
    def key(c: str, r: str) -> str:
        return hashlib.sha1(f"{c}\x00{r}".encode()).hexdigest()

    def f1(self, cands: list[str], refs: list[str]) -> np.ndarray:
        cands = [self.backbone._clean(c) for c in cands]
        refs = [self.backbone._clean(r) for r in refs]
        keys = [self.key(c, r) for c, r in zip(cands, refs)]
        todo = sorted({k: (c, r) for k, c, r in zip(keys, cands, refs) if k not in self.cache and c}.items())
        if todo:
            print(f"[insights] scoring {len(todo)} new pairs", flush=True)
            _, _, f = self.backbone.get_scorer().score([c for _, (c, _) in todo], [r for _, (_, r) in todo],
                                                       verbose=False, batch_size=self.backbone.BATCH_SIZE)
            new = {k: float(v) for (k, _), v in zip(todo, f)}
            self.cache.update(new)
            with self.path.open("a", encoding="utf-8") as fh:
                for k, v in new.items():
                    fh.write(json.dumps({"k": k, "f": v}) + "\n")
        return np.array([self.cache.get(k, 0.0) if c else 0.0 for k, c in zip(keys, cands)])


# ----------------------------------------------------------------------------- dialect identification
def dialect_labels(texts: list[str]) -> list[str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DID_MODEL, revision=DID_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(DID_MODEL, revision=DID_REVISION).eval()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            out += [model.config.id2label[int(k)] for k in model(**batch).logits.argmax(-1).cpu()]
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("results"))
    ap.add_argument("--no-did", action="store_true", help="skip the dialect-identification model")
    args = ap.parse_args()
    work = args.workdir.resolve()
    tables, charts = work / "report" / "tables", work / "report" / "charts"
    plt = setup_mpl()
    rng = np.random.default_rng(SEED)
    notes: dict = {}

    items = pd.DataFrame(data.read_jsonl(work / "scores" / "per_item_scores.jsonl"))
    raw = pd.DataFrame(data.read_jsonl(work / "predictions_all.jsonl"))
    raw_by_key = raw.set_index(["stage", "track", "id"])
    dev_items = items[items.stage == "dev"]
    base = dev_items[dev_items.system == "official_baseline"].copy()
    base["raw"] = [raw_by_key.loc[("dev", t, i)].answer for t, i in zip(base.track, base.id)]
    base["prompt_tokens"] = [raw_by_key.loc[("dev", t, i)].prompt_tokens for t, i in zip(base.track, base.id)]

    rows = {(split, t): data.load_rows(split, t) for split in ("train", "dev", "devtest") for t in TR}
    en_q = {split: {r["id"]: r["question"] for r in rows[(split, "en")]} for split in ("train", "dev", "devtest")}
    en_ref = {r["id"]: r["answer"] for r in rows[("dev", "en")]}
    topic = {i: topic_of(q + " " + en_ref.get(i, "")) for i, q in en_q["dev"].items()}
    intent = {i: qtype_of(q) for i, q in en_q["dev"].items()}

    # ------------------------------------------------------------------ 11 dataset statistics
    stats = []
    for split in ("train", "dev", "devtest"):
        for t in TR:
            rr = rows[(split, t)]
            qw = np.array([len(words(r["question"])) for r in rr])
            aw = np.array([len(words(r["answer"])) for r in rr if r.get("answer")])
            vocab = {w.lower() for r in rr if r.get("answer") for w in words(r["answer"])}
            stats.append({"split": split, "track": TNAME[t], "items": len(rr),
                          "unique images": len({r["image"] for r in rr}),
                          "question words mean": qw.mean(), "question words median": float(np.median(qw)),
                          "answer words mean": aw.mean() if len(aw) else np.nan,
                          "answer words median": float(np.median(aw)) if len(aw) else np.nan,
                          "answer words p95": float(np.percentile(aw, 95)) if len(aw) else np.nan,
                          "answer vocabulary": len(vocab) if vocab else np.nan,
                          "has answers": bool(len(aw))})
    stats_tab = pd.DataFrame(stats)
    write_table(stats_tab, tables / "11_dataset_statistics", "{:.1f}")
    ids = {split: set(en_q[split]) for split in en_q}
    notes["id_overlap"] = {"train&dev": len(ids["train"] & ids["dev"]), "train&devtest": len(ids["train"] & ids["devtest"]),
                           "dev&devtest": len(ids["dev"] & ids["devtest"])}
    notes["ids_shared_across_tracks"] = all({r["id"] for r in rows[(s, t)]} == ids[s] for s in ids for t in TR)

    # ------------------------------------------------------------------ 12 topic mix across splits
    mix = []
    for split in ("train", "dev", "devtest"):
        counts = pd.Series([topic_of(q) for q in en_q[split].values()]).value_counts(normalize=True) * 100
        for name, v in counts.items():
            mix.append({"split": split, "topic": name, "share %": v})
    mix_tab = pd.DataFrame(mix).pivot(index="topic", columns="split", values="share %").fillna(0)
    mix_tab = mix_tab[["train", "dev", "devtest"]].sort_values("train", ascending=False).reset_index()
    write_table(mix_tab, tables / "12_topic_mix_by_split", "{:.1f}")

    # ------------------------------------------------------------------ 13 train-dev overlap (nearest train question)
    from sklearn.feature_extraction.text import TfidfVectorizer
    train_ids = list(en_q["train"])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
    tm = vec.fit_transform([en_q["train"][i] for i in train_ids])
    nn = {}
    for split in ("dev", "devtest"):
        sp_ids = list(en_q[split])
        sims = (vec.transform([en_q[split][i] for i in sp_ids]) @ tm.T).toarray()
        best = sims.argmax(1)
        for i, b, s in zip(sp_ids, best, sims[np.arange(len(sp_ids)), best]):
            nn[(split, i)] = (float(s), train_ids[b])
    overlap = []
    for split in ("dev", "devtest"):
        s = np.array([v[0] for k, v in nn.items() if k[0] == split])
        overlap.append({"split": split, "items": len(s), "median similarity": float(np.median(s)),
                        "share >= 0.9 %": 100 * (s >= 0.9).mean(), "share >= 0.8 %": 100 * (s >= 0.8).mean(),
                        "share >= 0.6 %": 100 * (s >= 0.6).mean(),
                        "exact duplicate question %": 100 * np.mean([en_q[split][i].strip().lower() in
                                                                     {q.strip().lower() for q in en_q["train"].values()}
                                                                     for i in en_q[split]])})
    write_table(pd.DataFrame(overlap), tables / "13_train_overlap", "{:.2f}")
    sim_bins = [0, 0.5, 0.6, 0.7, 0.8, 1.01]
    sim_labels = ["< 0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", ">= 0.8"]
    dev_items = dev_items.assign(nn_sim=[nn[("dev", i)][0] for i in dev_items.id])
    dev_items["sim_bin"] = pd.cut(dev_items.nn_sim, sim_bins, labels=sim_labels, right=False).astype(str)
    by_sim = (dev_items[dev_items.system.isin(["official_baseline", "tfidf_retrieval", "echo_question"])]
              .groupby(["sim_bin", "system"]).mm_f1.mean().unstack().reindex(sim_labels))
    by_sim["n items (per track)"] = dev_items[dev_items.system == "official_baseline"].groupby("sim_bin").size() // len(TR)
    by_sim = by_sim.rename(columns=SHORT).reset_index().rename(columns={"sim_bin": "nearest-train similarity"})
    write_table(by_sim, tables / "13b_f1_by_train_similarity")

    # ------------------------------------------------------------------ 14 answer anatomy
    anat = []
    for t in TR:
        b = base[base.track == t]
        first = b.raw.map(first_turn)
        has_turn = b.raw.map(lambda x: bool(INVENTED_TURN.search(x)))
        complete = first.map(lambda x: bool(re.search(r"[.!?؟。]\s*$", x)))
        w_all, w_first = b.raw.map(lambda x: len(x.split())), first.map(lambda x: len(x.split()))
        anat.append({"track": TNAME[t], "n": len(b),
                     "invented next turn %": 100 * has_turn.mean(),
                     "cut mid-sentence by 64-token cap %": 100 * ((~has_turn) & (~complete)).mean(),
                     "spoken-style opener %": 100 * first.map(lambda x: bool(FILLER.match(x))).mean(),
                     "asks the user a question %": 100 * first.map(lambda x: any(QUESTION_BACK.search(s)
                                                                                for s in sentences(x)[1:])).mean(),
                     "words, whole output": w_all.mean(), "words, first turn": w_first.mean(),
                     "words, cleaned answer": b.raw.map(lambda x: len(clean(x).split())).mean(),
                     "words, reference": b.reference.map(lambda x: len(x.split())).mean(),
                     "output spent on invented turns %": 100 * (1 - w_first.sum() / w_all.sum()),
                     "sentences in first turn": first.map(lambda x: len(sentences(x))).mean()})
    anat_tab = pd.DataFrame(anat)
    write_table(anat_tab, tables / "14_answer_anatomy", "{:.1f}")

    # ------------------------------------------------------------------ 15 post-processing probes (official scorer)
    R = Rescorer(work)
    variants = {"as submitted": lambda x, ref: x,
                "first turn only": lambda x, ref: first_turn(x) or x,
                "cleaned (no openers, no questions back)": lambda x, ref: clean(x),
                "first sentence": lambda x, ref: (sentences(clean(x)) or [clean(x)])[0],
                "first two sentences": lambda x, ref: " ".join(sentences(clean(x))[:2]),
                **{f"first {k} words": (lambda k: lambda x, ref: first_words(clean(x), k))(k) for k in TRUNC_WORDS},
                "oracle length (cut to reference length)": lambda x, ref: first_words(clean(x), len(ref.split()))}
    probe_rows, probe_items = [], {}
    for t in TR:
        b = base[base.track == t]
        for name, fn in variants.items():
            cands = [fn(x, r) for x, r in zip(b.raw, b.reference)]
            f = R.f1(cands, list(b.reference))
            if name == "as submitted":
                assert abs(f.mean() - b.mm_f1.mean()) < 1e-4, (t, f.mean(), b.mm_f1.mean())
            probe_items[(t, name)] = f
            lo, hi = boot_ci(f)
            probe_rows.append({"variant": name, "track": TNAME[t], "F1": f.mean(), "CI low": lo, "CI high": hi,
                               "words": np.mean([len(c.split()) for c in cands]),
                               "delta vs submitted": f.mean() - b.mm_f1.mean()})
    probe = pd.DataFrame(probe_rows)
    probe_wide = probe.pivot(index="variant", columns="track", values="F1")[[TNAME[t] for t in TR]]
    probe_wide["Macro avg"] = probe_wide.mean(1)
    probe_wide["mean words"] = probe.groupby("variant").words.mean()
    probe_wide = probe_wide.reindex(list(variants)).reset_index()
    write_table(probe_wide, tables / "15_postprocessing_probes")
    write_table(probe, tables / "15b_postprocessing_probes_long")

    # ------------------------------------------------------------------ 16 metric calibration
    gold = {t: {r["id"]: r["answer"] for r in rows[("dev", t)]} for t in TR}
    dev_ids = sorted(gold["en"])
    cal, cross = [], np.zeros((len(TR), len(TR)))
    perm = rng.permutation(len(dev_ids))
    perm = np.roll(perm, 1)  # every item gets a different item's reference
    for j, t in enumerate(TR):
        refs = [gold[t][i] for i in dev_ids]
        floor = R.f1([refs[p] for p in perm], refs)
        cal.append({"track": TNAME[t], "shuffled reference (unrelated answer, same variety)": floor.mean()})
        for k, src in enumerate(TR):
            cross[k, j] = 1.0 if src == t else R.f1([gold[src][i] for i in dev_ids], refs).mean()
    cal_tab = pd.DataFrame(cal)
    sys_f1 = dev_items.groupby(["system", "track"]).mm_f1.mean()
    for s in ("echo_question", "official_baseline", "tfidf_retrieval", "baseline_trimmed"):
        cal_tab[SHORT[s]] = [sys_f1[(s, t)] for t in TR]
    cal_tab["Baseline, cleaned"] = [probe_items[(t, "cleaned (no openers, no questions back)")].mean() for t in TR]
    for t_idx, t in enumerate(TR):
        others = [cross[k, t_idx] for k in range(len(TR)) if k != t_idx]
        cal_tab.loc[t_idx, "best other-variety reference"] = max(others)
    floor_col = "shuffled reference (unrelated answer, same variety)"
    for s in ("Echo question", "Baseline", "TF-IDF retrieval", "Trimmed", "Baseline, cleaned"):
        cal_tab[f"normalised {s}"] = (cal_tab[s] - cal_tab[floor_col]) / (1 - cal_tab[floor_col])
    write_table(cal_tab, tables / "16_metric_calibration")
    cross_tab = pd.DataFrame(cross, index=[f"{TNAME[s]} reference" for s in TR], columns=[f"scored as {TNAME[t]}" for t in TR])
    write_table(cross_tab.reset_index().rename(columns={"index": "prediction"}), tables / "16b_cross_variety_references")

    # ------------------------------------------------------------------ 17 dialect identification
    did_tab = None
    if not args.no_did:
        texts, keys = [], []
        for t in ("msa", "arz", "ajp"):
            b = base[base.track == t]
            for i, ref, x in zip(b.id, b.reference, b.raw):
                texts += [ref, clean(x)]
                keys += [(t, i, "reference"), (t, i, "baseline")]
            to = raw[(raw.stage == "dev_text_only") & (raw.track == t)]
            for i, x in zip(to.id, to.answer):
                texts.append(clean(x)); keys.append((t, i, "baseline, no image"))
            for i, q in ((r["id"], r["question"]) for r in rows[("dev", t)]):
                texts.append(q); keys.append((t, i, "question"))
        labels = dialect_labels(texts)
        did = pd.DataFrame([{"track": k[0], "id": k[1], "source": k[2], "label": l} for k, l in zip(keys, labels)])
        did.to_csv(work / "scores" / "dialect_labels.csv", index=False)
        share = (did.groupby(["track", "source"]).label.value_counts(normalize=True).mul(100)
                 .unstack(fill_value=0).reindex(columns=list(DID_NAMES), fill_value=0))
        share["target label"] = [DID_TARGET[t] for t, _ in share.index]
        share["in target variety %"] = [share.loc[(t, s), DID_TARGET[t]] for t, s in share.index]
        did_tab = share.reset_index()
        did_tab["track"] = did_tab.track.map(TNAME)
        write_table(did_tab, tables / "17_dialect_identification", "{:.1f}")
        # does writing in the right variety pay off under the metric?
        bl = did[did.source == "baseline"].merge(base[["track", "id", "mm_f1"]], on=["track", "id"])
        bl["on target"] = bl.label == bl.track.map(DID_TARGET)
        pay = bl.groupby(["track", "on target"]).mm_f1.agg(["mean", "count"]).unstack()
        pay.columns = [f"{a} ({'on' if b else 'off'} target)" for a, b in pay.columns]
        pay = pay.reset_index()
        pay["track"] = pay.track.map(TNAME)
        write_table(pay, tables / "17b_f1_by_dialect_match")

    # ------------------------------------------------------------------ 18 what predicts an item's score
    pred = []
    for t in TR:
        b = base[base.track == t].copy()
        b["ref words"] = b.reference.map(lambda x: len(x.split()))
        b["question words"] = b.question.map(lambda x: len(x.split()))
        b["first-turn words"] = b.raw.map(lambda x: len(first_turn(x).split()))
        b["nearest-train similarity"] = [nn[("dev", i)][0] for i in b.id]
        tf = dev_items[(dev_items.system == "tfidf_retrieval") & (dev_items.track == t)].set_index("id").mm_f1
        b["TF-IDF retrieval F1"] = b.id.map(tf)
        b["echo-question F1"] = b.id.map(dev_items[(dev_items.system == "echo_question") & (dev_items.track == t)]
                                         .set_index("id").mm_f1)
        b["prompt tokens (image + text)"] = b.prompt_tokens
        row = {"track": TNAME[t]}
        for c in ("ref words", "question words", "first-turn words", "prompt tokens (image + text)",
                  "nearest-train similarity", "TF-IDF retrieval F1", "echo-question F1"):
            row[c] = b.mm_f1.corr(b[c], method="spearman")
        pred.append(row)
    write_table(pd.DataFrame(pred), tables / "18_item_score_correlates", "{:.3f}")
    rl_bins = [0, 15, 20, 25, 30, 40, 1000]
    rl_labels = ["<15", "15-19", "20-24", "25-29", "30-39", "40+"]
    base["ref_bin"] = pd.cut(base.reference.map(lambda x: len(x.split())), rl_bins, labels=rl_labels, right=False).astype(str)
    by_ref = base.groupby(["ref_bin", "track"]).mm_f1.agg(["mean", "count"]).reset_index()
    by_ref_tab = by_ref.pivot(index="ref_bin", columns="track", values="mean").reindex(rl_labels)[list(TR)].rename(columns=TNAME)
    by_ref_n = by_ref.pivot(index="ref_bin", columns="track", values="count").reindex(rl_labels)[list(TR)]
    out_ref = by_ref_tab.copy()
    for t in TR:
        out_ref[f"n {TNAME[t]}"] = by_ref_n[t]
    write_table(out_ref.reset_index().rename(columns={"ref_bin": "reference words"}), tables / "18b_f1_by_reference_length")

    # ------------------------------------------------------------------ 19 vision ablation by intent and topic
    abl = []
    to = items[items.system == "baseline_text_only"].set_index(["track", "id"]).mm_f1
    wi = base.set_index(["track", "id"]).mm_f1
    common = to.index.intersection(wi.index)
    diff = pd.DataFrame({"with": wi[common], "without": to[common]}).reset_index()
    diff["drop"] = diff["with"] - diff["without"]
    for col, mapping in (("question intent", intent), ("topic", topic)):
        diff[col] = diff.id.map(mapping)
        g = diff.groupby(col).agg(n=("drop", "size"), with_image=("with", "mean"), without_image=("without", "mean"),
                                  drop=("drop", "mean"), image_helps=("drop", lambda x: 100 * (x > 0).mean()))
        g = g.sort_values("drop", ascending=False).reset_index().rename(columns={col: "group"})
        g.insert(0, "by", col)
        abl.append(g)
    abl_tab = pd.concat(abl).rename(columns={"n": "items (all tracks)", "with_image": "F1 with image",
                                             "without_image": "F1 no image", "drop": "F1 drop",
                                             "image_helps": "image helps %"})
    write_table(abl_tab, tables / "19_vision_ablation_by_group")

    # ------------------------------------------------------------------ 20 statistical power
    pw = []
    for t in TR:
        b = base[base.track == t].set_index("id").mm_f1
        tr = dev_items[(dev_items.system == "baseline_trimmed") & (dev_items.track == t)].set_index("id").mm_f1[b.index]
        sd, sd_d = b.std(), (b - tr).std()
        pw.append({"track": TNAME[t], "n": len(b), "per-item F1 sd": sd, "standard error": sd / np.sqrt(len(b)),
                   "95% CI half-width": 1.96 * sd / np.sqrt(len(b)), "paired-difference sd": sd_d,
                   "min detectable paired difference (80% power)": 2.8 * sd_d / np.sqrt(len(b)),
                   "items needed for a 0.005 paired difference": int(np.ceil((2.8 * sd_d / 0.005) ** 2))})
    write_table(pd.DataFrame(pw), tables / "20_statistical_power")

    # ------------------------------------------------------------------ 21 devtest output profile vs dev
    prof = []
    for stage in ("dev", "devtest"):
        for t in TR:
            r = raw[(raw.stage == stage) & (raw.track == t)]
            prof.append({"stage": stage, "track": TNAME[t], "n": len(r),
                         "words, whole output": r.answer.map(lambda x: len(x.split())).mean(),
                         "words, first turn": r.answer.map(lambda x: len(first_turn(x).split())).mean(),
                         "invented next turn %": 100 * r.answer.map(lambda x: bool(INVENTED_TURN.search(x))).mean(),
                         "hit 64-token cap %": 100 * r.hit_max_new_tokens.eq(True).mean(),
                         "wrong script %": 100 * (~r.answer.map(lambda x: script_ok(x, t))).mean(),
                         "nearest-train similarity (median)": float(np.median([nn[(stage, i)][0] for i in r.id]))})
    write_table(pd.DataFrame(prof), tables / "21_devtest_vs_dev_outputs", "{:.2f}")

    # ================================================================== charts
    import matplotlib.colors as mcolors
    seq = mcolors.LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#6fa6e8", "#0d366b"])

    # C12 dataset: answer length per track and split (reference side)
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    pos, vals, cols, ticks = [], [], [], []
    for k, t in enumerate(TR):
        for j, split in enumerate(("train", "dev")):
            vals.append([len(words(r["answer"])) for r in rows[(split, t)]])
            pos.append(k * 3 + j)
            cols.append(TRACK_COLORS[t] if split == "dev" else "#c3c2b7")
        ticks.append(k * 3 + 0.5)
    bp = ax.boxplot(vals, positions=pos, widths=0.8, patch_artist=True, showfliers=False, medianprops={"color": INK})
    for p, c in zip(bp["boxes"], cols):
        p.set_facecolor(c); p.set_edgecolor(SURFACE)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c3c2b7", label="train"), Patch(color=INK2, label="dev (track colour)")],
              fontsize=7, loc="upper right")
    ax.set_xticks(ticks, [TNAME[t] for t in TR]); ax.set_ylabel("Words per reference answer")
    ax.set_title("Reference answer length by variety (train vs dev)")
    save(fig, charts / "12_dataset_answer_length.png")

    # C13 topic mix across splits
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    x = np.arange(len(mix_tab))
    for j, (split, c) in enumerate((("train", "#c3c2b7"), ("dev", COLORS["official_baseline"]), ("devtest", COLORS["tfidf_retrieval"]))):
        ax.bar(x + (j - 1) * 0.27, mix_tab[split], 0.26, color=c, label=split)
    ax.set_xticks(x, mix_tab.topic, rotation=20, ha="right", fontsize=8); ax.set_ylabel("% of questions")
    ax.set_title("Topic mix by split (keyword heuristic on English questions)"); ax.legend(fontsize=7)
    save(fig, charts / "13_topic_mix_by_split.png")

    # C14 train similarity: histogram + F1 by similarity bin
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    s_dev = [v[0] for k, v in nn.items() if k[0] == "dev"]
    axes[0].hist(s_dev, bins=30, color=COLORS["official_baseline"], edgecolor=SURFACE, linewidth=0.4)
    axes[0].set_xlabel("cosine to nearest train question"); axes[0].set_ylabel("dev items")
    axes[0].set_title("How close is dev to train?", fontsize=9)
    for s in ("Baseline", "TF-IDF retrieval", "Echo question"):
        key = {v: k for k, v in SHORT.items()}[s]
        axes[1].plot(by_sim["nearest-train similarity"], by_sim[s], marker="o", lw=2, ms=4, color=COLORS[key], label=s)
    axes[1].set_xlabel("nearest-train similarity"); axes[1].set_ylabel("F1 (all tracks)")
    axes[1].tick_params(axis="x", labelsize=7); axes[1].legend(fontsize=7)
    axes[1].set_title("Retrieval wins when train has a near twin", fontsize=9)
    save(fig, charts / "14_train_similarity.png")

    # C15 anatomy: where the words of each output go (stacked)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    y = np.arange(len(TR))
    cleaned = anat_tab["words, cleaned answer"].to_numpy()
    chat = anat_tab["words, first turn"].to_numpy() - cleaned
    runaway = anat_tab["words, whole output"].to_numpy() - anat_tab["words, first turn"].to_numpy()
    ax.barh(y, cleaned, color=COLORS["official_baseline"], label="the answer")
    ax.barh(y, chat, left=cleaned, color=COLORS["baseline_trimmed"], label="chit-chat (openers, questions back)")
    ax.barh(y, runaway, left=cleaned + chat, color=COLORS["tfidf_retrieval"], label="invented next turns")
    ax.scatter(anat_tab["words, reference"], y, marker="|", s=250, color=INK, zorder=4, label="reference length")
    ax.set_yticks(y, anat_tab.track); ax.invert_yaxis(); ax.set_xlabel("words (mean over dev)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_title("Where the 64 generated tokens go")
    save(fig, charts / "15_answer_anatomy.png")

    # C16 post-processing: F1 vs words kept
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for t in TR:
        ys = [probe_items[(t, f"first {k} words")].mean() for k in TRUNC_WORDS]
        ax.plot(TRUNC_WORDS, ys, marker="o", ms=4, lw=2, color=TRACK_COLORS[t], label=TNAME[t])
        ax.axhline(base[base.track == t].mm_f1.mean(), color=TRACK_COLORS[t], lw=1, ls=":")
        ref_med = np.median(base[base.track == t].reference.map(lambda x: len(x.split())))
        ax.axvline(ref_med, color=TRACK_COLORS[t], lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("first k words of the cleaned answer kept"); ax.set_ylabel("BERTScore-F1 (official)")
    ax.set_title("Removing the junk helps; cutting the answer shorter does not\n"
                 "(dotted: as submitted; dashed: median reference length)", fontsize=9)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, charts / "16_postprocessing_length_sweep.png")

    # C17 metric ruler: where each system sits between floor and ceiling
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    marks = [(floor_col, "Shuffled reference", "#c3c2b7", "o"), ("Echo question", "Echo question", COLORS["echo_question"], "s"),
             ("Baseline", "Baseline", COLORS["official_baseline"], "D"),
             ("Baseline, cleaned", "Baseline, cleaned", COLORS["baseline_trimmed"], "^"),
             ("TF-IDF retrieval", "TF-IDF retrieval", COLORS["tfidf_retrieval"], "v"),
             ("best other-variety reference", "Reference in another variety", INK, "*")]
    for k, t in enumerate(TR):
        ax.plot([cal_tab.loc[k, floor_col], 1.0], [k, k], color=GRID, lw=6, solid_capstyle="round", zorder=1)
        for col, lab, c, m in marks:
            ax.scatter(cal_tab.loc[k, col], k, color=c, marker=m, s=45, zorder=3, edgecolor=SURFACE, linewidth=0.8,
                       label=lab if k == 0 else None)
    ax.set_yticks(range(len(TR)), [TNAME[t] for t in TR]); ax.invert_yaxis()
    ax.set_xlabel("BERTScore-F1 (official). Bar = unrelated answer ... identical answer")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    ax.set_title("The metric's usable range is narrow")
    save(fig, charts / "17_metric_ruler.png")

    # C18 cross-variety reference matrix
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.imshow(cross, cmap=seq, vmin=np.min(cross), vmax=1); ax.grid(False)
    ax.set_xticks(range(len(TR)), [f"as {TNAME[t]}" for t in TR], rotation=25, fontsize=8)
    ax.set_yticks(range(len(TR)), [f"{TNAME[t]} ref" for t in TR], fontsize=8)
    mid = np.min(cross) + 0.55 * (1 - np.min(cross))
    for i in range(len(TR)):
        for j in range(len(TR)):
            ax.text(j, i, f"{cross[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white" if cross[i, j] > mid else INK)
    ax.set_title("The gold answer in one variety,\nscored against another", fontsize=9)
    save(fig, charts / "18_cross_variety_references.png")

    # C19 dialect identification
    if did_tab is not None:
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.8), sharey=True)
        palette = {"MSA": "#2a78d6", "CAI": "#eb6834", "BEI": "#1baf7a", "DOH": "#eda100", "RAB": "#e87ba4", "TUN": "#9a76d6"}
        srcs = ["question", "reference", "baseline", "baseline, no image"]
        for ax, t in zip(axes, ("msa", "arz", "ajp")):
            sub = did_tab[did_tab.track == TNAME[t]].set_index("source").reindex(srcs)
            left = np.zeros(len(srcs))
            for lab in DID_NAMES:
                v = sub[lab].fillna(0).to_numpy()
                ax.barh(range(len(srcs)), v, left=left, color=palette[lab], label=DID_NAMES[lab] if t == "msa" else None)
                left += v
            ax.set_yticks(range(len(srcs)), srcs, fontsize=8); ax.invert_yaxis()
            ax.set_title(f"{TNAME[t]} track", fontsize=9); ax.set_xlim(0, 100); ax.set_xlabel("%", fontsize=8)
        fig.legend(fontsize=7, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle("Which variety is the text actually in? (CAMeLBERT MADAR-6 dialect ID)", fontsize=10, fontweight="bold")
        save(fig, charts / "19_dialect_identification.png")

    # C20 F1 vs reference length
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for t in TR:
        ys = [v if by_ref_n.loc[lab, t] >= 20 else np.nan for lab, v in zip(rl_labels, by_ref_tab[TNAME[t]])]
        ax.plot(rl_labels, ys, marker="o", ms=4, lw=2, color=TRACK_COLORS[t], label=TNAME[t])
    ax.set_xlabel("reference length (words)"); ax.set_ylabel("baseline F1")
    ax.set_title("Score by reference length (bins with n >= 20)", fontsize=9); ax.legend(fontsize=7, ncol=2)
    save(fig, charts / "20_f1_by_reference_length.png")

    # C21 vision ablation by intent
    g = abl_tab[abl_tab.by == "question intent"]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.barh(range(len(g)), g["F1 drop"], color=COLORS["baseline_text_only"])
    for k, (v, n) in enumerate(zip(g["F1 drop"], g["items (all tracks)"])):
        ax.text(v + 0.002, k, f"{v:.3f} (n={n})", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(g)), g.group, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("F1 lost when the image is removed (all tracks)")
    ax.set_title("Which questions need the image most", fontsize=9)
    ax.set_xlim(0, g["F1 drop"].max() * 1.35)
    save(fig, charts / "21_vision_ablation_by_intent.png")

    # ================================================================== insights.md
    (work / "report" / "insights.json").write_text(json.dumps(notes, indent=1))
    md = ["# Second-level analysis (auto-generated by `mmcqa_baseline.insights`)\n",
          f"Ids shared across tracks: {notes['ids_shared_across_tracks']}. Id overlap between splits: {notes['id_overlap']}.\n"]
    for title, name in (("Dataset statistics", "11_dataset_statistics"), ("Topic mix by split", "12_topic_mix_by_split"),
                        ("Nearest train question", "13_train_overlap"), ("F1 by train similarity", "13b_f1_by_train_similarity"),
                        ("Answer anatomy", "14_answer_anatomy"), ("Post-processing probes", "15_postprocessing_probes"),
                        ("Metric calibration", "16_metric_calibration"), ("Cross-variety references", "16b_cross_variety_references"),
                        ("Dialect identification", "17_dialect_identification"), ("F1 by dialect match", "17b_f1_by_dialect_match"),
                        ("Item score correlates (Spearman)", "18_item_score_correlates"),
                        ("F1 by reference length", "18b_f1_by_reference_length"),
                        ("Vision ablation by group", "19_vision_ablation_by_group"), ("Statistical power", "20_statistical_power"),
                        ("Devtest vs dev outputs", "21_devtest_vs_dev_outputs")):
        p = tables / f"{name}.md"
        if p.exists():
            md += [f"\n## {title}\n", p.read_text(encoding="utf-8")]
    (work / "report" / "insights.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[insights] wrote {work / 'report' / 'insights.md'}")


if __name__ == "__main__":
    main()
