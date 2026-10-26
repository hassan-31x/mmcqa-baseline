"""Turn scores and run logs into the tables, charts and examples the report needs.

    python -m mmcqa_baseline.analysis --workdir results

Reads  scores/per_item_scores.jsonl, scores/official_metrics.json, predictions_all.jsonl, logs/
Writes report/tables/*.csv|.md, report/charts/*.png, report/examples/*.md, report/summary.md|.json

Every number in summary.md is computed here; nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, settings as S

TR = S.TRACKS
TNAME = S.TRACK_NAMES
SYSTEM_LABEL = {"official_baseline": "Official baseline (Qwen2.5-Omni-3B)",
                "baseline_text_only": "Baseline, image removed",
                "baseline_trimmed": "Baseline, invented turns trimmed",
                "echo_question": "Echo the question",
                "tfidf_retrieval": "TF-IDF train retrieval"}
SHORT = {"official_baseline": "Baseline", "baseline_text_only": "No image", "baseline_trimmed": "Trimmed",
         "echo_question": "Echo question", "tfidf_retrieval": "TF-IDF retrieval"}
COLORS = {"official_baseline": "#2a78d6", "tfidf_retrieval": "#eb6834", "echo_question": "#1baf7a",
          "baseline_trimmed": "#eda100", "baseline_text_only": "#e87ba4"}
TRACK_COLORS = dict(zip(TR, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
RNG = np.random.default_rng(S.SEED)

ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")
TURN = re.compile(r"\b(?:Human|User|Assistant)\s*:")
STRIP = "؟?!.,،:;\"'()[]«»*-–—"
COLLOQUIAL = {"اللي", "مش", "زي", "كمان", "بس", "إللي"}
EGY = {"ده", "دي", "دول", "كده", "كدا", "إزاي", "ازاي", "إيه", "ايه", "عشان", "علشان", "بتاع", "بتاعة",
       "بتاعت", "أوي", "اوي", "فين", "دلوقتي", "باين", "باينة", "بيورّي", "بتورّي", "بيوري", "بتوري", "هيا"}
LEV = {"شو", "هيك", "هاد", "هادا", "هاي", "هلق", "هلأ", "منيح", "كتير", "عم", "وين", "ليش", "بدو", "بدها",
       "هون", "هنيك", "مبين", "مبينة", "مبينين", "متل", "لهيك", "هيدا", "هيدي", "بيبين", "بتبين"}
MSA = {"هذه", "هذا", "التي", "الذي", "الذين", "يتم", "حيث", "إنها", "إنه", "ليس", "لذلك", "تُعد", "يُعد",
       "تعتبر", "يعتبر", "ذلك", "تلك"}
# Topic lexicon applied to the ENGLISH question + reference of each item (ids are shared across tracks,
# so every track inherits the same topic). First match wins, in this order.
TOPICS = [
    ("Food & drink", r"food|dish|eat|drink|coffee|tea\b|bread|sweet|dessert|cuisine|meal|spice|date[s]?\b|rice|meat|cook"),
    ("Religion & festivals", r"ramadan|eid|pray|mosque|relig|islam|church|festiv|celebrat|wedding|ceremon|holiday|pilgrim"),
    ("Clothing & textiles", r"cloth|wear|dress|garment|scarf|abaya|thobe|hijab|fabric|textile|embroider|costume|attire"),
    ("Architecture & places", r"build|architect|city|market|souq|souk|landmark|tower|palace|fort|castle|street|house|ruin|museum"),
    ("Arts, crafts & music", r"craft|potter|carpet|rug|music|instrument|\bart|calligraph|dance|paint|jewel|lantern|decor"),
    ("Nature & animals", r"camel|desert|animal|plant|palm|sea\b|falcon|horse|tree|mountain|oasis|bird|beach"),
]
# Question intent from the English question (993/1000 dev questions start with "What", so wh-words
# say nothing; the intent of the question does). First match wins, in this order.
QTYPES = [
    ("Significance / symbolism", r"signific|symbol|meaning|represent|role|importan|reflect|tradition"),
    ("Purpose / use", r"used|purpose|function|serve"),
    ("Preparation / material", r"made|prepar|ingredient|cook|material|craft"),
    ("Place / origin", r"where|country|region|origin|locat|city"),
    ("Occasion / time", r"when|occasion|time|season|festiv|celebrat"),
    ("Identify / name", r"\bname\b|called|what (?:is|are) (?:this|these|the)\b|identify|which|what type|what kind"),
    ("Describe the scene", r"element|feature|happening|activit|depict|describ|scene|indicate|event|show"),
]


# ----------------------------------------------------------------------------- helpers
def words(text: str) -> list[str]:
    return [w.strip(STRIP) for w in str(text).split() if w.strip(STRIP)]


def script_ok(text: str, track: str) -> bool:
    a, l = len(ARABIC.findall(text)), len(LATIN.findall(text))
    if a + l == 0:
        return False
    return (l / (a + l) >= 0.5) if track == "en" else (a / (a + l) >= 0.5)


def dialect_flags(text: str) -> dict:
    w = set(words(text))
    return {"egy": bool(w & EGY) or bool(w & COLLOQUIAL), "lev": bool(w & LEV) or bool(w & COLLOQUIAL),
            "msa": bool(w & MSA)}


def topic_of(text: str) -> str:
    low = text.lower()
    return next((name for name, pat in TOPICS if re.search(pat, low)), "Other")


def qtype_of(question: str) -> str:
    low = question.strip().lower()
    return next((name for name, pat in QTYPES if re.search(pat, low)), "Other")


def boot_ci(x: np.ndarray, n: int = S.BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    means = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_test(a: np.ndarray, b: np.ndarray, n: int = 10000) -> float:
    """Two-sided paired sign-flip permutation test on the mean difference."""
    d = a - b
    obs = abs(d.mean())
    flips = RNG.choice([-1.0, 1.0], size=(n, len(d)))
    return float(((np.abs((flips * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))


def write_table(df: pd.DataFrame, path: Path, floatfmt: str = "{:.4f}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else
                 ("" if isinstance(v, float) and np.isnan(v) else str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "legend.frameon": False, "font.family": "DejaVu Sans"})
    return plt


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)   # vector copy for LaTeX
    import matplotlib.pyplot as plt
    plt.close(fig)


# ----------------------------------------------------------------------------- analysis
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("results"))
    args = ap.parse_args()
    work = args.workdir.resolve()
    out = work / "report"
    tables, charts, examples = out / "tables", out / "charts", out / "examples"
    plt = setup_mpl()
    summary: dict = {}

    df = pd.DataFrame(data.read_jsonl(work / "scores" / "per_item_scores.jsonl"))
    official = json.loads((work / "scores" / "official_metrics.json").read_text())["results"]
    raw = pd.DataFrame(data.read_jsonl(work / "predictions_all.jsonl"))
    cfg = json.loads((work / "logs" / "run_config.json").read_text())
    completion = json.loads((work / "logs" / "completion.json").read_text())
    summary["run_config"] = {k: cfg[k] for k in ("official_commit", "dataset_revision", "model", "model_revision",
                                                 "max_new_tokens", "max_pixels", "quantize_4bit_nf4")}
    summary["environment"] = cfg["environment"]
    summary["completion"] = completion

    # English question/reference per id -> topic and question type for every track
    en = df[(df.track == "en") & df.system.isin(["official_baseline", "baseline_text_only"])]
    topic = {i: topic_of(q + " " + r) for i, q, r in zip(en.id, en.question, en.reference)}
    qtype = {i: qtype_of(q) for i, q in zip(en.id, en.question)}
    df["topic"] = df.id.map(topic).fillna("Other")
    df["qtype"] = df.id.map(qtype).fillna("Other")
    df["pred_words"] = df.prediction.map(lambda t: len(words(t)))
    df["ref_words"] = df.reference.map(lambda t: len(words(t)))

    stages = [s for s in ("dev", "train_sample") if (df.stage == s).any()]
    base = df[df.system == "official_baseline"]

    # 1. main results: official metrics + bootstrap CI, per stage x system x track, plus macro
    rows = []
    for stage in stages + (["dev_text_only"] if (df.stage == "dev_text_only").any() else []):
        for system in df[df.stage == stage].system.unique():
            track_f1 = []
            for t in TR:
                key = f"{system}|{stage}|{t}"
                if key not in official:
                    continue
                o = official[key]
                f = df[(df.system == system) & (df.stage == stage) & (df.track == t)].mm_f1.to_numpy()
                lo, hi = boot_ci(f)
                track_f1.append(f)
                rows.append({"stage": stage, "system": system, "track": TNAME[t], "n": o["n_items"],
                             "BERTScore-F1": o["BERTScore-F1"], "F1 95% CI": f"[{lo:.4f}, {hi:.4f}]",
                             "BERTScore-P": o["BERTScore-P"], "BERTScore-R": o["BERTScore-R"],
                             "BLEU": o["BLEU"], "ROUGE-L": o["ROUGE-L"], "Coverage": o["Coverage"]})
            if len(track_f1) == len(TR):
                boots = np.mean([f[RNG.integers(0, len(f), (S.BOOTSTRAP_SAMPLES, len(f)))].mean(1)
                                 for f in track_f1], axis=0)
                sub = [r for r in rows if r["stage"] == stage and r["system"] == system]
                rows.append({"stage": stage, "system": system, "track": "Macro avg", "n": sum(r["n"] for r in sub),
                             "BERTScore-F1": float(np.mean([r["BERTScore-F1"] for r in sub])),
                             "F1 95% CI": f"[{np.percentile(boots, 2.5):.4f}, {np.percentile(boots, 97.5):.4f}]",
                             **{m: float(np.mean([r[m] for r in sub])) for m in
                                ("BERTScore-P", "BERTScore-R", "BLEU", "ROUGE-L", "Coverage")}})
    main_tab = pd.DataFrame(rows)
    main_tab["system"] = main_tab.system.map(SYSTEM_LABEL)
    write_table(main_tab, tables / "01_main_results")
    summary["main_results"] = rows

    # 2. significance: baseline vs every other system on identical items (dev)
    sig = []
    for stage in stages:
        for other in ("tfidf_retrieval", "echo_question", "baseline_trimmed"):
            for t in TR:
                a = df[(df.system == "official_baseline") & (df.stage == stage) & (df.track == t)].set_index("id").mm_f1
                b = df[(df.system == other) & (df.stage == stage) & (df.track == t)].set_index("id").mm_f1
                common = a.index.intersection(b.index)
                if len(common) == 0:
                    continue
                a, b = a[common].to_numpy(), b[common].to_numpy()
                sig.append({"stage": stage, "comparison": f"Baseline vs {SHORT[other]}", "track": TNAME[t],
                            "n": len(common), "baseline F1": a.mean(), "other F1": b.mean(),
                            "difference": a.mean() - b.mean(), "baseline wins %": 100 * (a > b).mean(),
                            "p (paired permutation)": paired_test(a, b)})
    if (df.stage == "dev_text_only").any():
        for t in TR:
            a = base[(base.stage == "dev") & (base.track == t)].set_index("id").mm_f1
            b = df[(df.system == "baseline_text_only") & (df.track == t)].set_index("id").mm_f1
            common = a.index.intersection(b.index)
            if len(common):
                a, b = a[common].to_numpy(), b[common].to_numpy()
                sig.append({"stage": "dev (ablation ids)", "comparison": "With image vs image removed",
                            "track": TNAME[t], "n": len(common), "baseline F1": a.mean(), "other F1": b.mean(),
                            "difference": a.mean() - b.mean(), "baseline wins %": 100 * (a > b).mean(),
                            "p (paired permutation)": paired_test(a, b)})
    sig_tab = pd.DataFrame(sig)
    write_table(sig_tab, tables / "02_significance")
    summary["significance"] = sig

    # 3. output diagnostics for the baseline (what its answers look like)
    diag = []
    for stage in stages:
        for t in TR:
            b = base[(base.stage == stage) & (base.track == t)]
            if b.empty:
                continue
            flags = [dialect_flags(x) for x in b.prediction]
            rflags = [dialect_flags(x) for x in b.reference]
            own = {"arz": "egy", "ajp": "lev"}.get(t)
            diag.append({
                "stage": stage, "track": TNAME[t], "n": len(b),
                "pred words (mean)": b.pred_words.mean(), "ref words (mean)": b.ref_words.mean(),
                "length ratio pred/ref": (b.pred_words / b.ref_words.clip(lower=1)).median(),
                "hit 64-token cap %": 100 * b.hit_max_new_tokens.eq(True).mean(),
                "invented dialogue turn %": 100 * b.prediction.map(lambda x: bool(TURN.search(x))).mean(),
                "wrong script %": 100 * (~b.apply(lambda r: script_ok(r.prediction, t), axis=1)).mean(),
                "empty or failed %": 100 * ((b.prediction == "") | b.error.notna()).mean(),
                "MSA markers in pred %": 100 * np.mean([f["msa"] for f in flags]) if t != "en" else np.nan,
                "dialect markers pred %": 100 * np.mean([f[own] for f in flags]) if own else np.nan,
                "dialect markers ref %": 100 * np.mean([f[own] for f in rflags]) if own else np.nan,
                "precision - recall": (b.mm_p - b.mm_r).mean()})
    diag_tab = pd.DataFrame(diag)
    write_table(diag_tab, tables / "03_output_diagnostics", "{:.2f}")
    summary["output_diagnostics"] = diag

    dev = base[base.stage == "dev"] if "dev" in stages else base[base.stage == stages[0]]

    # 4. breakdowns: topic, question type, prediction length
    def breakdown(col, order=None):
        g = dev.groupby([col, "track"]).mm_f1.agg(["mean", "count"]).reset_index()
        piv = g.pivot(index=col, columns="track", values="mean").reindex(columns=list(TR))
        cnt = g.pivot(index=col, columns="track", values="count").reindex(columns=list(TR)).fillna(0).astype(int)
        piv = piv.rename(columns=TNAME)
        if (cnt.nunique(axis=1) <= 1).all():     # topic / intent: the same items in every track
            piv["n per track"] = cnt.iloc[:, 0]
        else:                                    # length bins differ per track, so give each count
            for t in TR:
                piv[f"n {TNAME[t]}"] = cnt[t]
        piv["all tracks"] = piv[[TNAME[t] for t in TR]].mean(1)
        piv = piv.sort_values("all tracks", ascending=False) if order is None else piv.reindex(order)
        n_cols = [c for c in piv.columns if str(c).startswith("n ")]
        piv[n_cols] = piv[n_cols].fillna(0).astype(int)
        return piv.reset_index()

    topic_tab = breakdown("topic")
    write_table(topic_tab, tables / "04_f1_by_topic")
    qtype_tab = breakdown("qtype")
    write_table(qtype_tab, tables / "05_f1_by_question_type")
    bins = [0, 10, 20, 30, 40, 60, 1000]
    labels = ["1-10", "11-20", "21-30", "31-40", "41-60", "60+"]
    dev = dev.assign(len_bin=pd.cut(dev.pred_words, bins=bins, labels=labels, include_lowest=True).astype(str))
    len_tab = breakdown("len_bin", order=labels)
    write_table(len_tab, tables / "06_f1_by_answer_length")
    summary["by_topic"] = topic_tab.to_dict("records")
    summary["by_question_type"] = qtype_tab.to_dict("records")

    # 5. cross-track: the same item in four varieties
    piv = dev.pivot_table(index="id", columns="track", values="mm_f1").reindex(columns=list(TR)).dropna()
    corr = piv.corr(method="spearman").rename(index=TNAME, columns=TNAME)
    write_table(corr.reset_index().rename(columns={"track": ""}), tables / "07_cross_track_spearman", "{:.3f}")
    summary["cross_track_spearman"] = corr.to_dict()

    # 6. metric agreement (per item, baseline + reference systems on dev)
    dsys = df[df.stage == (stages[0] if stages else "dev")]
    agree = []
    for t in TR:
        sub = dsys[dsys.track == t]
        row = {"track": TNAME[t], "mmBERT F1 vs ROUGE-L": sub.mm_f1.corr(sub.rougeL, method="spearman")}
        if sub.mbert_f1.notna().any():
            row["mmBERT F1 vs mBERT F1"] = sub.mm_f1.corr(sub.mbert_f1, method="spearman")
            for s in sub.system.unique():
                row[f"mBERT F1, {SHORT[s]}"] = sub[sub.system == s].mbert_f1.mean()
        agree.append(row)
    write_table(pd.DataFrame(agree), tables / "08_metric_agreement", "{:.3f}")
    summary["metric_agreement"] = agree

    # 7. train sample vs dev (does dev look like the data it was split from?)
    if "train_sample" in stages and "dev" in stages:
        tv = []
        for t in TR:
            for s in ("official_baseline", "tfidf_retrieval", "echo_question"):
                a = df[(df.system == s) & (df.stage == "dev") & (df.track == t)].mm_f1
                b = df[(df.system == s) & (df.stage == "train_sample") & (df.track == t)].mm_f1
                tv.append({"track": TNAME[t], "system": SHORT[s], "dev F1": a.mean(), "train-sample F1": b.mean(),
                           "difference": a.mean() - b.mean(), "n dev": len(a), "n train": len(b)})
        write_table(pd.DataFrame(tv), tables / "09_train_vs_dev")
        summary["train_vs_dev"] = tv

    # 8. efficiency
    r = raw[raw["mode"] == "official"]
    eff = {"generations": int(len(raw)), "official generations": int(len(r)),
           "latency mean s": float(r.latency_seconds.mean()), "latency median s": float(r.latency_seconds.median()),
           "latency p95 s": float(r.latency_seconds.quantile(0.95)),
           "new tokens mean": float(r.new_tokens.mean()),
           "decode tokens/s per worker": float((r.new_tokens / r.latency_seconds).median()),
           "prompt tokens mean": float(r.prompt_tokens.mean()),
           "peak GPU memory GB per worker": float(raw.peak_mem_gb.max())}
    wall = work / "logs" / "wall_clock.json"
    if wall.exists():
        w = json.loads(wall.read_text())
        eff.update({"inference wall clock h": w["inference_wall_seconds"] / 3600, "workers": w["workers"],
                    "gpus": ", ".join(w["gpus"]),
                    "throughput items/s (all workers)": len(raw) / max(w["inference_wall_seconds"], 1)})
        eff["projected hours, full train (40k) on this setup"] = 40000 / eff["throughput items/s (all workers)"] / 3600
    eff["projected T4-hours per 1,000 items, single worker"] = 1000 * eff["latency mean s"] / 3600
    for t in TR:
        eff[f"latency mean s, {TNAME[t]}"] = float(r[r.track == t].latency_seconds.mean())
    write_table(pd.DataFrame([{"metric": k, "value": v} for k, v in eff.items()]), tables / "10_efficiency", "{:.3f}")
    summary["efficiency"] = eff

    # 9. qualitative examples and an error sample for manual annotation
    examples.mkdir(parents=True, exist_ok=True)
    for t in TR:
        sub = dev[dev.track == t].sort_values("mm_f1")
        picks = [("Lowest F1", sub.head(4)), ("Highest F1", sub.tail(4).iloc[::-1]),
                 ("Random", sub.sample(min(4, len(sub)), random_state=S.SEED))]
        md = [f"# {TNAME[t]}: baseline answers next to the references (dev)\n"]
        for title, part in picks:
            md.append(f"## {title}\n")
            for _, x in part.iterrows():
                md.append(f"- **id** `{x.id[:12]}` | F1 {x.mm_f1:.3f} | P {x.mm_p:.3f} | R {x.mm_r:.3f} | "
                          f"topic {x.topic}\n  - **Q:** {x.question}\n  - **Reference:** {x.reference}\n"
                          f"  - **Baseline:** {x.prediction}\n")
        (examples / f"examples_{t}.md").write_text("\n".join(md), encoding="utf-8")
    err = dev.sample(min(200, len(dev)), random_state=S.SEED)[["track", "id", "topic", "question", "reference",
                                                              "prediction", "mm_f1"]]
    err = err.assign(manual_label="", note="")   # correct / partially_correct / incorrect, filled by hand
    err.to_csv(examples / "error_annotation_sample.csv", index=False)

    # ------------------------------------------------------------------ charts
    systems_dev = [s for s in ("official_baseline", "tfidf_retrieval", "echo_question") if
                   ((df.system == s) & (df.stage == "dev")).any()]
    mt = pd.DataFrame(rows)

    # C1 main result: grouped bars with 95% CIs, systems x tracks
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    width = 0.8 / max(len(systems_dev), 1)
    for k, s in enumerate(systems_dev):
        vals, errs = [], []
        for t in list(TR) + ["Macro avg"]:
            name = TNAME.get(t, t)
            rr = mt[(mt.stage == "dev") & (mt.system == s) & (mt.track == name)]
            v = float(rr["BERTScore-F1"].iloc[0]) if len(rr) else np.nan
            lo, hi = (map(float, rr["F1 95% CI"].iloc[0].strip("[]").split(", ")) if len(rr) else (v, v))
            vals.append(v)
            errs.append((v - lo, hi - v))
        x = np.arange(len(TR) + 1) + (k - (len(systems_dev) - 1) / 2) * width
        ax.bar(x, vals, width * 0.92, color=COLORS[s], label=SYSTEM_LABEL[s], zorder=2)
        ax.errorbar(x, vals, yerr=np.array(errs).T, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
        if s == "official_baseline":
            for xi, v, (_, up) in zip(x, vals, errs):
                ax.text(xi, v + up + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    lo_y = np.nanmin(mt[mt.stage == "dev"]["BERTScore-F1"]) - 0.03
    ax.set_ylim(max(0, lo_y), None)
    ax.set_xticks(np.arange(len(TR) + 1), [TNAME[t] for t in TR] + ["Macro avg"])
    ax.set_ylabel("BERTScore-F1 (mmBERT, official)")
    ax.set_title("Dev set: official baseline vs simple reference systems")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8)
    save(fig, charts / "01_main_results.png")

    # C2 per-item F1 distributions (baseline), one small panel per track on a shared axis
    fig, axes = plt.subplots(1, len(TR), figsize=(6.6, 2.2), sharex=True, sharey=True)
    lo_f = float(dev.mm_f1[dev.mm_f1 > 0].quantile(0.002)) if (dev.mm_f1 > 0).any() else 0.0
    bins_f = np.linspace(lo_f, float(dev.mm_f1.max()) + 1e-6, 30)
    echo_dev = df[(df.system == "echo_question") & (df.stage == "dev")]
    for ax, t in zip(axes, TR):
        f = dev[dev.track == t].mm_f1
        ax.hist(f, bins=bins_f, color=TRACK_COLORS[t], edgecolor=SURFACE, linewidth=0.4)
        ax.axvline(f.mean(), color=INK, lw=1)
        if len(echo_dev):
            ax.axvline(echo_dev[echo_dev.track == t].mm_f1.mean(), color=MUTED, lw=1, ls="--")
        ax.set_title(f"{TNAME[t]}: {f.mean():.3f}", fontsize=8)
        ax.set_xlabel("F1", fontsize=8)
    axes[0].set_ylabel("Items")
    fig.suptitle("Per-item scores, official baseline (dev). Solid: mean; dashed: echo-question mean",
                 fontsize=9, fontweight="bold")
    save(fig, charts / "02_f1_distribution.png")

    # C3 precision vs recall per track
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for s in systems_dev:
        for t in TR:
            rr = mt[(mt.stage == "dev") & (mt.system == s) & (mt.track == TNAME[t])]
            if len(rr):
                ax.scatter(rr["BERTScore-R"], rr["BERTScore-P"], s=40, color=COLORS[s],
                           marker="osD^"[TR.index(t)], edgecolor=SURFACE, linewidth=1.2, zorder=3)
    lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
    ax.plot(lims, lims, color=MUTED, lw=0.8, ls="--", zorder=1)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[s], marker="o", ls="", label=SHORT[s]) for s in systems_dev] + \
              [Line2D([], [], color=INK2, marker="osD^"[k], ls="", label=TNAME[t]) for k, t in enumerate(TR)]
    ax.legend(handles=handles, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_xlabel("BERTScore-R"); ax.set_ylabel("BERTScore-P")
    ax.set_title("Precision vs recall (dev)")
    save(fig, charts / "03_precision_recall.png")

    # C4 answer length: baseline vs reference, per track
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    pos, data_, cols, ticks = [], [], [], []
    for k, t in enumerate(TR):
        sub = dev[dev.track == t]
        data_ += [sub.ref_words.to_numpy(), sub.pred_words.to_numpy()]
        pos += [k * 3, k * 3 + 1]
        cols += ["#c3c2b7", COLORS["official_baseline"]]
        ticks.append(k * 3 + 0.5)
    bp = ax.boxplot(data_, positions=pos, widths=0.8, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.2})
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_edgecolor(SURFACE)
    ax.set_xticks(ticks, [TNAME[t] for t in TR]); ax.set_ylabel("Words per answer")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c3c2b7", label="Reference"),
                       Patch(color=COLORS["official_baseline"], label="Baseline")], fontsize=8, loc="upper right")
    ax.set_title("Answer length, baseline vs reference (dev)")
    save(fig, charts / "04_answer_length.png")

    # C5 failure modes per track (small multiples of rates, one measure per panel)
    dt = diag_tab[diag_tab.stage == ("dev" if "dev" in stages else stages[0])]
    measures = ["hit 64-token cap %", "invented dialogue turn %", "wrong script %"]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4), sharey=True)
    for ax, m in zip(axes, measures):
        vals = dt[m].to_numpy()
        ax.bar(range(len(vals)), vals, color=[TRACK_COLORS[t] for t in TR][:len(vals)], width=0.7)
        for k, v in enumerate(vals):
            ax.text(k, v + 1, f"{v:.0f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(range(len(vals)), dt.track, rotation=30, fontsize=7)
        ax.set_title(m.replace(" %", ""), fontsize=8)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("% of answers")
    fig.suptitle("How the baseline's answers go wrong (dev)", fontsize=10, fontweight="bold")
    save(fig, charts / "05_failure_modes.png")

    # C6 topic x track heatmap (sequential, one hue)
    import matplotlib.colors as mcolors
    seq = mcolors.LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#6fa6e8", "#0d366b"])
    for name, tab, fname in (("Topic", topic_tab, "06_topic_heatmap.png"),
                             ("Question intent", qtype_tab, "07_question_type_heatmap.png")):
        mat = tab[[TNAME[t] for t in TR]].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(6.2, 0.38 * len(tab) + 1.0))
        im = ax.imshow(mat, cmap=seq, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(TR)), [TNAME[t] for t in TR])
        ax.set_yticks(range(len(tab)), [f"{a} (n={n})" for a, n in zip(tab.iloc[:, 0], tab["n per track"])])
        lo_v, hi_v = np.nanmin(mat), np.nanmax(mat)
        mid = lo_v + 0.55 * (hi_v - lo_v)        # white ink only on the dark half of the ramp
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > mid else INK)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).outline.set_visible(False)
        ax.set_title(f"Baseline F1 by {name.lower()} (dev)")
        save(fig, charts / fname)

    # C8 F1 vs prediction length bucket
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    counts = dev.groupby(["len_bin", "track"]).size()
    for t in TR:
        # buckets with fewer than 10 answers are noise, so they are left out
        ys = [v if counts.get((lab, t), 0) >= 10 else np.nan for lab, v in zip(labels, len_tab[TNAME[t]])]
        ax.plot(labels, ys, marker="o", ms=5, lw=2, color=TRACK_COLORS[t], label=TNAME[t])
    ax.set_xlabel("Baseline answer length (words)"); ax.set_ylabel("Mean BERTScore-F1")
    ax.set_title("Score vs answer length (dev; buckets with n ≥ 10)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, charts / "08_f1_vs_length.png")

    # C9 cross-track correlation heatmap
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    m = corr.to_numpy()
    ax.imshow(m, cmap=seq, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(TR)), corr.columns, rotation=30); ax.set_yticks(range(len(TR)), corr.index)
    for i in range(len(TR)):
        for j in range(len(TR)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if m[i, j] > .5 else INK)
    ax.set_title("Same item, different variety:\nSpearman of per-item F1")
    save(fig, charts / "09_cross_track_correlation.png")

    # C10 ablation and train-vs-dev (two panels, same scale: F1)
    panels = []
    if (df.stage == "dev_text_only").any():
        panels.append("ablation")
    if "train_sample" in stages and "dev" in stages:
        panels.append("train")
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.8), squeeze=False)
        for ax, p in zip(axes[0], panels):
            if p == "ablation":
                abl = sig_tab[sig_tab.comparison == "With image vs image removed"]
                a_vals, b_vals, lab = abl["baseline F1"], abl["other F1"], ("With image", "Image removed")
                ca, cb = COLORS["official_baseline"], COLORS["baseline_text_only"]
                title = "Ablation: remove the image"
            else:
                tvd = pd.DataFrame(summary["train_vs_dev"])
                tvd = tvd[tvd.system == "Baseline"]
                a_vals, b_vals, lab = tvd["dev F1"], tvd["train-sample F1"], ("Dev", "Train sample")
                ca, cb = COLORS["official_baseline"], "#c3c2b7"
                title = "Dev vs train sample"
            x = np.arange(len(TR))
            ax.bar(x - 0.2, a_vals, 0.38, color=ca, label=lab[0])
            ax.bar(x + 0.2, b_vals, 0.38, color=cb, label=lab[1])
            ax.set_xticks(x, [TNAME[t] for t in TR], fontsize=8)
            low = min(np.nanmin(a_vals), np.nanmin(b_vals))
            ax.set_ylim(low - 0.03, max(np.nanmax(a_vals), np.nanmax(b_vals)) + 0.02)
            ax.set_title(title); ax.legend(fontsize=7, loc="upper right")
        axes[0][0].set_ylabel("BERTScore-F1")
        save(fig, charts / "10_ablation_and_train_vs_dev.png")

    # C11 latency distribution
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.hist(r.latency_seconds.clip(upper=r.latency_seconds.quantile(0.995)), bins=50,
            color=COLORS["official_baseline"], edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(r.latency_seconds.median(), color=INK, lw=1, ls="--")
    ax.text(r.latency_seconds.median(), ax.get_ylim()[1] * 0.92, f"  median {r.latency_seconds.median():.1f}s",
            fontsize=7, color=INK)
    ax.set_xlabel("Seconds per item (one worker, T4, 4-bit)"); ax.set_ylabel("Items")
    ax.set_title("Inference latency")
    save(fig, charts / "11_latency.png")

    # ------------------------------------------------------------------ summary
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    md = ["# Baseline run: summary (auto-generated)\n",
          f"Model `{cfg['model']}` @ `{cfg['model_revision'][:10]}` (4-bit NF4, greedy, max_new_tokens "
          f"{cfg['max_new_tokens']}, max_pixels {cfg['max_pixels']}); official starter kit "
          f"`{cfg['official_commit'][:10]}`; dataset `{cfg['dataset_revision'][:10]}`; "
          f"GPUs: {', '.join(cfg['environment']['gpus'])}.\n",
          "## Completion\n", "| stage/track | answered | errors | format check |", "|---|---|---|---|"]
    for k, st in completion["status"].items():
        md.append(f"| {k} | {st['answered']}/{st['planned']} | {st['errors']} | "
                  f"{completion['format_check'][k].split('|')[0].strip()} |")
    for title, name in (("Main results", "01_main_results"), ("Significance", "02_significance"),
                        ("Output diagnostics", "03_output_diagnostics"), ("F1 by topic", "04_f1_by_topic"),
                        ("F1 by question intent", "05_f1_by_question_type"),
                        ("F1 by answer length", "06_f1_by_answer_length"),
                        ("Cross-track Spearman", "07_cross_track_spearman"),
                        ("Metric agreement", "08_metric_agreement"), ("Train sample vs dev", "09_train_vs_dev"),
                        ("Efficiency", "10_efficiency")):
        p = tables / f"{name}.md"
        if p.exists():
            md += [f"\n## {title}\n", p.read_text(encoding="utf-8")]
    md.append("\nCharts: report/charts/ (PNG + PDF). Examples: report/examples/.\n")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[analysis] wrote {out}")


if __name__ == "__main__":
    main()
