"""Every setting of the run in one place.

The OFFICIAL block reproduces the organisers' reference system
(``baselines/baseline_qwen_omni.py`` in github.com/mmcultureqa-semeval27/mmcultureqa)
value for value; nothing in it is tuned. The RUN PLAN block only decides which
data the baseline is run on and in what order, so a 12-hour Kaggle session
spends its time on the most useful splits first.
"""

from __future__ import annotations

# ----------------------------------------------------------------------------- provenance
OFFICIAL_REPO_URL = "https://github.com/mmcultureqa-semeval27/mmcultureqa.git"
OFFICIAL_COMMIT = "7ec66a47acb82195a288525560c4acb26cb8b7ba"   # starter kit the settings below come from
DATASET_REPO = "QCRI/MMCQA-SemEval27"
DATASET_REVISION = "3090a38f8eb809a30dcf30a44ed653f77741c056"  # train + dev + devtest, MENA region
REGION = "mena"
TASK = "qa"                                                    # Task 2, Textual Visual QA
TRACKS = ("en", "msa", "arz", "ajp")
TRACK_NAMES = {"en": "English", "msa": "MSA", "arz": "Egyptian", "ajp": "Levantine"}

# ----------------------------------------------------------------------------- OFFICIAL settings
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"    # `main` when the baseline was released
QUANTIZE = True              # 4-bit NF4, double quantisation; compute dtype = bf16 if torch reports support, else fp16
                             # (torch 2.11 on a Kaggle T4 reports support, so the official code runs in bf16 there)
MAX_NEW_TOKENS = 64          # official default
MAX_PIXELS = 512 * 28 * 28   # official default image budget (<= 512 visual tokens)
DO_SAMPLE = False            # greedy decoding
ATTN_IMPLEMENTATION = "sdpa"
# Prompt strings are imported from the official script at run time; copied here for the report.
SYSTEM_PROMPT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
                 "perceiving auditory and visual inputs, as well as generating text and speech.")
INSTRUCTION = ("Answer the question about the image in one or two sentences, in {language}. "
               "Give only the answer.")

# ----------------------------------------------------------------------------- scorer (leaderboard)
SCORER_MODEL = "jhu-clsp/mmBERT-base"
SCORER_REVISION = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
SCORER_LAYER = 17
COMPARISON_SCORER = ("bert-base-multilingual-cased", 9)   # the scorer's optional second backbone

# ----------------------------------------------------------------------------- RUN PLAN
# Stages run in this order; each is (name, split, rows per track or None for all, mode).
# mode "official" = the baseline exactly as released; "text_only" = the same call with the
# image removed (an ablation, reported separately, never mixed into the baseline numbers).
# Full train (40,000 generations, ~100 T4-hours at the measured ~9 s/item) does not fit a
# Kaggle session, so train is covered by a seeded random sample that shares ids across tracks.
STAGES = [
    ("dev", "dev", None, "official"),                     # 4 x 1,000, has references -> the main result
    ("devtest", "devtest", None, "official"),             # 4 x 1,000, no references -> leaderboard files
    ("train_sample", "train", 250, "official"),           # 4 x 250, has references -> generalisation check
    ("dev_text_only", "dev", 250, "text_only"),           # 4 x 250, ablation: does the image matter?
]
SEED = 2027
WORKERS_PER_GPU = 2          # bnb 4-bit decoding at batch 1 leaves a T4 mostly idle; 2 processes per card
TIME_BUDGET_HOURS = 10.5     # workers stop taking new items after this, so scoring still fits in 12 h
BOOTSTRAP_SAMPLES = 2000
