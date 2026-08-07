# Autonomous Agent Prediction (Beta) — Kaggle

CODE repo for the Kaggle Competition: <https://www.kaggle.com/competitions/autonomous-agent-prediction-beta>

---

## Layout

```
.
├── agent.yaml ...           # (starter kit files, from the competition dataset)
├── data/train_01..16/       # 16 provided training datasets   [gitignored]
├── models.yaml              # available LLMs + token prices
├── run_local_eval.py        # local end-to-end agent evaluation
├── validate_submission.py   # pre-flight linter
├── wheels/                  # adk_submission + kaggle_kaggle  [gitignored]
├── sample_submission/       # reference agent from the starter kit, left untouched
├── scripts/
│   ├── profile_datasets.py       # characterise the 16 datasets
│   ├── bench_v2.py               # benchmark model zoo, no LLM cost -- imports run_automl.py directly
│   ├── bench_v2_driver.sh        # runs bench_v2.py once per dataset, one process each
│   ├── error_analysis.py         # AUC-attribution honest-split diagnostic (which datasets/segments are weak)
│   └── parse_eval_trace.py       # trace debugging
└── submissions/
    ├── 01_baseline/       # first working submission, 0.812
    ├── 02_lean/           # cheap-model variant, time-boxed zoo
    ├── 03_skilled/        # current real submission -- correctly invokes the bundled skill via run_skill_script
    ├── 03_skilled_haiku/  # local-test-only: same skill, Haiku orchestrator (sanity-checking under a cheaper model)
    ├── 03_skilled_opus/   # local-test-only: same skill, Opus orchestrator
    └── 04_reflective/     # experimental: live weak-segment diagnosis + targeted recipe, NOT promoted to 03_skilled
        ├── agent/           # <- the thing that gets zipped
        │   ├── agent.yaml
        │   ├── prompts/system.md
        │   ├── configs/sampling.yaml
        │   └── skills/tabular-automl/
        ├── output/          # eval traces           [gitignored]
        └── submission.zip                            [gitignored]
```

---

## Setup

**The evaluation harness runs in WSL Ubuntu.** `litellm >= 1.83` ships no Windows wheel and needs Rust + the MSVC linker to build from source; on Linux it installs from a wheel in seconds. WSL also reaches Docker Desktop directly.

```bash
wsl -d Ubuntu -e bash ./setup_wsl.sh      # creates ~/.venvs/kik and installs everything
```

The venv is built with `uv`, not `pip` -- `pip` isn't installed in it by design. Add packages with:

```bash
~/.local/bin/uv pip install --python ~/.venvs/kik/bin/python <package>
```

Then add an LLM key for local evaluation (Kaggle's own proxy is only used during official runs):

```powershell
Copy-Item .env.example .env
# edit .env -> GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
```

Local evaluation needs **Docker Desktop running** — it launches `gcr.io/kaggle-images/python` as the agent sandbox.

```powershell
docker pull gcr.io/kaggle-images/python
```

The offline benchmark and dataset profiler are pure pandas/sklearn and run fine under plain Windows Python, though `bench_v2.py`/`error_analysis.py` need lightgbm/catboost, which are only installed in the WSL venv.

---

## Workflow

The loop I actually use when iterating, cheapest step first:

```bash
# 1. Offline: is the modelling pipeline any good? (costs no LLM tokens, runs anywhere)
python scripts/bench_v2.py --dataset 1 --row-out /tmp/row.json
python scripts/bench_v2.py --summarize /tmp/rows_dir --out results.csv   # after running several datasets

# 2. Lint the agent config          [WSL]
~/.venvs/kik/bin/python validate_submission.py --agent-dir submissions/03_skilled/agent

# 3. Full local session on one dataset (costs your own API key)   [WSL + Docker]
~/.venvs/kik/bin/python run_local_eval.py \
    --submission-dir submissions/03_skilled/agent \
    --dataset train_01 --metric roc_auc

# 4. Read the trace                 [WSL]
~/.venvs/kik/bin/python scripts/parse_eval_trace.py --experiment-dir submissions/03_skilled
```

```powershell
# 5. Package and submit             [Windows]
Compress-Archive -Path submissions/03_skilled/agent/* -DestinationPath submissions/03_skilled/submission.zip -Force
kaggle competitions submit autonomous-agent-prediction-beta -f submissions/03_skilled/submission.zip -m "describe the change"
kaggle competitions submissions autonomous-agent-prediction-beta
```

Note: Reminder for step 5: `agent.yaml` has to end up at the root of the zip, not in a subfolder. `Compress-Archive`'s backslash path separators can also break `compile_submission` — building the zip with Python's `zipfile` module and `.as_posix()` arcnames is safer than `Compress-Archive` for anything beyond a quick local check.

---

## The baseline design

A cheap orchestrator LLM drives a **pre-tested bundled skill** rather than authoring ML code turn by turn. This attacks the two things that actually cost score: the $2 token budget, and the most common failure mode — *"agent completed without submitting any valid predictions."* The prompt forces a safety submission inside the first few minutes, then runs a cross-validated model zoo (LightGBM, a CV-tuned LightGBM variant, HistGradientBoosting, CatBoost, logistic regression, plus NNLS-weighted-blend and logistic-stacking ensemble candidates) and picks per dataset by CV. `01_baseline` and `02_lean` both scored ~0.81 by instructing the agent to run the skill via `run_command`, which silently doesn't work — bundled skills are only reachable through `run_skill_script`, never materialized as files. `03_skilled` fixed that and is the current real submission (0.812 public / 0.782 private, ties `01_baseline`'s score with a materially more capable pipeline underneath it — every improvement since has landed inside noise).

`04_reflective` is a live experiment layered on top of `03_skilled`'s pipeline: after the full zoo runs, it honest-split-diagnoses whether the winning model has a statistically significant weak segment, and if so, the orchestrator can apply a recipe that targets exactly that segment and re-runs. Measured offline at a small but real +0.0003 mean CV improvement on the 12/16 datasets where a segment is found. Not promoted to `03_skilled` — kept as a separate submission until the with-recipe CV comparison's known label-leakage bias is fixed and measured.

## Error Debug

- `gemini-2.5-*` supports only one tool; `deepseek-r1-0528` supports none. Both fail instantly.
- Newer Anthropic models reject `temperature` in `generate_content_config`.
- Skill directory names must be **lowercase kebab-case** (`tabular-automl`, not `tabular_automl`).
- Any `{python_identifier}` in a prompt is treated as a state-injection variable and errors if undefined.
- No symlinks, no `../` anywhere in the archive.
- The session ends the instant the agent replies without a tool call.
- A hex `row_id` can coincidentally parse as scientific notation with a huge exponent, and numpy's float parser segfaults on the overflow instead of raising -- force id columns to string dtype at CSV read time, everywhere a CSV gets read (this bit both `run_automl.py` and the harness's own loading code separately).
- LightGBM/HistGradientBoosting default to using every available core; on this dev machine that measured a ~900x slowdown (an OpenMP thread-pool pathology, not a per-dataset cost). Pin `OMP_NUM_THREADS` and each model's own thread count explicitly.
