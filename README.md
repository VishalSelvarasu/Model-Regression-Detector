# model-regression-detector

A CI-integrated regression test suite for an LLM classification pipeline —
built to catch prompt and model regressions automatically on every pull
request, the same way you'd unit-test any other piece of production code.

**Domain:** classifying free-text industrial/embedded fault reports into
`sensor_fault`, `communication_error`, `mechanical_fault`, or `nominal`.
Chosen to mirror real fault-analysis and industrial-automation work rather
than a generic customer-support-ticket demo.

**Cost: $0.** Every LLM call — both the classifier itself and the judge
model used for semantic scoring — runs on [Groq's](https://console.groq.com/keys)
free-tier API. No OpenAI account, no local model downloads, nothing to install
beyond a handful of small Python packages.

## Why this exists

Golden-dataset regression testing, drift detection over time, and a CI gate
are the same tools MLOps/LLMOps teams use to keep prompt and model changes
from silently degrading production quality. This project is a small, real
implementation of that pattern — not a tutorial clone — meant to demonstrate
applied testing/CI discipline around an LLM component, which is a directly
transferable skill regardless of the specific domain.

## How it works

```
golden dataset (data/golden_dataset.json)
        │
        ▼
tests/test_regression.py  ──▶  category: exact string match (fast, free, deterministic)
   (DeepEval, per-PR gate)  ──▶  summary: GEval judged by Groq (semantic, still free)
        │
        ▼
scripts/run_eval_and_record.py  ──▶  data/run_history.json  ──▶  src/drift.py
   (pass-rate recorder)              (persisted history)         (rolling-window comparison)
```

Two things worth calling out about how this differs from a naive first pass:

1. **Category uses plain string comparison, not an LLM judge.** With only
   four possible categories, an LLM-as-judge metric is slower, costs a call,
   and is *less* deterministic than `==`. GEval is reserved for `summary`,
   the one field that's free text and genuinely needs semantic judgment.
2. **Run history persistence is real, not aspirational.** `check_drift()`
   is only useful if something actually calls `save_run()` after every
   official run. `scripts/run_eval_and_record.py` is that missing piece,
   and the CI workflow calls it on every merge to `main` and commits the
   updated `data/run_history.json` back automatically.

## Repo structure

```
model-regression-detector/
├── prompts/
│   ├── v1.yaml              # baseline, zero-shot
│   └── v2.yaml               # + two few-shot examples, for comparing versions
├── data/
│   ├── golden_dataset.json   # 15 hand-written test cases (expand this!)
│   └── run_history.json      # pass-rate history, written by CI, read by drift.py
├── src/
│   ├── feature.py            # the classifier itself (Groq-backed)
│   ├── groq_judge.py         # routes DeepEval's GEval judge through Groq too
│   ├── drift.py               # rolling-window drift comparison
│   ├── report.py              # markdown summary renderer
│   └── alerting.py            # optional Slack alert, safe no-op if unconfigured
├── scripts/
│   └── run_eval_and_record.py # runs golden set, saves pass rate, checks drift
├── tests/
│   └── test_regression.py     # the actual pytest/DeepEval suite CI runs
├── .github/workflows/eval.yml # gate (PRs) + record-history (push to main)
├── .env.example
└── requirements.txt
```

## Setup (local)

1. **Get a free Groq API key:** https://console.groq.com/keys — no credit
   card required.

2. **Install dependencies** (a virtualenv is recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure your key:**
   ```bash
   cp .env.example .env
   # then edit .env and paste your GROQ_API_KEY
   ```

4. **Run the test suite:**
   ```bash
   deepeval test run tests/test_regression.py -v
   ```
   First run may prompt you about creating a free Confident AI account for
   a hosted dashboard — that's optional, DeepEval, skip it if you like; the
   tests still run fully locally either way.

5. **Run the pass-rate recorder** (this is what actually builds up drift
   history over time):
   ```bash
   python scripts/run_eval_and_record.py           # records a real run
   python scripts/run_eval_and_record.py --dry-run  # preview only
   ```

## Setup (CI)

1. Push this repo to GitHub.
2. Add one repo secret: **Settings → Secrets and variables → Actions →
   New repository secret** → name it `GROQ_API_KEY`.
3. Make sure **Settings → Actions → General → Workflow permissions** is
   set to "Read and write permissions" — the `record-history` job needs
   this to commit `run_history.json` back to `main`.
4. Open a PR that touches `prompts/**` or `src/**` and watch the `gate`
   job run. Merge it and watch `record-history` run and push an updated
   `data/run_history.json`.

No other secrets are required. Slack alerting is optional and commented
out in `.github/workflows/eval.yml` — see `src/alerting.py` if you want to
wire it up later.

## Testing a prompt change (this is the actual point of the project)

```bash
# Baseline
python scripts/run_eval_and_record.py --version v1 --dry-run

# Try the few-shot variant
python scripts/run_eval_and_record.py --version v2 --dry-run
```

Compare the two pass rates. In CI, `PROMPT_VERSION` is set as an env var
in the workflow — bump it to test a different prompt on a given PR without
touching any code.

## Roadmap / next steps

- **Expand the golden dataset** from 15 to 60–80 cases. 45 draft
  candidates are staged in `data/golden_dataset_candidates.json` — review
  each label (edge cases especially), correct anything you disagree with,
  then merge with `python scripts/merge_candidates.py --ids case_016,...`
  (or `--all` once fully reviewed). An unreviewed machine-drafted dataset
  is not a golden dataset — the review IS the work.
- **Track summary-quality pass rate over time too**, not just category —
  right now only category feeds `run_history.json`.
- Once there's a few weeks of real history, put an actual drift chart
  (pass rate over time) in this README with real numbers — that's the
  difference between "I followed a tutorial" and "I built something."

## Design decisions, for anyone reading this as a portfolio piece

- **Groq instead of OpenAI**, for both the classifier and the DeepEval
  judge model (`src/groq_judge.py`) — GEval defaults to OpenAI otherwise,
  which would quietly require a paid API key.
- **String match for category, GEval only for summary** — don't reach for
  an LLM judge where a deterministic check is strictly better.
- **`gate` vs `record-history` as separate CI jobs** — an unmerged PR
  branch should never be able to corrupt the drift baseline that future
  PRs get compared against.
- **GitHub's built-in `$GITHUB_STEP_SUMMARY`** instead of a PR-comment
  bot — zero extra permissions, zero extra GitHub Action dependency,
  renders markdown natively in the Actions tab.
