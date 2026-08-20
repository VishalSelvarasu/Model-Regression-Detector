# model-regression-detector

A CI-integrated regression test suite for an LLM classification pipeline,
built to catch prompt and model regressions automatically on pull requests that
touch gated paths, the same way you would unit-test any other production
component.

**Domain:** classifying free-text industrial and embedded-system fault reports
into `sensor_fault`, `communication_error`, `mechanical_fault`, or `nominal`.

The project is designed to run on Groq's free plan. The classifier currently
uses `openai/gpt-oss-20b` through the Groq API. The project originally used
`llama-3.1-8b-instant`, but that model was decommissioned mid-project and began
returning 404 responses. The advisory semantic scorer also uses the Groq API,
so no OpenAI account or local model download is required.

## How it works

`data/golden_dataset.json` contains 60 hand-authored synthetic fault reports
with expected classifications that were manually reviewed for this project.
They are not drawn from a real production logging system.

`tests/test_regression.py` checks predicted categories with exact string
comparison. The 57 cases with hard, unambiguous expected labels form the
blocking per-pull-request CI gate. Three cases with genuinely ambiguous
ground truth are advisory: an uncertain label should not block a merge as
though it were an objective regression.

The test suite also evaluates generated summaries with GEval through the
Groq-backed judge. Summary scoring is advisory and does not block a merge.

`scripts/run_eval_and_record.py` runs the golden dataset and writes the
resulting pass rate and run provenance to `data/run_history.json`.

`src/drift.py` compares two complete 7-run windows against the persisted
historical baseline. Drift evaluation therefore requires at least 14
recorded runs before a comparison can be made.

## Findings

I started with a 15-case dataset. `v1` scored 93.3%, with `case_013` -
"slight jitter in encoder feedback, probably nothing" - coming back as
`nominal`. `v2` added generic few-shot examples and changed nothing: same case,
same prediction, same 93.3%. The examples were simply too broad to help with
that failure.

For `v3`, I told the model to ignore reporter hedging when the report still
describes a fault. That did move `case_013`, but in the wrong direction:
`nominal` became `mechanical_fault` while the expected label was
`sensor_fault`. The overall score stayed at 93.3%, and no other case regressed.
That was a useful warning that an unchanged aggregate score can hide a real
behaviour change.

`v4` was the first clean fix. I expanded the `sensor_fault` definition to name
encoders and probes explicitly, which fixed `case_013` and brought the original
15 cases to 100%. After I expanded the dataset to 60 cases, `v4` scored 98.3%.

I also tried using GEval as a blocking summary check, but I stopped trusting it
for that job. It failed `case_002` because "twice this morning" was paraphrased
as "brief outages", yet it passed another summary that actually introduced
detail not present in the source. I kept the summary score, but only as an
advisory signal.

The pass-rate recorder had a more serious bug. It caught evaluation exceptions
and still exited 0, so CI could look green after every API call had failed.
Worse, that run then wrote a 0.0 pass rate into the drift history. So the fix
is simple: count evaluation errors, exit non-zero if any occurred, and do not
record the run.

`case_037` is still open. The report describes a fault that has already been
resolved, but the classifier returns `mechanical_fault` instead of `nominal`.
The hedging rule from `v3` helps with reports that downplay active faults, but it
also makes this resolved-fault case harder. I'm leaving it unresolved rather
than forcing a prompt change that might bring back the original `case_013`
problem.

Mid-project, `llama-3.1-8b-instant` was decommissioned and started returning
HTTP 404. Every test failed at once. That is basically the failure mode this
project is meant to catch, and it happened for real rather than as a demo.

I switched the classifier to `openai/gpt-oss-20b`. On the 60-case set it
reproduced the 98.3% result and the same `case_037` failure, which gave me some
confidence that the finding was not just an artifact of the retired model.

`v5` was a negative result. I added a resolved-fault instruction specifically
for `case_037`; it still failed, and `case_060` flipped at the same time. I kept
`v5` in the repo because failed prompt changes are evidence too.

The free-tier quota is also shaping the experiment. Groq's 200,000-token daily
limit works out to roughly six complete 60-case runs at the current workload.
That puts a real ceiling on CI frequency and makes repeated sampling expensive,
so I have not yet done enough repeat runs to claim determinism on
`openai/gpt-oss-20b`.

## Limitations

`case_037` is still open. The current prompt does not reliably distinguish a
resolved fault from language that merely hedges or downplays an active fault,
so the known failure remains part of the test evidence rather than being
treated as solved.

Determinism has not been verified on `openai/gpt-oss-20b`. The category
assertion itself is an exact-match test, but the classifier depends on a hosted
LLM call; repeat runs are therefore needed before claiming that identical
inputs always produce identical outputs with the current model.

The GitHub required-check configuration interacts poorly with the workflow's
path filter. Pull requests that change only files outside the gated paths do
not start the workflow, so the required check is never produced and those PRs
cannot merge without an explicit bypass. The path filter reduces unnecessary
API use, but the branch-protection configuration needs to account for that
tradeoff.

## Repo structure

```text
model-regression-detector/
├── prompts/
│   ├── v1.yaml                     # baseline prompt
│   ├── v2.yaml                     # prompt iteration
│   ├── v3.yaml                     # prompt iteration
│   ├── v4.yaml                     # prompt iteration
│   └── v5.yaml                     # resolved-fault experiment
├── data/
│   ├── golden_dataset.json         # 60 reviewed hand-authored synthetic cases
│   ├── golden_dataset_candidates.json
│   │                               # staging area for proposed dataset additions
│   └── run_history.json            # pass-rate and provenance history written by CI
├── src/
│   ├── feature.py                  # Groq-backed classifier
│   ├── groq_judge.py               # routes GEval judging through Groq
│   ├── drift.py                    # rolling-window drift comparison
│   ├── report.py                   # markdown summary renderer
│   └── alerting.py                 # optional Slack alerting
├── scripts/
│   ├── run_eval_and_record.py      # runs the golden set and records results
│   └── merge_candidates.py         # merges reviewed candidate cases
├── tests/
│   ├── test_regression.py          # blocking and advisory regression evaluations
│   └── test_drift.py               # drift/history infrastructure unit tests
├── .github/
│   └── workflows/
│       └── eval.yml                # PR gate and main-branch history recording
├── .env.example
├── LICENSE
└── requirements.txt
```

## Setup (local)

**Get a Groq API key:**

Create an API key at https://console.groq.com/keys.

**Install dependencies:**

A virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Configure the API key:**

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY.
```

**Run the blocking category regression tests:**

```bash
pytest tests/test_regression.py -v -k "test_category_classification"
```

This command runs only the 57 hard-label cases used by the blocking CI gate.
The three ambiguous cases live in the separate `test_category_advisory` test
because uncertain ground truth should not prevent a merge. These tests do not
require a Confident AI account. If you run the
DeepEval summary evaluation separately and receive a prompt to create an
account for the hosted dashboard, that account is optional. Skip the prompt
to keep the evaluation local.

**Run the drift infrastructure tests:**

```bash
pytest tests/test_drift.py -v
```

These tests use temporary history files and do not make API calls.

**Run the pass-rate recorder:**

This command builds the historical data used for drift detection. A drift
comparison is available after at least 14 recorded runs.

```bash
python scripts/run_eval_and_record.py            # record a real run
python scripts/run_eval_and_record.py --dry-run  # preview without saving
```

## Setup (CI)

Push the repository to GitHub.

Add the `GROQ_API_KEY` repository secret under:

**Settings → Secrets and variables → Actions → New repository secret**

Under **Settings → Actions → General → Workflow permissions**, enable
**Read and write permissions**. The history-recording job requires write
access to commit updates to `data/run_history.json`.

Open a pull request that changes any of the following paths:

```text
prompts/**
src/**
scripts/**
tests/**
data/golden_dataset.json
```

The `gate` job runs exact-match category checks for the 57 hard-label cases.
The three ambiguous cases remain advisory because ambiguous ground truth
should not be treated as a merge-blocking regression. After the pull request
is merged, the `record-history` job runs and commits the updated
`data/run_history.json` to `main`.

Because the workflow is path-filtered while the `gate` check is required,
pull requests that touch only files outside these paths do not produce the
required check and cannot merge without a bypass.

No additional secrets are required. Slack alerting is optional and disabled by
default in `.github/workflows/eval.yml`; its implementation is in
`src/alerting.py`.

## Testing a prompt change

Run each prompt version against the same dataset so the results remain
comparable:

```bash
python scripts/run_eval_and_record.py --version v1 --dry-run
python scripts/run_eval_and_record.py --version v2 --dry-run
python scripts/run_eval_and_record.py --version v3 --dry-run
python scripts/run_eval_and_record.py --version v4 --dry-run
python scripts/run_eval_and_record.py --version v5 --dry-run
```

Compare the category pass rates and inspect the individual cases whose outcomes
changed between versions. Treat the three ambiguous category cases and the
summary evaluation as diagnostic evidence, not as blocking results.

In CI, the prompt version is selected through the `PROMPT_VERSION` environment
variable in the workflow. Change that value to evaluate a different prompt
version without modifying the classifier implementation.