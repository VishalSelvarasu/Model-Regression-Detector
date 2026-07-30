# model-regression-detector

A CI-integrated regression test suite for an LLM classification pipeline,
built to catch prompt and model regressions automatically on every pull
request, the same way you would unit-test any other production component.

**Domain:** classifying free-text industrial and embedded-system fault reports
into `sensor_fault`, `communication_error`, `mechanical_fault`, or `nominal`.

The project is designed to run on Groq's free plan. Both the classifier and
the model used for advisory semantic scoring use the Groq API, so no OpenAI
account or local model download is required.

## How it works

1. `data/golden_dataset.json` provides the reviewed inputs and expected
   classifications.

2. `tests/test_regression.py` checks the predicted category with an exact
   string comparison. This deterministic category test is the blocking
   per-pull-request CI gate.

3. The test suite also evaluates generated summaries with GEval through the
   Groq-backed judge. Summary scoring is advisory and does not block a merge.

4. `scripts/run_eval_and_record.py` runs the golden dataset and writes the
   resulting pass rate to `data/run_history.json`.

5. `src/drift.py` compares recent results with the persisted historical
   baseline to detect performance changes over time.

## Findings

Prompt iteration showed why aggregate accuracy is not enough to understand a
model change. `v1` scored 93.3% on the original 15-case dataset, with
`case_013` - "slight jitter in encoder feedback, probably nothing" -
incorrectly classified as `nominal`. `v2` added generic few-shot examples but
produced the same result and the same 93.3% score. The examples were too broad
to address the specific failure.

`v3` added an instruction to ignore reporter hedging when the report still
describes a fault. That changed the prediction for `case_013` from `nominal`
to `mechanical_fault`, but the expected category was `sensor_fault`, so the
case still failed and the total score remained 93.3%. No other case regressed.
The instruction moved the failure without fixing it, demonstrating that an
unchanged score can conceal a meaningful change in model behaviour.

`v4` extended the `sensor_fault` definition to explicitly include encoders and
probes. That resolved `case_013`, producing 100% accuracy on the original
15-case dataset. After the golden dataset was expanded to 60 cases, `v4`
scored 98.3%.

GEval was too inconsistent to remain a blocking check. It failed `case_002`
because the generated summary paraphrased "twice this morning" as "brief
outages", while passing another case whose summary introduced detail that was
not present in the source report. Because the judge penalized an acceptable
paraphrase while overlooking a genuine fabrication, summary scoring was
demoted from a merge gate to an advisory diagnostic.

The pass-rate recorder also contained a silent-failure path. It caught every
exception raised during evaluation and still exited with status code `0`.
Consequently, CI reported success after all 15 API calls failed and then
committed a `0.0` pass rate into the drift baseline. The recorder now counts
evaluation errors and calls `sys.exit(1)` when any occur, preventing failed
runs from appearing successful or contaminating the historical data.

`case_037` remains unresolved. Its report describes an issue that has already
been resolved, but the classifier predicts `mechanical_fault` rather than
`nominal`. That behaviour conflicts with the `v3` instruction: telling the
model to discount reporter hedging also pushes it away from `nominal` when the
reporter is describing recovery rather than uncertainty. The case is still
open because fixing it requires distinguishing a resolved fault from hedged
language without reintroducing the original `case_013` failure.

## Repo structure

```text
model-regression-detector/
├── prompts/
│   ├── v1.yaml                     # baseline prompt
│   ├── v2.yaml                     # prompt iteration
│   ├── v3.yaml                     # prompt iteration
│   └── v4.yaml                     # prompt iteration
├── data/
│   ├── golden_dataset.json         # 60 reviewed golden test cases
│   ├── golden_dataset_candidates.json
│   │                               # staging area for proposed dataset additions
│   └── run_history.json            # pass-rate history written by CI
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
│   └── test_regression.py          # category and advisory summary evaluations
├── .github/
│   └── workflows/
│       └── eval.yml                # PR gate and main-branch history recording
├── .env.example
└── requirements.txt
```

## Setup (local)

1. **Get a Groq API key:**

   Create an API key at https://console.groq.com/keys.

2. **Install dependencies:**

   A virtual environment is recommended.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure the API key:**

   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY.
   ```

4. **Run the blocking category tests:**

   ```bash
   pytest tests/test_regression.py -v -k test_category
   ```

   These tests do not require a Confident AI account. If you run the DeepEval
   summary evaluation separately and receive a prompt to create an account for
   the hosted dashboard, that account is optional. Skip the prompt to keep the
   evaluation local.

5. **Run the pass-rate recorder:**

   This command builds the historical data used for drift detection.

   ```bash
   python scripts/run_eval_and_record.py            # record a real run
   python scripts/run_eval_and_record.py --dry-run  # preview without saving
   ```

## Setup (CI)

1. Push the repository to GitHub.

2. Add the `GROQ_API_KEY` repository secret under:

   **Settings → Secrets and variables → Actions → New repository secret**

3. Under **Settings → Actions → General → Workflow permissions**, enable
   **Read and write permissions**. The history-recording job requires write
   access to commit updates to `data/run_history.json`.

4. Open a pull request that changes any of the following paths:

   ```text
   prompts/**
   src/**
   scripts/**
   tests/**
   data/golden_dataset.json
   ```

   The `gate` job will run the deterministic category tests. After the pull
   request is merged, the `record-history` job will run and commit the updated
   `data/run_history.json` to `main`.

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
```

Compare the category pass rates and inspect the individual cases whose outcomes
changed between versions. Treat the summary evaluation as diagnostic evidence,
not as a blocking result.

In CI, the prompt version is selected through the `PROMPT_VERSION` environment
variable in the workflow. Change that value to evaluate a different prompt
version without modifying the classifier implementation.
