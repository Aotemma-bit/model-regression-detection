# Model Regression Detection

A small FastAPI service that classifies customer-support prompts with an OpenAI chat model, scores the run, and compares it with a saved baseline. Results stay in a local SQLite file. The API key is read from the environment and is not returned by the API or written into the database.

Categories are `billing`, `technical`, `account`, and `general`.

## Layout

| Path | Purpose |
| --- | --- |
| `app/main.py` | HTTP API |
| `app/evaluator.py` | Prompt, model call, response parsing |
| `app/metrics.py` | Accuracy, macro precision, recall, F1, regression rules |
| `app/storage.py` | SQLite runs, case results, and the baseline flag |
| `app/dataset.py` | JSONL loader |
| `data/golden_dataset.jsonl` | 24 labeled classification cases |
| `tests/` | Metrics, evaluator, and API tests with a fake model client |

## Setup

Python 3.12 is expected. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`. Leave the key out of shell history, tickets, and git. `.env` is gitignored. A placeholder value is treated as missing, so the service will not send traffic until a real key is present.

`DATABASE_PATH` and `DATASET_PATH` may be relative to this directory. The default database is `data/evaluations.sqlite3`.

## Run the tests

Tests do not need an API key and do not call OpenAI. The evaluator and API tests inject a scripted client.

```bash
source .venv/bin/activate
pytest
```

## Start the API

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

The interactive docs are at `http://127.0.0.1:8000/docs`.

### Endpoints

`GET /health` reports the process, the configured model name, and whether a real API key is set. It does not return the key.

`POST /evaluate` reads the golden dataset when the body is `{}`. Set `save_as_baseline` to promote that run. Later runs are compared with the previous baseline before the new run replaces it.

```bash
curl -s -X POST http://127.0.0.1:8000/evaluate \
  -H 'content-type: application/json' \
  -d '{"save_as_baseline": true}'
```

You can evaluate a subset with `case_ids`, or send inline cases instead of the file:

```bash
curl -s -X POST http://127.0.0.1:8000/evaluate \
  -H 'content-type: application/json' \
  -d '{
    "save_as_baseline": false,
    "cases": [
      {
        "id": "bill-001",
        "input": "I was charged twice for the same invoice.",
        "expected_category": "billing",
        "description": "Duplicate charge belongs in billing."
      }
    ]
  }'
```

`GET /results` returns the latest run. Pass `?run_id=` to read an older one.

`GET /metrics` recomputes scores for the latest run, or `?run_id=`, and compares them with the saved baseline.

A missing key returns 503. A bad dataset or invalid case returns 400. If every model call fails, nothing is stored and the API returns 502. There is no stored run until at least one case succeeds, so `GET /metrics` and `GET /results` return 404 before the first successful evaluation.

## Scoring

Accuracy is correct predictions divided by the number of cases. Precision, recall, and F1 are macro averages over the categories that appear in the expected labels or the predictions. A reply that does not contain one of the four categories is incorrect.

A metric regresses when `baseline - current` is greater than `REGRESSION_DROP_THRESHOLD` (default `0`, so any drop counts). A case regresses when the baseline prediction was correct and the current prediction is not. Either signal sets `regression.regressed` to true.

Logs go to stdout. Model failures are logged with the case id and a redacted error string.
