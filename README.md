# Cost-Aware LLM Routing Optimization

A data-driven Operations Research web application that assigns every benchmark query to exactly one LLM. It uses uploaded evaluation data to balance actual dataset cost, measured quality, and configurable synthetic latency.

The application does not invent model prices, quality scores, latency measurements, optimization outcomes, or charts. Every statistic, baseline, allocation, solver result, and CSV export is calculated from the dataset currently loaded in the backend.

## Contents

1. [What the project solves](#what-the-project-solves)
2. [Architecture](#architecture)
3. [Installation and startup](#installation-and-startup)
4. [Dataset requirements](#dataset-requirements)
5. [What happens during upload](#what-happens-during-upload)
6. [Synthetic latency](#synthetic-latency)
7. [Optimization methods](#optimization-methods)
8. [Frontend pages](#frontend-pages)
9. [API reference](#api-reference)
10. [Exports and state](#exports-and-state)
11. [Troubleshooting](#troubleshooting)

## What The Project Solves

For each benchmark query `i` and each available model `j`, the backend creates a binary routing decision:

```text
x[i,j] = 1  model j is selected for query i
x[i,j] = 0  otherwise
```

Every query must receive exactly one model:

```text
sum(j) x[i,j] = 1
```

The application implements two Operations Research methods:

1. **Integer Linear Programming (ILP)**: minimize total query-level cost while enforcing a minimum average quality and maximum average synthetic latency.
2. **Goal Programming**: minimize weighted, normalized deviations from cost, quality, and latency targets.

It also calculates cheapest-model, highest-quality-model, and lowest-latency-model baselines and lets users rerun optimization over changing constraints.

## Architecture

```text
Next.js dashboard, port 3000
          |
          | multipart upload and JSON requests
          v
FastAPI backend, port 8000
          |
          +-- Parse JSON, CSV, XLSX
          +-- Normalize records
          +-- Validate and align query-model data
          +-- Calculate synthetic latency and statistics
          +-- Build cost, quality, and latency matrices
          +-- Run PuLP/CBC ILP or goal programming
          +-- Produce baselines, sensitivity results, and exports
```

The frontend is a control and presentation layer. The backend owns all data handling and numerical work. It keeps potentially large `raw_output` values server-side and does not send them in dashboard or normal routing responses.

## Repository Layout

```text
backend/
  main.py                 FastAPI API, data pipeline, solvers, and exports
  requirements.txt        Python packages for the API
frontend/
  app/page.tsx            Dashboard UI and API requests
  app/styles.css          Responsive dashboard styling
  package.json            Next.js scripts and dependencies
Data/                     Included MMLU-Pro benchmark JSON files
requirements.txt          Combined Python dependency list
```

The older `app/`, `analysis/`, `evaluation/`, `optimization/`, `prediction/`, and `preprocessing/` directories are the previous Streamlit prototype. The active web application is `frontend/` plus `backend/`.

## Installation And Startup

### Prerequisites

- Python 3.11 or newer. In the installer, select **Add Python to PATH**.
- Node.js 20 or newer.
- Internet access for one-time dependency installation.

Check both runtimes:

```powershell
python --version
node --version
```

### Start the backend

Open PowerShell in the project root:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

Verify that FastAPI is available:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

The response contains `"status":"ok"`. FastAPI's interactive documentation is at http://127.0.0.1:8000/docs.

### Start the frontend

In a second PowerShell:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open http://localhost:3000.

The frontend requests the API at `http://127.0.0.1:8000`. To change it:

```powershell
$env:NEXT_PUBLIC_API_URL = "http://127.0.0.1:8001"
npm run dev
```

## Dataset Requirements

Upload two or more model benchmark files where each model was evaluated on the same queries. The repository already includes a usable sample set under `Data/`; choose **Load repository sample** in the dashboard to process it.

### JSON

The JSON loader supports the supplied structure:

```json
{
  "model_name": "example-model",
  "records": [
    {
      "index": 1,
      "origin_query": "Example question",
      "prompt_tokens": 273,
      "completion_tokens": 339,
      "cost": 0.005904,
      "score": 1.0,
      "prediction": "B",
      "ground_truth": "B",
      "raw_output": "..."
    }
  ]
}
```

If a record lacks its own model value, the backend propagates the top-level `model_name` to it. Record-level `model` or `model_name` fields take precedence.

### CSV and XLSX

The normalized minimum schema is:

```text
query_id, query, model, quality, prompt_tokens, completion_tokens, cost
```

The parser accepts common source aliases:

| Internal field | Accepted source fields |
| --- | --- |
| `query_id` | `query_id`, `index` |
| `query` | `origin_query`, `query`, `question` |
| `model` | `model`, `model_name` |
| `quality` | `score`, `quality`, `performance` |
| `prompt_tokens` | `prompt_tokens` |
| `completion_tokens` | `completion_tokens` |
| `cost` | `cost` |

Optional fields are `prediction`, `ground_truth`, `raw_output`, and `latency`. Uploaded latency does not replace the project synthetic-latency model.

## What Happens During Upload

1. The browser creates a multipart request containing the selected files.
2. It sends `POST /upload` to FastAPI on port 8000.
3. The backend selects a parser:
   - JSON: Python `json`
   - CSV: pandas `read_csv`
   - XLSX: pandas `read_excel` with openpyxl
4. The parser transforms every source row into this internal representation:

```text
query_id, query, model, quality, prompt_tokens, completion_tokens,
cost, latency, prediction, ground_truth, raw_output
```

5. The validator checks:
   - empty or malformed files;
   - missing query, query ID, model, quality, cost, or token fields;
   - non-numeric values;
   - negative costs or token counts;
   - quality outside the `0..1` range;
   - duplicate query-model combinations;
   - one query ID mapping to conflicting query text.
6. It calculates expected combinations as:

```text
number of unique queries x number of detected models
```

7. Missing combinations are reported as warnings. Optimization uses only complete queries that have one record for every detected model, so a missing result cannot give a model an artificial advantage.
8. If validation passes, the normalized dataframe is stored in backend memory. Previous ILP, goal-programming, baseline, and sensitivity results are cleared because they belong to an older dataset.

The current alignment implementation requires a stable `query_id` or `index` for each record. Query text is normalized and checked for consistency, but it is not used to silently join unrelated records.

## Synthetic Latency

Benchmark summaries do not provide reliable per-query API response latency. The application never interprets a top-level benchmark `time_taken` value as per-query latency, and it never calls the number actual API latency.

It calculates **Synthetic Latency** for every query-model pair:

```text
T[i,j] = T0 + alpha * prompt_tokens[i,j] + beta * completion_tokens[i,j]
```

Dashboard defaults:

```text
T0    = 0.20
alpha = 0.001
beta  = 0.002
```

Use the Latency configuration page to change the parameters. This recalculates synthetic-latency statistics and matrices. Solvers are only rerun when the user presses their run button.

## Backend Matrices

After validation, FastAPI makes three query-by-model matrices from complete records:

```text
C[i,j]  record-level cost from the uploaded dataset
Q[i,j]  record-level quality from the uploaded dataset
T[i,j]  calculated synthetic latency
```

Rows are queries and columns are models. These are the only inputs used by the optimization methods.

## Optimization Methods

### ILP

The backend uses PuLP with the CBC solver.

Objective:

```text
minimize sum(i,j) C[i,j] * x[i,j]
```

Constraints:

```text
sum(j) x[i,j] = 1                         for every query i
sum(i,j) Q[i,j] * x[i,j] >= N * Q_min
sum(i,j) T[i,j] * x[i,j] <= N * T_max
x[i,j] is binary
```

`Q_min` is the requested minimum average quality, and `T_max` is the requested maximum average synthetic latency.

If the problem is infeasible, FastAPI does not relax constraints or choose an arbitrary fallback. It reports:

- highest achievable average quality, assigning each query to its highest-quality option;
- lowest achievable average latency, assigning each query to its fastest option;
- cheapest possible routing cost, assigning each query to its cheapest option.

### Goal Programming

Goal Programming preserves the exactly-one-model constraint and introduces deviation variables:

```text
cost    + d_cost_minus    - d_cost_plus    = target_cost
quality + d_quality_minus - d_quality_plus = N * target_quality
latency + d_latency_minus - d_latency_plus = N * target_latency
```

Only undesirable deviations are penalized:

```text
cost_weight    * d_cost_plus     / target_cost
quality_weight * d_quality_minus / (N * target_quality)
latency_weight * d_latency_plus  / (N * target_latency)
```

That means the model does not penalize cost below target, quality above target, or latency below target. The three weights must sum to exactly `1.0`. Invalid weights cause an explicit API validation error; they are never silently normalized.

### Baselines

Baselines are calculated from the loaded data, not fixed model names:

1. **Cheapest Model**: lowest average query cost, routed to every query.
2. **Highest Quality Model**: highest average quality, routed to every query.
3. **Lowest Latency Model**: lowest average synthetic latency, routed to every query.

Each result includes total cost, average quality, average latency, allocation, and constraint status when ILP limits are supplied.

### Sensitivity Analysis

Each sensitivity point performs a new solver run and records:

```text
parameter value, solver status, total cost, average quality,
average synthetic latency, model allocation
```

Supported analyses:

- ILP minimum-quality sweep;
- ILP maximum-synthetic-latency sweep;
- cost-heavy, quality-heavy, latency-heavy, and balanced goal-weight scenarios.

Infeasible points remain in the response instead of being discarded.

## Frontend Pages

| Page | Function |
| --- | --- |
| Dataset upload | Upload model files, view validation, or load included samples. |
| Dataset overview | KPI values and charts for quality, cost, tokens, and synthetic latency. |
| Statistical analysis | Model statistics plus cost-quality-latency scatter plots. |
| Latency configuration | Set `T0`, `alpha`, and `beta`. |
| ILP optimization | Configure constraints and run the CBC solver. |
| Goal programming | Configure targets and weights and run goal programming. |
| Baseline comparison | Calculate and compare calculated baselines and saved solver results. |
| Sensitivity analysis | Run parameter and weight scenarios. |
| Query routing | Search saved ILP routes; the API paginates records. |
| Export | Download available calculated CSV data. |

## API Reference

| Method | Endpoint | Function |
| --- | --- | --- |
| `GET` | `/health` | Check whether FastAPI is running. |
| `POST` | `/upload` | Upload JSON, CSV, or XLSX as `files`. |
| `POST` | `/dataset/load-sample` | Read repository JSON files from `Data/`. |
| `GET` | `/dataset/summary` | Return dataset KPI values. |
| `GET` | `/dataset/models` | Return model-level statistics. |
| `GET` | `/dataset/statistics` | Return statistics and chart data. |
| `POST` | `/latency/configure` | Update synthetic-latency parameters. |
| `POST` | `/optimize/ilp` | Run constrained ILP. |
| `POST` | `/optimize/goal` | Run goal programming. |
| `POST` | `/baseline/cheapest` | Calculate cheapest-model baseline. |
| `POST` | `/baseline/quality` | Calculate highest-quality baseline. |
| `POST` | `/baseline/latency` | Calculate lowest-latency baseline. |
| `POST` | `/sensitivity` | Run sensitivity analysis. |
| `GET` | `/results` | Return saved results. |
| `GET` | `/results/routing` | Return searchable, paginated route rows. |
| `GET` | `/results/comparison` | Return comparison-ready result rows. |
| `GET` | `/export` | Download routing, comparison, sensitivity, or summary CSV. |

## Exports And State

The export endpoint creates CSV files from the current in-memory backend state:

- `routing`: selected routes for a saved strategy;
- `comparison`: calculated baseline and solver metrics;
- `sensitivity`: saved scenario results;
- `summary`: current dataset KPI values.

Run an optimization or analysis before exporting it. No placeholder rows are created. The backend state is intentionally in memory for this project, so restarting the FastAPI process clears the loaded dataset and all saved results.

## Development Verification

Build the frontend:

```powershell
Set-Location frontend
npm run build
```

Check browser availability at http://localhost:3000 and API availability at http://127.0.0.1:8000/health.

## Troubleshooting

### The dashboard says backend is unreachable or Failed to fetch

No FastAPI process is listening on port 8000. Start it:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Then check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

### Python is not recognized

Install Python 3.11+ and select **Add Python to PATH**. Close and reopen PowerShell after installation.

### Upload validation fails

Read the returned validation message. Common causes are a missing model, query ID/text, quality, cost, token field, invalid numeric value, duplicate query-model combination, or a quality value outside `0..1`.

### ILP is infeasible

The requested average quality and latency limits conflict for the dataset. Use the returned diagnostics to lower the minimum quality or increase the allowed synthetic latency. The application intentionally does not change constraints for the user.

### Goal Programming rejects the request

Ensure cost, quality, and latency weights add exactly to `1.00`.

## Limitations

- Cost and quality apply only to the uploaded benchmark, not universally to a model.
- Synthetic latency is a transparent token-based estimate, not live API latency.
- Optimization uses only queries evaluated by every detected model.
- For production deployment, replace in-memory state with a database, add authentication and per-user dataset ownership, and move large sensitivity jobs to a queue with solver resource limits.
