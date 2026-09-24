from __future__ import annotations

import io
import json
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pulp
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ["query_id", "query", "model", "quality", "prompt_tokens", "completion_tokens", "cost"]
NUMERIC_COLUMNS = ["quality", "prompt_tokens", "completion_tokens", "cost"]
MODEL_FIELDS = ("model", "model_name")
QUERY_FIELDS = ("origin_query", "query", "question")
QUALITY_FIELDS = ("score", "quality", "performance")


class LatencyConfig(BaseModel):
    fixed_overhead: float = Field(0.20, ge=0)
    prompt_coefficient: float = Field(0.001, ge=0)
    completion_coefficient: float = Field(0.002, ge=0)


class IlpRequest(BaseModel):
    minimum_quality: float = Field(..., ge=0)
    maximum_average_latency: float = Field(..., ge=0)


class GoalRequest(BaseModel):
    target_cost: float = Field(..., ge=0)
    target_quality: float = Field(..., ge=0)
    target_latency: float = Field(..., ge=0)
    cost_weight: float = Field(..., ge=0)
    quality_weight: float = Field(..., ge=0)
    latency_weight: float = Field(..., ge=0)

    @model_validator(mode="after")
    def weights_sum_to_one(self):
        if not math.isclose(self.cost_weight + self.quality_weight + self.latency_weight, 1.0, abs_tol=1e-6):
            raise ValueError("Cost, quality, and latency weights must sum to 1.")
        return self


class SensitivityRequest(BaseModel):
    parameter: Literal["minimum_quality", "maximum_latency", "goal_weights"]
    start: float | None = None
    end: float | None = None
    step: float | None = None
    ilp: IlpRequest | None = None
    goal: GoalRequest | None = None


@dataclass
class DatasetStore:
    frame: pd.DataFrame | None = None
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    sensitivity: list[dict[str, Any]] = field(default_factory=list)


store = DatasetStore()
app = FastAPI(title="Cost-Aware LLM Routing Optimization", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def as_json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def records_to_json(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: as_json(value) for key, value in row.items()} for row in records]


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def normalize_json(payload: Any, source: str) -> list[dict[str, Any]]:
    objects = payload if isinstance(payload, list) else [payload]
    normalized: list[dict[str, Any]] = []
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError(f"{source}: JSON must contain an object or array of objects.")
        top_model = first_present(item, MODEL_FIELDS)
        rows = item.get("records", [item])
        if not isinstance(rows, list):
            raise ValueError(f"{source}: records must be an array.")
        for position, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            normalized.append(normalize_record(row, source, top_model, position))
    return normalized


def normalize_record(row: dict[str, Any], source: str, inherited_model: Any = None, position: int = 1) -> dict[str, Any]:
    query = first_present(row, QUERY_FIELDS)
    query_id = row.get("query_id", row.get("index"))
    return {
        "query_id": str(query_id) if query_id is not None and str(query_id).strip() else None,
        "query": str(query).strip() if query is not None else None,
        "model": first_present(row, MODEL_FIELDS) or inherited_model,
        "quality": first_present(row, QUALITY_FIELDS),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "cost": row.get("cost"),
        "latency": row.get("latency"),
        "prediction": row.get("prediction"),
        "ground_truth": row.get("ground_truth"),
        "raw_output": row.get("raw_output"),
        "source": source,
        "source_position": position,
    }


def parse_upload(name: str, content: bytes) -> list[dict[str, Any]]:
    suffix = Path(name).suffix.lower()
    if not content:
        raise ValueError(f"{name}: file is empty.")
    if suffix == ".json":
        try:
            return normalize_json(json.loads(content.decode("utf-8-sig")), name)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}: malformed JSON ({exc.msg}).") from exc
    try:
        raw = pd.read_csv(io.BytesIO(content)) if suffix == ".csv" else pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"{name}: unable to read file ({exc}).") from exc
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise ValueError(f"{name}: unsupported type. Upload JSON, CSV, or XLSX.")
    return [normalize_record(row, name, position=index + 1) for index, row in enumerate(raw.to_dict("records"))]


def validate_and_align(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not rows:
        raise ValueError("No records were found in the uploaded files.")
    frame = pd.DataFrame(rows)
    errors: list[str] = []
    warnings: list[str] = []
    for col in REQUIRED_COLUMNS:
        missing = frame[col].isna() | frame[col].astype(str).str.strip().eq("")
        if missing.any():
            errors.append(f"{missing.sum()} records are missing {col.replace('_', ' ')}.")
    if errors:
        return frame, {"valid": False, "errors": errors, "warnings": warnings}
    frame["query_id"] = frame["query_id"].astype(str)
    frame["query"] = frame["query"].astype(str).str.strip()
    frame["model"] = frame["model"].astype(str).str.strip()
    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        invalid = frame[col].isna()
        if invalid.any():
            errors.append(f"{invalid.sum()} records have non-numeric {col.replace('_', ' ')}.")
    if (frame[["prompt_tokens", "completion_tokens", "cost"]] < 0).any().any():
        errors.append("Token counts and cost must not be negative.")
    if ((frame["quality"] < 0) | (frame["quality"] > 1)).any():
        errors.append("Quality values must be between 0 and 1.")
    pair_dupes = frame.duplicated(["query_id", "model"], keep=False)
    if pair_dupes.any():
        errors.append(f"{pair_dupes.sum()} duplicate query-model combinations were detected.")
    conflicting_queries = frame.groupby("query_id")["query"].nunique()
    if (conflicting_queries > 1).any():
        errors.append(f"{(conflicting_queries > 1).sum()} query IDs map to different query text values.")
    if errors:
        return frame, {"valid": False, "errors": errors, "warnings": warnings}
    frame["total_tokens"] = frame["prompt_tokens"] + frame["completion_tokens"]
    models = sorted(frame["model"].unique().tolist())
    queries = sorted(frame["query_id"].unique().tolist())
    expected = len(models) * len(queries)
    actual = len(frame)
    missing = expected - actual
    if missing:
        warnings.append(f"{missing} query-model combinations are missing. Optimization will use only complete query rows.")
    complete_ids = frame.groupby("query_id")["model"].nunique()
    complete_ids = complete_ids[complete_ids == len(models)].index
    if len(complete_ids) == 0:
        errors.append("No queries have responses from every detected model.")
    summary = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "queries": len(queries),
        "complete_queries": len(complete_ids),
        "models": models,
        "records": actual,
        "expected_combinations": expected,
        "missing_combinations": missing,
        "duplicates": int(pair_dupes.sum()),
        "latency_note": "Synthetic latency is calculated from configurable token coefficients; it is not observed API latency.",
    }
    return frame, summary


def require_dataset() -> pd.DataFrame:
    if store.frame is None:
        raise HTTPException(400, "Upload a valid dataset before using this endpoint.")
    return store.frame


def with_latency(frame: pd.DataFrame) -> pd.DataFrame:
    config = store.latency
    copied = frame.copy()
    copied["latency"] = config.fixed_overhead + config.prompt_coefficient * copied["prompt_tokens"] + config.completion_coefficient * copied["completion_tokens"]
    return copied


def complete_frame() -> pd.DataFrame:
    frame = with_latency(require_dataset())
    model_count = frame["model"].nunique()
    valid_ids = frame.groupby("query_id")["model"].nunique()
    return frame[frame["query_id"].isin(valid_ids[valid_ids == model_count].index)].copy()


def model_stats(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = frame.groupby("model", as_index=False).agg(
        records=("query_id", "count"), average_quality=("quality", "mean"), minimum_quality=("quality", "min"), maximum_quality=("quality", "max"),
        total_quality=("quality", "sum"), average_cost=("cost", "mean"), total_cost=("cost", "sum"), minimum_cost=("cost", "min"), maximum_cost=("cost", "max"),
        average_prompt_tokens=("prompt_tokens", "mean"), average_completion_tokens=("completion_tokens", "mean"), total_prompt_tokens=("prompt_tokens", "sum"),
        total_completion_tokens=("completion_tokens", "sum"), total_tokens=("total_tokens", "sum"), average_latency=("latency", "mean"),
        minimum_latency=("latency", "min"), maximum_latency=("latency", "max"),
    )
    if "prediction" in frame and "ground_truth" in frame:
        correct = frame[frame["prediction"].notna() & frame["ground_truth"].notna()].assign(correct=lambda x: x["prediction"].astype(str) == x["ground_truth"].astype(str)).groupby("model")["correct"].sum()
        result["correct_responses"] = result["model"].map(correct).fillna(0).astype(int)
    return records_to_json(result.to_dict("records"))


def routing_metrics(routing: pd.DataFrame) -> dict[str, Any]:
    total = len(routing)
    allocation = routing["model"].value_counts().sort_index()
    return {
        "total_cost": float(routing["cost"].sum()),
        "average_quality": float(routing["quality"].mean()),
        "average_latency": float(routing["latency"].mean()),
        "total_tokens": int(routing["total_tokens"].sum()),
        "model_allocation": {model: {"queries": int(count), "percentage": round(100 * count / total, 2)} for model, count in allocation.items()},
    }


def matrices(frame: pd.DataFrame) -> tuple[list[str], list[str], dict[tuple[str, str], dict[str, Any]]]:
    queries = sorted(frame["query_id"].unique().tolist())
    models = sorted(frame["model"].unique().tolist())
    lookup = {(str(row.query_id), row.model): row for row in frame.itertuples(index=False)}
    return queries, models, lookup


def solve_ilp(minimum_quality: float, maximum_latency: float) -> dict[str, Any]:
    frame = complete_frame()
    queries, models, lookup = matrices(frame)
    problem = pulp.LpProblem("cost_aware_llm_routing", pulp.LpMinimize)
    x = {(q, m): pulp.LpVariable(f"x_{uuid.uuid4().hex[:10]}", cat=pulp.LpBinary) for q in queries for m in models}
    problem += pulp.lpSum(lookup[q, m].cost * x[q, m] for q in queries for m in models)
    for q in queries:
        problem += pulp.lpSum(x[q, m] for m in models) == 1
    problem += pulp.lpSum(lookup[q, m].quality * x[q, m] for q in queries for m in models) >= len(queries) * minimum_quality
    problem += pulp.lpSum(lookup[q, m].latency * x[q, m] for q in queries for m in models) <= len(queries) * maximum_latency
    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    diagnostics = feasibility_diagnostics(frame)
    base = {"status": status, "constraints": {"minimum_quality": minimum_quality, "maximum_average_latency": maximum_latency}, "diagnostics": diagnostics}
    if status != "Optimal":
        base["message"] = "No feasible solution exists under the current constraints." if status == "Infeasible" else f"Solver finished with status: {status}."
        return base
    chosen = [lookup[q, m]._asdict() for q in queries for m in models if pulp.value(x[q, m]) > 0.5]
    routing = pd.DataFrame(chosen)
    metrics = routing_metrics(routing)
    base.update(metrics)
    base["objective_value"] = float(pulp.value(problem.objective))
    base["constraints"].update({"quality_satisfied": metrics["average_quality"] >= minimum_quality - 1e-8, "latency_satisfied": metrics["average_latency"] <= maximum_latency + 1e-8})
    base["routing"] = records_to_json(routing.drop(columns=["raw_output"], errors="ignore").to_dict("records"))
    return base


def feasibility_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    by_query = frame.groupby("query_id")
    cheapest = by_query.apply(lambda part: part.loc[part["cost"].idxmin()])
    highest_quality = by_query.apply(lambda part: part.loc[part["quality"].idxmax()])
    fastest = by_query.apply(lambda part: part.loc[part["latency"].idxmin()])
    return {"highest_achievable_quality": float(highest_quality["quality"].mean()), "lowest_achievable_latency": float(fastest["latency"].mean()), "cheapest_possible_cost": float(cheapest["cost"].sum())}


def solve_goal(request: GoalRequest) -> dict[str, Any]:
    frame = complete_frame()
    queries, models, lookup = matrices(frame)
    n = len(queries)
    problem = pulp.LpProblem("goal_programming_llm_routing", pulp.LpMinimize)
    x = {(q, m): pulp.LpVariable(f"x_{uuid.uuid4().hex[:10]}", cat=pulp.LpBinary) for q in queries for m in models}
    deviations = {name: (pulp.LpVariable(f"{name}_minus", lowBound=0), pulp.LpVariable(f"{name}_plus", lowBound=0)) for name in ("cost", "quality", "latency")}
    cost = pulp.lpSum(lookup[q, m].cost * x[q, m] for q in queries for m in models)
    quality = pulp.lpSum(lookup[q, m].quality * x[q, m] for q in queries for m in models)
    latency = pulp.lpSum(lookup[q, m].latency * x[q, m] for q in queries for m in models)
    for q in queries:
        problem += pulp.lpSum(x[q, m] for m in models) == 1
    problem += cost + deviations["cost"][0] - deviations["cost"][1] == request.target_cost
    problem += quality + deviations["quality"][0] - deviations["quality"][1] == n * request.target_quality
    problem += latency + deviations["latency"][0] - deviations["latency"][1] == n * request.target_latency
    # Normalize against nonzero targets. Only undesirable deviations are penalized.
    objective = request.cost_weight * deviations["cost"][1] / max(request.target_cost, 1e-9)
    objective += request.quality_weight * deviations["quality"][0] / max(n * request.target_quality, 1e-9)
    objective += request.latency_weight * deviations["latency"][1] / max(n * request.target_latency, 1e-9)
    problem += objective
    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    result: dict[str, Any] = {"status": status, "targets": request.model_dump(), "normalization": "Undesirable deviations are divided by their respective target totals.", "objective_value": float(pulp.value(problem.objective)) if pulp.value(problem.objective) is not None else None}
    if status != "Optimal":
        result["message"] = f"Solver finished with status: {status}."
        return result
    chosen = [lookup[q, m]._asdict() for q in queries for m in models if pulp.value(x[q, m]) > 0.5]
    routing = pd.DataFrame(chosen)
    result.update(routing_metrics(routing))
    result["deviations"] = {"cost": float(pulp.value(deviations["cost"][1])), "quality": float(pulp.value(deviations["quality"][0])), "latency": float(pulp.value(deviations["latency"][1]))}
    result["routing"] = records_to_json(routing.drop(columns=["raw_output"], errors="ignore").to_dict("records"))
    return result


def baseline(kind: str, minimum_quality: float | None = None, maximum_latency: float | None = None) -> dict[str, Any]:
    frame = complete_frame()
    aggregate = frame.groupby("model").agg(cost=("cost", "mean"), quality=("quality", "mean"), latency=("latency", "mean"))
    metric = {"cheapest": "cost", "quality": "quality", "latency": "latency"}[kind]
    chosen_model = aggregate[metric].idxmax() if kind == "quality" else aggregate[metric].idxmin()
    routing = frame[frame["model"] == chosen_model].copy()
    result = {"status": "Calculated", "strategy": {"cheapest": "Cheapest Model", "quality": "Highest Quality Model", "latency": "Lowest Latency Model"}[kind], "selected_model": chosen_model, **routing_metrics(routing)}
    if minimum_quality is not None and maximum_latency is not None:
        result["constraints"] = {"minimum_quality": minimum_quality, "maximum_average_latency": maximum_latency, "quality_satisfied": result["average_quality"] >= minimum_quality, "latency_satisfied": result["average_latency"] <= maximum_latency}
    result["routing"] = records_to_json(routing.drop(columns=["raw_output"], errors="ignore").to_dict("records"))
    return result


@app.get("/health")
def health():
    return {"status": "ok", "dataset_loaded": store.frame is not None}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    file_details = []
    for file in files:
        content = await file.read()
        try:
            parsed = parse_upload(file.filename or "upload", content)
            rows.extend(parsed)
            file_details.append({"filename": file.filename, "size": len(content), "records": len(parsed), "status": "accepted"})
        except ValueError as exc:
            errors.append(str(exc))
            file_details.append({"filename": file.filename, "size": len(content), "records": 0, "status": "rejected"})
    if errors:
        return {"validation": {"valid": False, "errors": errors, "warnings": []}, "files": file_details}
    frame, validation = validate_and_align(rows)
    if validation["valid"]:
        store.frame = frame
        store.results.clear()
        store.sensitivity.clear()
    return {"validation": validation, "files": file_details}


@app.post("/dataset/load-sample")
def load_sample():
    paths = sorted((ROOT / "Data").glob("*.json"))
    if not paths:
        raise HTTPException(404, "No sample JSON files were found in Data.")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(normalize_json(json.loads(path.read_text(encoding="utf-8")), path.name))
    frame, validation = validate_and_align(rows)
    if not validation["valid"]:
        return {"validation": validation}
    store.frame = frame
    store.results.clear()
    store.sensitivity.clear()
    return {"validation": validation}


@app.get("/dataset/summary")
def dataset_summary():
    frame = with_latency(require_dataset())
    return {"queries": int(frame["query_id"].nunique()), "models": int(frame["model"].nunique()), "records": len(frame), "total_cost": float(frame["cost"].sum()), "average_quality": float(frame["quality"].mean()), "total_tokens": int(frame["total_tokens"].sum()), "average_latency": float(frame["latency"].mean()), "latency_label": "Synthetic Latency"}


@app.get("/dataset/models")
def dataset_models():
    return {"models": model_stats(with_latency(require_dataset()))}


@app.get("/dataset/statistics")
def dataset_statistics():
    frame = with_latency(require_dataset())
    return {"models": model_stats(frame), "scatter": records_to_json(frame.groupby("model", as_index=False).agg(cost=("cost", "mean"), quality=("quality", "mean"), latency=("latency", "mean"), tokens=("total_tokens", "mean")).to_dict("records"))}


@app.post("/latency/configure")
def configure_latency(config: LatencyConfig):
    store.latency = config
    return {"config": config.model_dump(), "statistics": model_stats(with_latency(require_dataset())), "label": "Synthetic Latency"}


@app.post("/optimize/ilp")
def optimize_ilp(request: IlpRequest):
    result = solve_ilp(request.minimum_quality, request.maximum_average_latency)
    store.results["ilp"] = result
    return result


@app.post("/optimize/goal")
def optimize_goal(request: GoalRequest):
    result = solve_goal(request)
    store.results["goal"] = result
    return result


@app.post("/baseline/{kind}")
def optimize_baseline(kind: Literal["cheapest", "quality", "latency"], request: IlpRequest | None = None):
    result = baseline(kind, request.minimum_quality if request else None, request.maximum_average_latency if request else None)
    store.results[kind] = result
    return result


@app.get("/results")
def get_results():
    return store.results


@app.get("/results/comparison")
def comparison():
    comparison_rows = []
    for key, result in store.results.items():
        if result.get("status") not in {"Optimal", "Calculated"}:
            continue
        comparison_rows.append({"strategy": result.get("strategy", key.upper()), "total_cost": result["total_cost"], "average_quality": result["average_quality"], "average_latency": result["average_latency"], "allocation": result["model_allocation"], "status": result["status"]})
    return {"comparison": comparison_rows}


@app.get("/results/routing")
def routing(strategy: str = "ilp", page: int = Query(1, ge=1), page_size: int = Query(25, ge=5, le=100), search: str = "", model: str | None = None):
    result = store.results.get(strategy)
    if not result or "routing" not in result:
        raise HTTPException(404, "No saved routing exists for this strategy.")
    rows = result["routing"]
    if search:
        token = search.lower()
        rows = [row for row in rows if token in str(row.get("query", "")).lower() or token in str(row.get("query_id", "")).lower()]
    if model:
        rows = [row for row in rows if row.get("model") == model]
    start = (page - 1) * page_size
    return {"total": len(rows), "page": page, "page_size": page_size, "items": rows[start:start + page_size]}


@app.post("/sensitivity")
def sensitivity(request: SensitivityRequest):
    if request.parameter == "goal_weights":
        base = request.goal or GoalRequest(target_cost=1, target_quality=.8, target_latency=1, cost_weight=.34, quality_weight=.33, latency_weight=.33)
        scenarios = [("Cost-heavy", .7, .2, .1), ("Quality-heavy", .2, .7, .1), ("Latency-heavy", .2, .1, .7), ("Balanced", .34, .33, .33)]
        rows = []
        for label, cost_weight, quality_weight, latency_weight in scenarios:
            result = solve_goal(base.model_copy(update={"cost_weight": cost_weight, "quality_weight": quality_weight, "latency_weight": latency_weight}))
            rows.append({"parameter_value": label, "status": result["status"], "total_cost": result.get("total_cost"), "average_quality": result.get("average_quality"), "average_latency": result.get("average_latency"), "model_allocation": result.get("model_allocation", {})})
    else:
        if request.start is None or request.end is None or request.step is None or request.step <= 0:
            raise HTTPException(422, "Start, end, and a positive step are required for this sensitivity run.")
        values = np.arange(request.start, request.end + request.step / 2, request.step)
        base = request.ilp or IlpRequest(minimum_quality=.8, maximum_average_latency=2)
        rows = []
        for value in values:
            run = solve_ilp(float(value), base.maximum_average_latency) if request.parameter == "minimum_quality" else solve_ilp(base.minimum_quality, float(value))
            rows.append({"parameter_value": round(float(value), 8), "status": run["status"], "total_cost": run.get("total_cost"), "average_quality": run.get("average_quality"), "average_latency": run.get("average_latency"), "model_allocation": run.get("model_allocation", {})})
    store.sensitivity = rows
    return {"parameter": request.parameter, "runs": rows}


@app.get("/export")
def export(kind: Literal["routing", "comparison", "sensitivity", "summary"] = "routing", strategy: str = "ilp"):
    if kind == "routing":
        rows = store.results.get(strategy, {}).get("routing", [])
        filename = f"{strategy}-routing.csv"
    elif kind == "comparison":
        rows = comparison()["comparison"]
        filename = "strategy-comparison.csv"
    elif kind == "sensitivity":
        rows = store.sensitivity
        filename = "sensitivity.csv"
    else:
        rows = [dataset_summary()]
        filename = "dataset-summary.csv"
    if not rows:
        raise HTTPException(404, "There is no data available for this export yet.")
    output = io.StringIO()
    pd.DataFrame(rows).to_csv(output, index=False)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
