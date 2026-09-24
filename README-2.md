# Cost-Aware LLM Routing: Full Working Guide

This guide explains what to upload, what the backend does after upload, how
Synthetic Latency is calculated, how ILP and Goal Programming work, how to
choose their settings, and how to explain the output.

## 1. What This Project Does

The project selects exactly one LLM for each query. It does not assume one
model is best for all questions.

For every query i and model j:

~~~text
x[i,j] = 1 when model j is selected for query i
x[i,j] = 0 otherwise
~~~

The aim is to make a query-by-query routing plan that balances:

- Cost
- Quality
- Synthetic Latency

This is routing optimization. Model evaluation tells how a model performs
overall; routing decides which model should receive each individual query.

## 2. What Data To Upload

Upload benchmark result files for at least two LLMs evaluated on the same
queries. Usually one file represents one model.

Example:

~~~text
deepseek.json
gemini-flash.json
gemini-pro.json
gpt.json
~~~

The project already has sample MMLU-Pro JSON files in Data. Click Load
repository sample in the app to process them together.

### Required Data

| Field | Purpose |
| --- | --- |
| Query ID | Matches the same question across models. |
| Query text | Lets users inspect the question and validates alignment. |
| Model | Identifies the candidate LLM. |
| Quality | Measures that model result for that query. |
| Cost | Actual record-level dataset cost. |
| Prompt tokens | Used in Synthetic Latency. |
| Completion tokens | Used in Synthetic Latency. |

Quality must be from 0 to 1:

~~~text
1.0 means correct or full quality
0.0 means incorrect or no quality
~~~

The app uses uploaded record-level quality and cost. It never creates fictional
scores or model prices.

### JSON Input

~~~json
{
  "model_name": "example-model",
  "records": [
    {
      "index": 1,
      "origin_query": "Example question",
      "prompt_tokens": 200,
      "completion_tokens": 300,
      "cost": 0.0025,
      "score": 1.0,
      "prediction": "B",
      "ground_truth": "B"
    }
  ]
}
~~~

The top-level model_name is copied to each record if the record does not have
its own model or model_name field.

### CSV Or XLSX Input

~~~text
query_id,query,model,quality,prompt_tokens,completion_tokens,cost
1,"What is 2 + 2?",Model A,1.0,50,40,0.0002
1,"What is 2 + 2?",Model B,1.0,60,55,0.0005
~~~

The normalizer accepts aliases:

| Meaning | Allowed names |
| --- | --- |
| Query ID | query_id, index |
| Query | origin_query, query, question |
| Model | model, model_name |
| Quality | score, quality, performance |

## 3. Why All Models Need The Same Queries

The optimizer needs alternatives for the same question:

~~~text
                 Model A       Model B       Model C
Query 1          data          data          data
Query 2          data          data          data
Query 3          data          data          data
~~~

If a model has no response for a query, the application reports it as missing.
Only complete queries with a result from every detected model enter
optimization. A model cannot look cheaper merely because its result is absent.

## 4. What Happens After Upload

### Step 1: Browser Sends Files

The Next.js dashboard sends multipart files to:

~~~text
POST http://127.0.0.1:8000/upload
~~~

### Step 2: FastAPI Parses Files

| File type | Parser |
| --- | --- |
| JSON | Python json |
| CSV | pandas read_csv |
| XLSX/XLS | pandas read_excel |

### Step 3: Data Is Normalized

Every source record becomes:

~~~text
query_id, query, model, quality, prompt_tokens, completion_tokens,
cost, latency, prediction, ground_truth, raw_output
~~~

JSON, CSV, and Excel can therefore use the same backend pipeline.

### Step 4: Validation Runs

The backend checks:

- malformed, empty, or unsupported files;
- missing query, ID, model, quality, cost, or token data;
- non-numeric data;
- negative cost or token values;
- quality outside 0 to 1;
- duplicate query-model pairs;
- conflicting query text for one ID;
- missing model responses.

It shows:

~~~text
Queries
Models
Records
Expected combinations = queries x models
Missing combinations
Complete queries
~~~

For example, 3,000 queries and 5 models should create 15,000 query-model
records. If 50 are absent, the dashboard reports 50 missing combinations.

### Step 5: Backend Stores The Dataset

After successful validation, FastAPI keeps the normalized dataframe in memory.
Old optimization and analysis results are cleared, because they belong to the
previous dataset. The dashboard can now show real statistics and charts.

## 5. Synthetic Latency

### Why It Is Synthetic

The benchmark does not include reliable API response time for every query.
Top-level benchmark time_taken is the duration of an entire run, not the time
for a single model response. The project therefore never presents an estimate
as Actual API Latency.

### Formula

~~~text
T[i,j] = T0 + alpha * P[i,j] + beta * R[i,j]
~~~

| Symbol | Meaning |
| --- | --- |
| T[i,j] | Synthetic Latency of query i on model j |
| T0 | Fixed per-request overhead |
| P[i,j] | Prompt tokens |
| R[i,j] | Completion tokens |
| alpha | Prompt token coefficient |
| beta | Completion token coefficient |

Default settings:

~~~text
T0 = 0.20
alpha = 0.001
beta = 0.002
~~~

### Example

For 200 prompt tokens and 600 completion tokens:

~~~text
T = 0.20 + (0.001 x 200) + (0.002 x 600)
T = 0.20 + 0.20 + 1.20
T = 1.60
~~~

This is an estimate in Synthetic Latency units. It is not a claim that the
request took 1.60 seconds in reality.

### Effects Of Parameters

- Larger T0 increases the estimate for every request.
- Larger alpha penalizes long prompts.
- Larger beta penalizes long model outputs.

Changing the latency configuration recalculates every T[i,j]. Rerun ILP or
Goal Programming afterward to obtain a route using the new assumptions.

## 6. Matrices Used By The Solver

The backend converts complete records into three query-by-model matrices:

~~~text
C[i,j] = uploaded record-level cost
Q[i,j] = uploaded record-level quality
T[i,j] = calculated Synthetic Latency
~~~

Example:

~~~text
                 Model A       Model B
Cost, Query 1    0.0002        0.0005
Quality, Query 1 1.0           1.0
Cost, Query 2    0.0006        0.0014
Quality, Query 2 0.0           1.0
~~~

Model A is cheaper for both queries, but it is wrong for Query 2. The solver
can mix models to meet an overall quality requirement at minimum cost.

## 7. ILP: Integer Linear Programming

### Decision Variable

~~~text
x[i,j] = 1 if model j is selected for query i
x[i,j] = 0 otherwise
~~~

### Objective

~~~text
Minimize total cost = sum(i,j) C[i,j] * x[i,j]
~~~

### Constraint 1: One Model Per Query

~~~text
sum(j) x[i,j] = 1 for every query i
~~~

### Constraint 2: Minimum Average Quality

~~~text
sum(i,j) Q[i,j] * x[i,j] >= N * Q_min
~~~

N is the number of complete queries. Q_min is the chosen minimum average
quality. If N is 3,000 and Q_min is 0.85, selected quality must total at least
2,550. For binary scores, that means at least 85% selected answers are correct.

This is an overall average constraint, not a separate requirement for every
individual query.

### Constraint 3: Maximum Average Synthetic Latency

~~~text
sum(i,j) T[i,j] * x[i,j] <= N * T_max
~~~

If N is 3,000 and T_max is 2.0, total selected latency must be at most 6,000.
This is also an average constraint.

### Full ILP Formulation

~~~text
Minimize:
  sum(i,j) C[i,j] * x[i,j]

Subject to:
  sum(j) x[i,j] = 1                              for every i
  sum(i,j) Q[i,j] * x[i,j] >= N * Q_min
  sum(i,j) T[i,j] * x[i,j] <= N * T_max
  x[i,j] is binary
~~~

The backend uses PuLP with the CBC solver.

## 8. Choosing ILP Quality And Latency Inputs

The optimizer does not choose Q_min or T_max. They are your business, academic,
or service policy requirements.

### Choosing Minimum Average Quality

| Scenario | Starting Q_min range |
| --- | --- |
| Cost-saving experiment | 0.75 to 0.80 |
| Balanced benchmark routing | 0.80 to 0.85 |
| Quality-sensitive routing | 0.85 to 0.90 |
| Strict routing | 0.90 or higher if feasible |

Higher Q_min usually causes higher cost because the solver must select more
high-quality responses.

Never choose a value above the Highest achievable quality shown in the result.
If that diagnostic is 0.9303, a minimum quality of 0.95 is impossible.

### Choosing Maximum Average Synthetic Latency

T_max is the highest average estimated latency you allow.

Use the Lowest achievable latency diagnostic:

- below that value: impossible;
- slightly above it: strict speed requirement;
- much higher: gives the solver more low-cost routing freedom.

If lowest achievable latency is 1.2617, a reasonable sensitivity range is:

~~~text
1.3, 1.5, 1.7, 2.0
~~~

### Trade-Off

~~~text
Higher Q_min -> usually higher cost
Lower T_max -> more restricted routing, possibly higher cost
~~~

Run Sensitivity Analysis to demonstrate these trade-offs with your data.

## 9. How To Read An ILP Result

Example:

~~~text
Status: Optimal
Total cost: $1.9783
Average quality: 0.8500
Average Synthetic Latency: 1.6192
Objective: 1.9783
~~~

| Result | Explanation |
| --- | --- |
| Optimal | CBC proved this is the lowest-cost feasible route. |
| Total cost | Sum of all selected query-model costs. |
| Average quality | Mean quality of selected responses. |
| Average Synthetic Latency | Mean token-based latency estimate. |
| Objective | Same as cost because ILP minimizes total cost. |
| Allocation | Count and percentage of queries sent to each model. |

For Q_min = 0.85 and T_max = 2.00:

~~~text
0.8500 >= 0.85: quality requirement met
1.6192 <= 2.00: latency requirement met
~~~

Quality often equals the lower threshold exactly because a cost-minimizing
solver avoids extra quality that is not required.

The result diagnostics mean:

| Diagnostic | Meaning |
| --- | --- |
| Highest achievable quality | Best possible average quality if every query uses its best model. |
| Lowest achievable latency | Fastest possible route if every query uses its fastest model. |
| Cheapest possible cost | Lowest cost if quality and latency are ignored. |

These are separate reference bounds and may not occur in one route together.

If status is Infeasible, no route satisfies the exact requirements. Relax a
constraint deliberately; the application does not silently change it.

## 10. Goal Programming

ILP uses strict limits. Goal Programming uses preferred targets and priorities.

### Inputs

~~~text
Target total cost
Target average quality
Target average Synthetic Latency
Cost weight
Quality weight
Latency weight
~~~

Weights must total exactly 1.0.

Example:

~~~text
Target cost: 2.00
Target quality: 0.85
Target latency: 1.60
Cost weight: 0.50
Quality weight: 0.30
Latency weight: 0.20
~~~

### Deviation Equations

~~~text
Total cost + d_cost_minus - d_cost_plus = target cost
Total quality + d_quality_minus - d_quality_plus = N * target quality
Total latency + d_latency_minus - d_latency_plus = N * target latency
~~~

| Variable | Meaning |
| --- | --- |
| d_cost_plus | Cost above target, undesirable. |
| d_cost_minus | Cost below target, desirable. |
| d_quality_minus | Quality below target, undesirable. |
| d_quality_plus | Quality above target, desirable. |
| d_latency_plus | Latency above target, undesirable. |
| d_latency_minus | Latency below target, desirable. |

### Goal Objective

~~~text
Minimize:
  cost weight * d_cost_plus / target cost
  + quality weight * d_quality_minus / (N * target quality)
  + latency weight * d_latency_plus / (N * target latency)
~~~

Only undesirable deviations receive a penalty. This means lower cost, higher
quality, and lower latency are not punished.

## 11. ILP Versus Goal Programming

| Question | ILP | Goal Programming |
| --- | --- | --- |
| Is quality a hard requirement? | Yes | No, it is a target. |
| Is latency a hard requirement? | Yes | No, it is a target. |
| Main result | Lowest-cost feasible routing. | Best weighted compromise. |
| Best use | Non-negotiable policy or SLA. | Flexible trade-offs. |

Use ILP when quality or latency cannot be violated. Use Goal Programming when
you want to express priorities among competing targets.

Goal output includes total cost, average quality, average Synthetic Latency,
targets, deviations, weights, allocation, and query-level routing. A positive
quality deviation means the target was missed; it is a visible trade-off, not
a hard constraint violation.

## 12. Baselines, Sensitivity, Routing, And Exports

### Baselines

The application calculates three data-derived references:

1. Cheapest Model: lowest average cost model, used for every query.
2. Highest Quality Model: highest average quality model, used for every query.
3. Lowest Latency Model: lowest average Synthetic Latency model, used for every query.

Compare them with ILP and Goal Programming to show why query-level routing is
useful.

### Sensitivity Analysis

Sensitivity Analysis reruns optimization while changing:

- minimum quality;
- maximum Synthetic Latency;
- goal weights.

For every scenario it saves:

~~~text
Parameter value, solver status, total cost, average quality,
average Synthetic Latency, and model allocation
~~~

Infeasible scenarios are important results and remain visible.

### Query Routing

The final routing table provides:

~~~text
Query ID, query text, selected model, quality, cost,
prompt tokens, completion tokens, total tokens, Synthetic Latency
~~~

### Exports

| Export | Contents |
| --- | --- |
| Routing CSV | Selected model and metrics per query. |
| Comparison CSV | Baseline and optimization totals. |
| Sensitivity CSV | Scenario results. |
| Summary CSV | Dataset KPI values. |

All exported values come from the active uploaded dataset.

## 13. Recommended Demo Flow

1. Upload every model benchmark file or load repository sample.
2. Show validation counts and explain complete query coverage.
3. Show data-derived quality, cost, and token statistics.
4. Explain the Synthetic Latency formula.
5. Run ILP with Q_min = 0.85 and a feasible T_max such as 2.0.
6. Explain status, cost, achieved quality, latency, and allocation.
7. Change one constraint and show the resulting trade-off.
8. Run Goal Programming with cost-heavy and quality-heavy weights.
9. Compare strategies, inspect routing rows, and export results.

## 14. Limitations

- Dataset costs and quality are benchmark observations, not universal model facts.
- Synthetic Latency is a token-based estimate, not actual measured API latency.
- Only complete query-model combinations are optimized.
- FastAPI state is in memory, so restarting it clears the active dataset.
- A production deployment should add persistence, authentication, job queues,
  solver resource limits, and observed API telemetry.
