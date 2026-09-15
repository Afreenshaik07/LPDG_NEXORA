# NEXORA 2026 — Data Science

## 1. Project Overview

This project addresses the LPDG Innovation Hub Selection Challenge 2026 from a Data Science perspective.

The operational task is:

> Rank the 15 gateways that should receive field attention each week.

The solution treats this as a constrained ranking problem rather than a simple faulty/not-faulty classification task.

The ranking combines:
- current meter-read performance,
- recent deterioration,
- persistence of low performance,
- supporting telemetry instability.

The final priority score is an ordinal ranking signal, not a calibrated probability.

## 2. Final Data Science Approach

The production score is:

```text
45%  Current business risk
35%  Recent deterioration
10%  Persistence
10%  Technical confirmation
```

Formula:

```text
visit_priority_score =
    0.45 × risk_success
  + 0.35 × risk_success_change
  + 0.10 × risk_persistence
  + 0.10 × technical_confirmation
```

Each component is converted into a relative 0–1 risk score.

The production decision is the top 15 eligible gateways for each decision week.

## 2A. Operational Interpretation

The submission CSV must keep the exact five-column schema required by LPDG, so risk tiers are kept in the separate operational summary rather than added to `predictions.csv`.

For manager-facing interpretation, the required top-15 list is split by relative weekly rank:

| Rank within selected 15 | Priority tier | Operational meaning |
|---:|---|---|
| 1–5 | Critical | strongest relative evidence |
| 6–10 | High | strong evidence within the constrained capacity |
| 11–15 | Medium | selected, but weaker relative evidence |

These tiers describe **relative priority**, not calibrated probability. They do not replace the numerical score or the required top-15 selection rule.

Run:

```bash
python make_operational_summary.py --predictions predictions.csv
```

This creates:

```text
analysis/outputs/final_weekly_summary.csv
analysis/outputs/final_cost_summary.csv
```

The weekly summary shows the selected count, score spread, risk-tier counts, top-ranked gateway, and its explanation.

## 2B. Cost-Aware Decision

The challenge economics are explicit:

```text
€380 per dispatched visit
€600 per faulty gateway-week
15 visits × 8 weeks = €45,600 fixed dispatch cost
```

The €600 amount is recurring when a faulty gateway remains unresolved. Therefore a missed fault can become more expensive over time than an early field intervention.

The model is evaluated as a ranking decision, not only as a prediction problem.

## 2C. Honest Failure Handling

A high score is not proof of a physical fault. False positives can spend the €380 visit budget; false negatives can leave a recurring €600 gateway-week cost.

Known limitations include:

- historical field visits are selection-biased;
- later challenge weeks have stale meter-read business observations;
- missing deterioration is treated as unknown/neutral rather than zero;
- telemetry instability can have multiple causes;
- the analytical 60% failure proxy is not LPDG's hidden ground truth.

A strong review should therefore explain at least one case where the ranking can be wrong and what evidence would reduce that uncertainty.

## 3. Data Used

Only the supplied challenge data is used.

Main inputs:

```text
data/
├── gateway_master.csv
├── meter_read_success.csv
├── field_visits.csv
├── engineer_review_2026-02.xlsx
└── telemetry/
    └── month=YYYY-MM/
        └── part-*.parquet
```

The telemetry loader discovers monthly partitions dynamically rather than relying on a fixed final month.

The supplied meter-read data ends earlier than the telemetry data. For later decision weeks, the latest observed meter-read business performance is carried forward while newer telemetry continues to update the technical picture. Missing deterioration information is treated as unknown/neutral rather than as observed zero change.

## 4. Data Quality Decisions

A key data-quality finding was that `offline_duration_sec` behaves as a firmware-reported counter/state. It is therefore represented using the final observed weekly value rather than summing hourly observations.

Event/count measures such as disconnections and reboots are aggregated across the week.

The project also distinguishes between:
- a value being absent,
- a value being zero,
- a value being present but potentially problematic.

These decisions are documented in `DECISIONS.md`.

## 5. Prediction Cutoff

For a decision week beginning on Monday, only information strictly before that Monday is used.

Conceptually:

```python
week_start < decision_week
```

This prevents information from the week being predicted from leaking into the ranking.

## 6. Historical Evaluation

The project includes leakage-safe historical analyses covering:

- repair vs. no-error behaviour,
- feature ablation,
- cost comparison,
- threshold sensitivity,
- future-horizon robustness,
- early-detection behaviour,
- final model comparison.

The historical failure definition used in these analyses is an analytical proxy:

```text
success_rate < 60%
```

This is not treated as LPDG's hidden ground truth.

The final evaluation uses a separate hidden ground truth supplied by LPDG during assessment.

---

## 6A. Statistical Evidence

Before selecting the production score, the project uses descriptive statistical
evidence to understand how the supplied historical outcomes relate to the
business signal.

Among leakage-safe historical visits with sufficient pre-visit history:

| Measure | Repair | No-error |
|---|---:|---:|
| Average 4-week meter-read success | 65.8% | 80.7% |
| Latest pre-visit success | 54.6% | 79.4% |

Lower recent meter-read success was also more common among repaired visits.

For example:

| Pre-visit success | Repair rate | No-error rate |
|---:|---:|---:|
| < 50% | 23.7% | 10.9% |
| < 60% | 32.5% | 14.9% |
| < 70% | 46.5% | 20.6% |
| < 80% | 69.3% | 28.6% |
| < 90% | 91.2% | 51.4% |

These are descriptive historical relationships, not causal estimates or
fleet-wide failure probabilities.

The statistical evidence is used to support feature selection and then checked
through leakage-safe ranking backtests, ablation, cost analysis, and horizon
robustness.

---

## 7. Business Cost Context

The challenge cost model used for analysis is:

```text
€380 per dispatched visit
€600 per faulty gateway-week
```

The analysis therefore focuses on the operational cost of ranking decisions rather than treating all errors as equally expensive.

Threshold sensitivity and horizon analyses are included to show how conclusions change under different assumptions.

See `DECISIONS.md` for the detailed judgement and rejected alternatives.

## 8. Repository Structure

```text
NEXORA-2026/
│
├── analysis/
│   ├── 01_*.py ... 22_*.py
│   └── outputs/
│
├── data/
│   └── challenge data
│
├── src/
│   ├── data.py
│   ├── scoring.py
│   └── generate_predictions.py
│
├── predictions.csv
├── predictions_baseline.csv
├── DECISIONS.md
├── AI-USAGE.md
├── requirements.txt
├── validate_submission.py
├── baseline_3sigma.py
└── README.md
```

The challenge dataset is not intended to be committed to the public repository.

## 9. Environment

Tested Python environment:

```text
Python 3.14
pandas 3.0.5
pyarrow 25.0.1
openpyxl 3.1.5
matplotlib 3.11.1
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

The runtime path is offline and does not require an API key, cloud service, or model download.

## 10. Run the Production Pipeline

From the project root:

```bash
python -m src.generate_predictions --data data --out predictions.csv
```

Expected production output:

```text
8 decision weeks
15 gateways per week
120 prediction rows
```

The current challenge period is:

```text
2026-02-02 to 2026-03-23
```

## 11. Validate the Submission

Run:

```bash
python validate_submission.py predictions.csv
```

Expected result:

```text
predictions.csv: OK
15 ranked gateways for each of 8 weeks
```

The validator checks the required columns, row count, ranks, duplicate gateways, numeric scores, and reason length.

## 12. Testing a Later Decision Period

The generator also supports a configurable start week and number of weeks.

Example:

```bash
python -m src.generate_predictions \
    --data data \
    --out live_test.csv \
    --start-week 2026-04-06 \
    --num-weeks 1
```

This was used to verify that the prediction generator is not tied to the original March 2026 end date.

## 13. Output

`predictions.csv` contains:

```text
week_start
rank
gateway_id
score
reason
```

The `score` is an ordinal visit-priority score.

The `reason` field provides a concise operational explanation based on the available evidence.

Example:

```text
meter-read success is 47%;
success fell 35% recently;
1,672s offline;
389 disconnections
```

## 14. Limitations

The solution cannot:

- prove that a gateway is physically broken,
- identify the exact failed hardware component,
- replace technician inspection,
- produce a calibrated failure probability,
- create meter-read observations that are not present in the supplied data,
- guarantee that every selected visit is necessary,
- guarantee a successful repair,
- reproduce LPDG's hidden ground truth before the final evaluation.

Important judgement limitations are documented in `DECISIONS.md`.

## 15. Documentation

### `DECISIONS.md`

Contains the five major Data Science choices, alternatives considered, evidence, cost implications, uncertainty, limitations, and the final production decision.

### `AI-USAGE.md`

Documents how AI assistance was used and includes a concrete example of an AI-generated interpretation that was identified and corrected.

## 16. Recording

Final 6–8 minute walkthrough:

```text
[ADD FINAL PUBLIC/UNLISTED RECORDING LINK BEFORE SUBMISSION]
```

The recording should be accessible without an organisation-specific approval request.

## 17. Final Submission Checklist

Before submission, verify:

```text
[ ] README.md present
[ ] DECISIONS.md present
[ ] AI-USAGE.md present
[ ] requirements.txt present
[ ] predictions.csv generated
[ ] predictions.csv passes validator
[ ] main command works from project root
[ ] no dataset committed to public Git history
[ ] no secrets or laptop-specific paths
[ ] final recording link works in a private browser
[ ] repository is public before the deadline
```

## 18. Final Principle

The solution is intentionally focused on a defensible operational ranking rather than unnecessary model complexity. The final presentation emphasises the decision, the cost of getting it wrong, the reasons behind each selected gateway, and the uncertainty that remains.

The key Data Science conclusion is:

> Current business performance is the strongest signal, recent deterioration adds early-warning value, and persistence/technical evidence provide supporting confirmation.
