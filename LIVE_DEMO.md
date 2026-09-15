# NEXORA 2026 — Live Demo Runbook

## Goal

Show that the solution is rerunnable, explainable, and easy to change during the live session.

## A. Normal production run

From the repository root:

```bash
python -m src.generate_predictions --data data --out predictions.csv
python validate_submission.py predictions.csv
python make_operational_summary.py --predictions predictions.csv
```

Expected result:

- 120 prediction rows;
- 15 ranked gateways per week;
- validator reports `OK`;
- an operations summary is written to `analysis/outputs/`.

## B. What to show on screen

1. `src/data.py` — explain the cutoff and the weekly gateway-level dataset.
2. `src/scoring.py` — show the four score components.
3. `src/generate_predictions.py` — show that the decision week is configurable.
4. Run the command.
5. Open `predictions.csv`.
6. Open `analysis/outputs/final_weekly_summary.csv`.
7. Pick rank 1 from one week and explain its `reason`.

## C. One concrete change to rehearse

Do not change the submission schema or the 15-gateway rule.

For a live Data Science change, temporarily alter the two business weights while keeping the total at 1.00:

```text
Original:
45% current success
35% deterioration
10% persistence
10% technical confirmation

Example live change:
50% current success
30% deterioration
10% persistence
10% technical confirmation
```

Then regenerate a separate test file:

```bash
python -m src.generate_predictions --data data --out live_test.csv --start-week 2026-04-06 --num-weeks 1
```

Do not submit `live_test.csv`.

Explain:

> "I changed the business weighting, kept the rest of the pipeline fixed, and would compare how much the top-15 list changes before deciding whether the change is justified."

## D. Threshold explanation

The threshold is a sensitivity control, not the final dispatch rule.

The challenge requires exactly 15 gateways, so the final action is:

```text
eligible gateways -> rank -> select top 15
```

A very high threshold can make the eligible pool too small and increase missed-fault cost.

## E. Priority-tier explanation

The operational summary uses relative tiers within the selected 15:

- ranks 1–5: Critical
- ranks 6–10: High
- ranks 11–15: Medium

This avoids implying that an ordinal score of 0.80 or 0.90 is a calibrated probability or an absolute danger threshold.

## F. Failure case

Be ready to show one gateway that looked risky but did not produce a repair outcome, or one historical episode the ranking missed.

Say:

> "This is where the ranking can be wrong. The score is a prioritisation signal, not a diagnosis."

Then explain what evidence was available and what additional evidence would have helped.

## G. What one more week buys

Prioritise:

1. new field outcomes to evaluate the ranking;
2. episode-aware handling of repeated recommendations;
3. threshold re-evaluation using new outcomes and cost;
4. testing incremental value of technical signals with fresher business observations.


## Optional Dashboard

If asked for a visual interface, launch:

```bash
streamlit run dashboard.py
```

Use it only after the normal Python pipeline has produced and validated
`predictions.csv`.

The dashboard is read-only and uses the same prediction output. It does not
change the scoring logic, call an API, or use external data.

A useful explanation is:

> "The Python pipeline remains the source of truth. The dashboard is only an
> optional presentation layer for the same generated top-15 decisions."
