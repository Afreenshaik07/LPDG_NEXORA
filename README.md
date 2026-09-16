# NEXORA 2026 — Data Science Solution

## LPDG Innovation Hub Selection Challenge 2026

> **Evidence-Driven Weekly Field-Visit Prioritization for a LoRaWAN Utility Gateway Fleet**

**Candidate:** Afreen Shaik  
**Specialization Track:** Part 2 — Data Science  
**Institution:** Rajeev Gandhi Memorial College of Engineering and Technology (RGMCET)

---

## Executive Overview

NEXORA 2026 is a Data Science decision-support solution for prioritizing field visits across a fleet of approximately **320 LoRaWAN utility gateways**.

The operational constraint is straightforward:

> **The field team can perform only 15 gateway visits per week.**

Therefore, the problem is not simply to predict whether an individual gateway will fail.

The real operational question is:

> **Which 15 gateways should receive field attention first this week?**

NEXORA converts historical and current gateway evidence into an interpretable weekly ranking using:

- Current meter-read success
- Success deterioration
- Persistence of poor performance
- Supporting technical telemetry

The final system produces:

- **8 decision weeks**
- **15 gateways per week**
- **120 final prediction rows**
- Strict temporal cutoff
- Leakage-safe historical evaluation
- Explicit missing-meter handling
- Feature ablation
- Historical field-visit analysis
- Operational cost analysis
- Early-warning analysis
- Horizon robustness analysis
- Ranking stability analysis
- Professional Streamlit dashboard
- Reproducible offline prediction pipeline

### Final Production Score

```text
Current Success Risk          45%
Success Deterioration Risk    35%
Persistence Risk              10%
Technical Confirmation        10%
```

The final score is an **ordinal ranking score**, not a calibrated probability of gateway failure.

---

# Business Problem

LPDG operates a large fleet of utility gateways responsible for relaying meter-reading information.

The field team has a fixed operational capacity of only **15 visits per week**.

This creates two competing risks:

### Unnecessary Field Visit

A technician is sent to a gateway that does not require physical intervention.

### Missed High-Risk Gateway

A gateway requiring attention is not selected and poor performance may continue.

The Data Science objective is therefore to use the available operational evidence to construct a **Top-15 weekly field-visit priority list**.

---

# Operational Constraint

| Metric | Value |
|---|---:|
| Gateway master records | 332 |
| Monitored gateways | 320 |
| Weekly visit capacity | 15 |
| Cost per field visit | €380 |
| Weekly visit capacity cost | €5,700 |
| Decision weeks | 8 |
| Final predictions | 120 |

The nominal weekly field capacity is:

```text
15 × €380 = €5,700
```

Across eight decision weeks:

```text
8 × €5,700 = €45,600
```

---

# Challenge Decision Weeks

The final production pipeline generates rankings for all eight required decision weeks.

| Week | Decision Date |
|---:|---|
| 1 | 2026-02-02 |
| 2 | 2026-02-09 |
| 3 | 2026-02-16 |
| 4 | 2026-02-23 |
| 5 | 2026-03-02 |
| 6 | 2026-03-09 |
| 7 | 2026-03-16 |
| 8 | 2026-03-23 |

For every decision week:

```text
Approximately 320 gateways
          ↓
    Risk assessment
          ↓
    Weekly ranking
          ↓
      Select Top 15
          ↓
   Field-visit queue
```

---

# Solution Architecture

<img width="944" height="2352" alt="mermaid-diagram (1)" src="https://github.com/user-attachments/assets/94b0496d-aa0b-4f45-a613-3eca00809d2c" />

---

# Data Sources

The solution uses multiple supplied operational datasets rather than relying on a single source.

## 1. Gateway Master

- **332 gateways**
- Contains gateway-level information.
- Blank `decommissioned` status is treated as in service.

## 2. Telemetry

- **1,433,387 rows**
- **320 unique gateways**
- Coverage from **2025-08-01 to 2026-03-31**

Telemetry provides supporting technical information including:

- Offline duration
- Disconnections
- Reboots
- Reboot duration

## 3. Meter Read Success

- **7,226 observations**
- Weekly meter-read success data
- Available through **2026-01-26**

Meter-read success is the primary business-performance signal.

## 4. Historical Field Visits

- **642 historical visits**
- **247 unique gateways**
- Outcomes include:

```text
Kein Fehler gefunden
Fehler behoben
Kein Zugang
```

These historical outcomes provide evidence about actual operational events.

## 5. Engineer Review

- **110 matched records**
- Review outcomes include:

```text
Normal
Schlecht
```

Engineer-review information is used only from the point where it would have been available under the temporal cutoff.

---

# Dataset Coverage

### Gateway Master

```text
Gateway records: 332
```

### Telemetry

```text
Rows:             1,433,387
Unique gateways:  320
Start:            2025-08-01
End:              2026-03-31
```

### Meter Read Success

```text
Rows: 7,226
Frequency: Weekly
Available through: 2026-01-26
```

### Historical Field Visits

```text
Visits: 642
Unique gateways: 247
```

### Engineer Review

```text
Matched records: 110
```

---

# Data Preparation

## Weekly Gateway × Week Dataset

The raw event-level and weekly datasets are transformed into a common analytical structure:

```text
Gateway × Week
```

Each row represents one gateway in one weekly decision context.

The final weekly analytical dataset contains:

```text
Shape:              (10225, 29)
Unique gateways:    320
```

The production generation process reports:

```text
Decision period: 2026-02-02 to 2026-03-23
Weekly dataset shape: (10225, 29)
Available data: 2025-07-28 to 2026-03-30
Unique gateways: 320
Rows written: 120
Weeks: 8
```

---

# Temporal Leakage Prevention

Temporal leakage was treated as a core engineering requirement.

For every decision week, only information available **before the Monday decision cutoff** is used.
<img width="1156" height="352" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/b2c7fd44-2cf1-4923-9d8e-b290c53986ec" />


## Information excluded

- Future meter observations
- Future telemetry
- Future field-visit outcomes
- Future engineer-review information

## Information allowed

- Historical meter observations
- Historical telemetry
- Historical field-visit evidence
- Previously available gateway information

This ensures that the historical evaluation follows the same information constraints as the production prediction process.

---

# Missing Meter Data

A missing meter observation is **not treated as zero** and is **not automatically interpreted as failure**.

The pipeline explicitly tracks:

```text
meter_observed
last_meter_week
meter_age_weeks
```

## Historical Success

When a valid historical success observation exists, it is carried forward where appropriate.

## No Historical Success

When no historical success is available, the fleet median is used as a fallback.

## Success Change

`success_change` is calculated only when actual meter observations support the comparison.

This prevents missing data from creating artificial deterioration.

---

# Feature Engineering

The final production solution uses four major evidence groups.
<img width="2005" height="832" alt="mermaid-diagram (2)" src="https://github.com/user-attachments/assets/94e7b2e8-a540-4dc7-b135-458ce31ff307" />


## 1. Current Success

Recent meter-read success represents current gateway performance.

Lower success indicates weaker recent performance.

## 2. Success Deterioration

`success_change` captures whether performance is improving or deteriorating.

This is important because two gateways with similar current performance can have different trajectories.

## 3. Persistence

`low_success_streak` captures sustained poor performance.

A gateway that remains weak over several observations should not disappear from the ranking simply because it has already appeared in a previous weekly recommendation.

## 4. Technical Confirmation

Supporting telemetry includes:

- Offline duration
- Disconnection count
- Reboot count
- Reboot duration

Technical signals are deliberately given a smaller weight so that noisy telemetry does not dominate the business-performance signal.

---

# Important Data-Quality Correction

During development, an issue was identified in the handling of:

```text
offline_duration_sec
```

The initial implementation summed hourly values and produced implausibly large weekly durations.

This was corrected so that the final weekly value represents the appropriate observed weekly offline duration.

Disconnections and reboots are aggregated because their counts represent events.

The corrected implementation was incorporated into the final production pipeline.

This correction is also documented in `AI-USAGE.md`.

---

# Defining "Needs a Visit"

The supplied data does not provide a complete unbiased ground-truth label for:

```text
Needs Field Visit
```

Therefore, NEXORA defines higher visit priority using multiple observable signals.

A gateway receives higher priority when the available evidence indicates:

1. Weak current meter-read performance
2. Negative performance deterioration
3. Persistent low performance
4. Supporting technical problems

This is a **risk-prioritization definition**, not a claim that the gateway is guaranteed to fail.

---

# Why Historical Visits Are Not Treated as Perfect Labels

Historical field visits are operationally selected.

A gateway not appearing in the field-visit table does not necessarily mean that it was healthy.

Therefore, directly using historical visits as a supervised target could cause a model to learn the historical dispatch process rather than the underlying gateway risk.

The final approach instead uses historical visits as **evidence for validation and feature selection**.

---

# Historical Field-Visit Evidence

A leakage-safe historical analysis identified:

```text
289 matched field visits
```

with complete pre-visit weekly information.

Outcomes:

```text
114 Repair
175 No Error Found
```

## Pre-Visit Performance

| Metric | Repair | No Error Found |
|---|---:|---:|
| Average 4-week success | 0.6578 | 0.8074 |
| Latest success | 0.5459 | 0.7942 |

The difference provides evidence that recent meter-read performance contains useful information for prioritization.

However, this evidence does not imply that low success alone is sufficient to determine whether a visit is required.

---

# Visual Analytics

## Figure 1 — Historical Performance Before Field Visits

<img width="810" height="502" alt="image" src="https://github.com/user-attachments/assets/fc06289f-9040-4092-9e50-5b403908b8c9" />


### What the Figure Shows

This figure compares average 4-week meter-read success immediately before historical field visits.

The average success before repair visits was approximately:

```text
0.6578
```

The average success before no-error-found visits was approximately:

```text
0.8074
```

### Interpretation

Repair visits were historically preceded by weaker recent meter-read performance than visits where technicians found no error.

This supports the use of recent meter-read success as an important component of visit-priority scoring.

At the same time, the difference is not treated as a perfect classifier because historical field visits are selectively observed.

---

## Figure 2 — Historical Repair Rate and Economic Visit Threshold

<img width="902" height="557" alt="image" src="https://github.com/user-attachments/assets/6e70cdb3-ae27-44ff-b19d-a124d1528949" />


### What the Figure Shows

This figure compares observed historical repair rates across relative-risk bands.

The repair rate becomes substantially higher in the higher-risk bands.

The chart also includes an economic reference based on:

```text
Visit cost = €380
Missed-failure cost = €600
```

The break-even calculation is:

```text
Break-even probability
= Visit cost / Missed-failure cost

= €380 / €600

= 0.6333

≈ 63.3%
```

### Interpretation

The economic threshold provides a way to reason about when sending a technician may be economically justified compared with the potential cost of missing a failure.

This is an **economic reference point**.

The NEXORA priority score itself is not a calibrated failure probability.

---

## Figure 3 — Top 15 Gateways for Week Starting 2026-02-02
<img width="997" height="606" alt="image" src="https://github.com/user-attachments/assets/76fdefb0-4a3a-4e39-b127-1929c6f52ede" />


### What the Figure Shows

This figure presents the actual Top-15 operational output of the production pipeline for:

```text
Week starting: 2026-02-02
```

The horizontal bars represent visit-priority scores.

Gateways are ordered from highest to lowest priority.

### Interpretation

The purpose of the ranking is to convert multiple operational signals into a practical field-visit queue.

Instead of producing only:

```text
Failure / No Failure
```

the system produces:

```text
Rank 1
Rank 2
Rank 3
...
Rank 15
```

This directly matches the operational constraint of 15 available visits per week.

---

# What the Three Visualizations Demonstrate

The three figures represent three stages of the Data Science solution.

### 1. Historical Relationship

Lower recent meter-read success is associated with historical repair outcomes.

### 2. Economic Interpretation

Higher-risk bands show higher historical repair rates and can be interpreted against the economics of field visits.

### 3. Operational Decision

The final scoring system converts the evidence into a Top-15 field-visit ranking.

Therefore:

```text
Raw Operational Data
        ↓
Historical Evidence
        ↓
Risk Signals
        ↓
Priority Score
        ↓
Weekly Ranking
        ↓
Top-15 Field-Visit Decision
```

---

# Final Risk Scoring

The production score is:

```text
visit_priority_score =
    0.45 × success_risk
  + 0.35 × success_change_risk
  + 0.10 × persistence_risk
  + 0.10 × technical_confirmation
```

## Weight Breakdown

| Signal | Weight | Purpose |
|---|---:|---|
| Current success risk | 45% | Current business performance |
| Success-change risk | 35% | Detect deterioration |
| Persistence risk | 10% | Capture sustained weakness |
| Technical confirmation | 10% | Supporting technical evidence |

The final score is used for **relative ranking**.

It should not be interpreted as a probability.

---

# Business Risk

The core business-risk component is:

```text
business_risk =
    0.50 × success_risk
  + 0.50 × success_change_risk
```

Current performance and deterioration therefore form the main business signal.

Technical confirmation is intentionally limited to 10%.

---

# Percentile Risk Transformation

The raw features have different units and scales.

The solution converts them into comparable percentile-based risk values.

For success:

```text
Lower success → Higher risk
```

For deterioration:

```text
More negative change → Higher risk
```

For persistence:

```text
Longer low-success streak → Higher risk
```

This creates an interpretable common risk scale.

---

# Operational Priority Tiers

After calculating the priority score, gateways are ranked.

| Rank | Operational Tier |
|---|---|
| 1–5 | Critical |
| 6–10 | High |
| 11–15 | Medium |

These tiers are communication categories applied after ranking.

They are not probability estimates.

---

# Why a Black-Box Supervised Model Was Not Selected

A conventional supervised failure classifier was considered.

However, the supplied data does not provide a complete unbiased target for:

```text
Needs Field Visit
```

Historical visits are affected by operational selection.

Using observed field visits directly as labels could result in learning the historical dispatch behavior rather than the actual gateway risk.

The final evidence-first approach was therefore selected because it provides:

- transparency;
- temporal validity;
- interpretable features;
- direct operational ranking;
- reproducibility;
- easier explanation to field operations.

---

# Feature Ablation

Different feature combinations were tested against historical evidence.

| Feature Combination | Recall@15 | Precision | Estimated Cost |
|---|---:|---:|---:|
| Current success only | 35.2% | 71.2% | €675,900 |
| Success + deterioration | 37.7% | 75.9% | €656,100 |
| Business-only combined | 37.4% | 75.2% | €659,100 |
| Final + technical confirmation | 37.6% | 75.7% | €657,300 |
| Success + persistence | 33.2% | 67.4% | €691,500 |

### Interpretation

Current success provides a useful baseline.

Adding deterioration improves historical ranking performance, demonstrating that **trajectory information contains additional signal beyond current performance**.

The final production approach adds technical confirmation while maintaining the main emphasis on business performance and deterioration.

---

# Baseline vs Final Production Approach

## Baseline — Current Success Only

```text
Recall@15: 35.2%
Precision: 71.2%
Estimated cost: €675,900
```

## Final Production Approach

```text
Recall@15: 37.6%
Precision: 75.7%
Estimated cost: €657,300
```

### Historical Difference

```text
€675,900 − €657,300
= €18,600
```

The €18,600 value is an evaluated historical backtest difference and is **not a guaranteed future saving**.

---

# Horizon Robustness

The solution was evaluated beyond only the immediate one-week horizon.

| Horizon | Estimated Savings |
|---|---:|
| 1 week | €4,200 |
| 2 weeks | €16,200 |
| 3 weeks | €24,600 |

This analysis tests whether the ranking logic remains useful when the evaluation horizon changes.

---

# Early-Warning Analysis

The solution was also evaluated across:

```text
162 historical episodes
```

| Approach | Detected Before Event | Detected by Start | Never Detected | Missed Cost |
|---|---:|---:|---:|---:|
| Success only | 26 | 16% | 49% | €177,000 |
| Success + deterioration | 37 | 37% | 14% | €67,800 |
| Production score | 34 | 33.3% | 21% | €82,800 |

### Interpretation

The addition of deterioration substantially improves early identification compared with relying on current success alone.

This is one of the main reasons deterioration receives **35%** of the final score.

---

# Threshold Sensitivity

Threshold sensitivity was tested across several values.

Thresholds from:

```text
0.4 through 0.7
```

produced the same Top-15 selection and evaluated cost:

```text
€657,300
```

Higher thresholds changed selections and increased evaluated cost:

```text
0.8 → €665,700
0.9 → €728,700
```

This provides a robustness check around the operational ranking.

---

# Ranking Stability

The final production ranking was compared with a 60/40 alternative weighting across all eight decision weeks.

| Decision Week | Top-15 Overlap |
|---|---:|
| 2026-02-02 | 12/15 |
| 2026-02-09 | 13/15 |
| 2026-02-16 | 12/15 |
| 2026-02-23 | 11/15 |
| 2026-03-02 | 11/15 |
| 2026-03-09 | 10/15 |
| 2026-03-16 | 9/15 |
| 2026-03-23 | 10/15 |

### Overall Stability

```text
Mean Top-15 overlap: 73.3%
Mean absolute rank change: 3.65
```

This indicates that the weekly Top-15 selection is not completely dependent on one exact weighting configuration.

---

# Repeated Recommendations

NEXORA does not impose a blanket no-repeat rule.

A gateway can appear in multiple weekly Top-15 rankings if its risk remains elevated.

Instead, persistence is explicitly represented using:

```text
low_success_streak
```

This prevents the system from artificially suppressing a gateway simply because it was previously selected.

---

# End-to-End Production Pipeline

```text
Load supplied datasets
        ↓
Identify active gateways
        ↓
Build Gateway × Week dataset
        ↓
Apply strict temporal cutoff
        ↓
Handle missing meter observations
        ↓
Calculate current success
        ↓
Calculate success deterioration
        ↓
Calculate persistence
        ↓
Calculate technical confirmation
        ↓
Transform signals to percentile risks
        ↓
Apply 45 / 35 / 10 / 10 scoring
        ↓
Rank gateways within each week
        ↓
Select Top 15
        ↓
Write predictions.csv
        ↓
Validate submission
        ↓
Display results through dashboard
```


---

# Final Prediction Output

The production file is:

```text
predictions.csv
```

It contains:

```text
120 rows
8 decision weeks
15 ranked gateways per week
```

Coverage:

```text
2026-02-02 → 2026-03-23
```

The validation result is:

```text
predictions.csv: OK
15 ranked gateways for each of 8 weeks
2026-02-02 to 2026-03-23
```

---

# Reproducibility

The prediction pipeline is designed to run offline using the supplied datasets and local project files.

No external API calls, internet access or model downloads are required for prediction generation.

## Generate Predictions

```bash
python -m src.generate_predictions --data data --out predictions.csv
```

## Validate Submission

```bash
python validate_submission.py predictions.csv
```

The default command generates the complete eight-week prediction file.

---

# Streamlit Decision Dashboard

A professional Streamlit dashboard was developed as the presentation and decision-support layer.

## Dashboard Features

- Weekly decision-date selection
- KPI cards
- Decision snapshot
- Top-15 priority visualization
- Score-by-rank visualization
- Ranked gateway table
- Critical / High / Medium priority tiers
- Gateway drill-down
- Business-impact information
- Methodology view

The dashboard uses the same production scoring logic as the prediction pipeline.

It does not create a separate prediction model.

---

# Live Dashboard

## NEXORA 2026 Dashboard

**https://nexora-lpdg.streamlit.app/**

The dashboard provides an interactive view of the weekly gateway priorities and allows the decision logic to be explored at gateway level.

---

# Walkthrough Video

## 6–8 Minute Demonstration

**Watch the NEXORA 2026 demonstration:**

https://drive.google.com/file/d/1bzDyNQmgAiVvHSn1ncgWZZGxF2S5c_8_/view?usp=sharing

The walkthrough demonstrates:

1. Business problem
2. Dataset understanding
3. Data preparation
4. Temporal leakage prevention
5. Missing-data handling
6. Feature engineering
7. Scoring methodology
8. Historical evidence
9. Ablation results
10. Business impact
11. Prediction generation
12. Submission validation
13. Live Streamlit dashboard
14. Limitations

---

# GitHub Repository

## Source Code

https://github.com/Afreenshaik07/LPDG_NEXORA

The repository contains the prediction pipeline, analysis, validation, documentation and dashboard implementation.

---

# Project Structure

```text
NEXORA-2026/
│
├── analysis/
│   ├── exploratory analysis
│   ├── validation scripts
│   └── generated outputs
│
├── data/
│
├── src/
│   ├── data.py
│   ├── scoring.py
│   └── generate_predictions.py
│
├── predictions.csv
├── predictions_baseline.csv
├── baseline_3sigma.py
├── validate_submission.py
│
├── make_charts.py
├── make_risk_chart.py
├── make_operational_summary.py
│
├── dashboard.py
│
├── README.md
├── DECISIONS.md
├── AI-USAGE.md
├── DASHBOARD.md
├── LIVE_DEMO.md
├── STATISTICAL_EVIDENCE.md
├── requirements.txt
└── .gitignore
```

---

# Key Project Files

| File | Purpose |
|---|---|
| `src/data.py` | Data loading, weekly preparation and feature construction |
| `src/scoring.py` | Risk transformation and final scoring |
| `src/generate_predictions.py` | Production prediction generation |
| `predictions.csv` | Final 120-row submission |
| `predictions_baseline.csv` | Baseline output |
| `baseline_3sigma.py` | Baseline analysis |
| `validate_submission.py` | Submission validation |
| `dashboard.py` | Streamlit decision dashboard |
| `DECISIONS.md` | Engineering and Data Science decisions |
| `STATISTICAL_EVIDENCE.md` | Statistical validation |
| `AI-USAGE.md` | AI-assisted development record |
| `DASHBOARD.md` | Dashboard documentation |
| `LIVE_DEMO.md` | Demo documentation |
| `analysis/` | Supporting analysis and experiments |
| `outputs/` | Generated visualizations |

---

# Statistical Evidence

Detailed statistical and operational validation is documented in:

```text
STATISTICAL_EVIDENCE.md
```

The analysis includes:

- Historical field-visit comparison
- Feature ablation
- Baseline comparison
- Leakage-safe cutoff backtesting
- Operational cost analysis
- Horizon robustness
- Early-warning analysis
- Threshold sensitivity
- Ranking stability

---

# Engineering and Data Science Decisions

Major methodology decisions are documented in:

```text
DECISIONS.md
```

Key decisions include:

- strict temporal cutoff;
- explicit missing-meter handling;
- current success as a major signal;
- deterioration as a major signal;
- persistence as supporting evidence;
- technical telemetry as confirmation;
- no blanket no-repeat rule;
- ordinal ranking rather than probability claims;
- evidence-first approach instead of a black-box supervised classifier.

---

# AI Usage

AI assistance was used during development for:

- brainstorming;
- implementation guidance;
- debugging;
- technical explanations;
- documentation refinement.

The final data-processing decisions, scoring methodology, validation results and outputs were reviewed during development.

An important implementation issue involving:

```text
offline_duration_sec
```

was identified during validation.

The initial aggregation produced implausibly large weekly values because hourly values were being summed incorrectly.

The issue was corrected and the corrected implementation was incorporated into the final production pipeline.

AI assistance supported development, while the final methodology and outputs were validated against the supplied data.

---

# Limitations

## 1. Historical Visits Are Selective

Historical field visits are not a complete unbiased ground-truth dataset.

## 2. Sparse Meter Observations

Some gateways have stale or limited meter observations.

The pipeline explicitly tracks observation availability and age.

## 3. Telemetry Noise

Technical telemetry can contain noisy or incomplete measurements.

Therefore, technical confirmation receives only 10% of the final score.

## 4. Ordinal Score

The final score is intended for ranking.

It should not be interpreted as a calibrated probability.

## 5. Unobserved Operational Factors

Operational information not present in the supplied datasets cannot be incorporated.

## 6. Human Judgement

The system supports field planning and does not replace field-engineer judgement.

---

# What the Solution Can Do

NEXORA can:

- rank gateways by relative visit priority;
- identify weak recent performance;
- detect deterioration;
- capture persistent poor performance;
- incorporate supporting technical evidence;
- generate a weekly Top-15 field-visit queue;
- compare feature combinations;
- evaluate historical operational cost;
- evaluate early-warning behavior;
- evaluate ranking stability;
- provide an interactive dashboard;
- reproduce the same result from the same input data.

---

# What the Solution Cannot Do

NEXORA cannot:

- guarantee that a selected gateway will fail;
- provide a calibrated failure probability;
- know operational information absent from the supplied datasets;
- treat historical field visits as unbiased ground truth;
- replace human field-engineer judgement.

---

# Cost-to-Decision Perspective

The solution is optimized around the actual operational decision rather than a generic classification objective.

The evaluation therefore considers:

- Recall@15
- Precision
- Historical repair evidence
- Estimated operational cost
- Early-warning performance
- Horizon robustness
- Ranking stability
- Data availability
- Temporal validity

The purpose is to make the fixed weekly field-visit capacity more evidence-driven.

---

# One More Week of Data

The pipeline is designed for weekly updates.

When new data becomes available:

```text
New observations
       ↓
Update Gateway × Week dataset
       ↓
Refresh current success
       ↓
Refresh deterioration
       ↓
Update persistence
       ↓
Update technical evidence
       ↓
Recalculate risk
       ↓
Generate new Top-15 ranking
```

The same strict temporal cutoff is maintained.

---

# Core Data Science Insight

The central insight from the analysis is:

> **Current performance alone is not sufficient for field-visit prioritization.**

A gateway with moderate current performance but rapidly deteriorating success can represent a different operational risk from a gateway with similar current performance but stable behavior.

Therefore, the final production score combines:

```text
Current Performance
        +
Deterioration
        +
Persistence
        +
Technical Confirmation
        ↓
Visit Priority Score
        ↓
Weekly Ranking
```

The analysis shows that adding deterioration improves historical ranking performance compared with using current success alone.

---

# Why the Final Score Is Interpretable

Every component of the final score corresponds to an observable operational concept.

```text
45% → How weak is the gateway now?
35% → Is the gateway getting worse?
10% → Has poor performance persisted?
10% → Does technical telemetry provide confirmation?
```

This makes it possible to explain why a gateway appears in the Top 15 instead of relying on an opaque prediction.

---

# Final Result

The final NEXORA 2026 solution provides a validated and reproducible weekly gateway-prioritization pipeline.

It:

- processes approximately 320 monitored gateways;
- uses supplied operational data;
- prevents temporal leakage;
- handles missing meter observations explicitly;
- combines current performance, deterioration, persistence and technical evidence;
- generates exactly 15 ranked gateways per week;
- produces 120 final prediction rows;
- validates the final CSV;
- provides historical statistical evidence;
- evaluates operational cost;
- evaluates early-warning behavior;
- evaluates robustness;
- evaluates ranking stability;
- provides a professional Streamlit dashboard;
- runs offline;
- and provides a reproducible production workflow.

---

# Key Takeaway

The problem solved by NEXORA 2026 is not simply:

> **Which gateways might fail?**

It is:

> **Given a fixed capacity of 15 field visits per week, which gateways should receive attention first?**

The final decision workflow is:

```text
OBSERVE
   ↓
MEASURE
   ↓
COMPARE
   ↓
DETECT DETERIORATION
   ↓
CHECK PERSISTENCE
   ↓
CONFIRM WITH TECHNICAL SIGNALS
   ↓
CALCULATE PRIORITY SCORE
   ↓
RANK
   ↓
SELECT TOP 15
```

The result is an interpretable, leakage-safe and reproducible Data Science decision-support system for weekly field-visit prioritization.

---

# Submission Artifacts

The final submission contains:

- `predictions.csv`
- `DECISIONS.md`
- `AI-USAGE.md`
- `README.md`
- `STATISTICAL_EVIDENCE.md`
- `dashboard.py`
- Supporting analysis scripts
- Validation scripts
- Streamlit dashboard
- 6–8 minute demonstration recording

---


---

## Project Links

### GitHub

https://github.com/Afreenshaik07/LPDG_NEXORA

### Live Dashboard

https://nexora-lpdg.streamlit.app/

### Demo Recording

https://drive.google.com/file/d/1bzDyNQmgAiVvHSn1ncgWZZGxF2S5c_8_/view?usp=sharing
