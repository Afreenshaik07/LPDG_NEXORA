# NEXORA 2026 — Data Science Decisions

## 1. Part 2 Area

### Selected area

**Data Science**

I selected Data Science because the challenge is fundamentally a repeated operational decision:

> Which 15 gateways should the field team visit next week?

The goal is therefore not to build a complicated model for its own sake.

The goal is to use the supplied data to make this ranking decision more defensible and more useful than a simple rule.

The solution is designed around:

- measurable business impact;
- forward-looking evaluation;
- cost-sensitive decision making;
- explainable ranking;
- explicit uncertainty and limitations.

---

# 2. Business Problem

The field team can visit at most 15 gateways each week.

The operational problem is therefore a **ranking problem under a hard capacity constraint**, not a normal binary classification problem.

A classifier might say:

> "These 47 gateways are faulty."

That does not answer the actual business question.

The real question is:

> "Which 15 gateways should receive the limited field capacity first?"

Therefore the final system produces a relative priority score and ranks the eligible gateways.

The supplied challenge economics are:

- €380 for every dispatched visit;
- €600 for every gateway that remains faulty for one week.

The €600 cost can recur across multiple weeks, which makes early intervention more valuable than late intervention.

---

# 3. What "Needs a Visit" Means

There is no official ground-truth definition supplied for "needs a visit."

I therefore define a gateway as having **higher visit priority** when the available evidence indicates stronger operational risk.

The main evidence is:

1. current meter-read performance;
2. recent deterioration in meter-read performance;
3. persistence of low performance;
4. supporting technical instability.

This is a **priority definition**, not a claim that every selected gateway is physically broken.

For historical evaluation, I also use a transparent proxy:

> A future failure episode is a week where meter-read success falls below 60%.

This 60% boundary is an evaluation choice made for this analysis. It is not presented as LPDG's hidden ground truth.

---

# 4. Five Important Data Science Choices

## Choice 1 — Current condition ranking with an early-intervention objective

### Alternatives considered

**Alternative A:**

Detect gateways that appear faulty right now.

**Alternative B:**

Prioritize gateways that should be visited early because their current signals indicate elevated future risk.

### Decision

I chose a **current-condition ranking with an early-intervention objective**.

### Why?

The actual deliverable is a weekly top-15 visit list based on information available before the decision Monday. The ranking therefore needs to reflect current operational condition while using recent deterioration to identify gateways where early action may be valuable.

The challenge economics make timing important. A missed failure costs €600 per gateway per week, so leaving a faulty gateway unattended for several weeks can cost significantly more than catching it early.

### What this gets wrong

Some gateways may deteriorate for reasons that are temporary or unrelated to a field-serviceable problem.

Therefore a high priority score is not a diagnosis and should not be interpreted as proof of a physical fault.

---

## Choice 2 — Use a hard classification threshold or a ranking score?

### Alternatives considered

**Alternative A:**

Classify gateways as faulty/not faulty.

**Alternative B:**

Produce a continuous priority score and rank gateways.

### Decision

I chose a **relative risk/priority score**.

### Why?

The field team has exactly 15 available visits in the submission format.

Therefore the operational question is which gateways are highest priority relative to the rest of the fleet.

The score is therefore an **ordinal ranking signal**, not a probability.

For example:

> A score of 0.80 does not mean an 80% probability of failure.

It means the gateway has high relative priority compared with the other eligible gateways.

---

## Choice 3 — Use current performance only or include deterioration?

### Alternatives considered

I tested:

- current success only;
- success + recent deterioration;
- success + persistence;
- business-only combinations;
- the final combined model including technical confirmation.

### Decision

Recent deterioration was retained as an important signal.

### Evidence

The leakage-safe feature-ablation study produced:

| Strategy | Recall@15 | Precision@15 | Historical proxy cost |
|---|---:|---:|---:|
| Current success only | 35.2% | 71.2% | €675,900 |
| Success + deterioration | **37.7%** | **75.9%** | **€656,100** |
| Business-only combined | 37.4% | 75.2% | €659,100 |
| Final + technical confirmation | 37.6% | 75.7% | €657,300 |
| Success + persistence | 33.2% | 67.4% | €691,500 |

All reported costs in this document are historical **proxy** costs from the analytical evaluation, not the hidden final score.

The important result is that **current performance alone was weaker than current performance plus deterioration**.

The success+deterioration design produced the lowest historical proxy cost among the tested alternatives.

### What this gets wrong

Deterioration can be noisy. A single bad week does not necessarily mean a gateway will fail.

That is why deterioration is treated as one component of a ranking rather than as an automatic field-dispatch trigger.

---

## Choice 4 — Why retain persistence and technical confirmation?

### Alternatives considered

I tested whether persistence and technical telemetry were useful enough to remove or retain.

### Evidence

The current production score was:

- 41.3% Recall@15 at a 1-week horizon;
- 37.4% Recall@15 at a 2-week horizon;
- 33.8% Recall@15 at a 3-week horizon.

It remained close to the simpler success+deterioration model at longer horizons.

The two models also produced substantial recommendation overlap on the actual eight challenge weeks:

- mean top-15 overlap: **73.3%**;
- minimum top-15 overlap: **60.0%**.

This means the extra persistence/technical information mainly affects the marginal gateways around the top-15 boundary rather than completely changing the decision.

### Decision

I retained the richer production score because the actual operational question is the next week's visit list, where it performed best in the 1-week horizon test, while still remaining close to the simpler alternative over longer horizons.

### Important qualification

The ablation results also show that most of the useful predictive signal comes from current business performance and deterioration.

Therefore persistence and technical signals are treated as **supporting evidence**, not as dominant drivers.

---

## Choice 5 — How should historical field visits be used?

### Alternatives considered

**Alternative A:**

Treat `field_visits.csv` outcomes as the true label.

**Alternative B:**

Use them as historical evidence and diagnostic/backtesting information, while acknowledging their selection bias.

### Decision

I chose **Alternative B**.

### Why?

Historical field visits are not a random sample of all gateways.

A technician was already sent because somebody suspected a problem.

Therefore:

> A repair outcome among visited gateways does not represent an unbiased fleet-wide failure probability.

The field-visit data is therefore useful for:

- understanding failure patterns;
- comparing repair vs no-error cases;
- sanity-checking the ranking;
- historical backtesting.

It is **not** treated as the hidden ground truth.

The separate hidden ground truth used by the final evaluation is not contained in the supplied historical visit outcomes or engineer review.

---

# 5. Feature Engineering

## Business signal

Meter-read success is calculated as:

`meters_read / meters_expected`

Lower success means greater business impact.

## Deterioration

Recent change is:

`current_success_rate - previous_success_rate`

More negative values indicate stronger deterioration.

## Persistence

The solution tracks consecutive weeks below an internal low-success level.

This is useful as supporting evidence for recurring problems.

## Technical confirmation

The technical evidence uses:

- offline duration;
- backhaul disconnections;
- reboot count;
- reboot duration.

`offline_duration_sec` is handled as a firmware-reported counter/state and therefore uses the final observed value for the week rather than summing it across hourly observations.

Event counts are summed across the week.

---

# 6. Final Production Score

The production model uses four components:

### Current business risk — 45%

Higher risk when meter-read success is lower.

### Recent deterioration — 35%

Higher risk when success has fallen recently.

### Persistence — 10%

Higher risk when low performance persists.

### Technical confirmation — 10%

Supporting evidence from telemetry instability.

The final formula is:

`0.45 × risk_success`

`+ 0.35 × risk_success_change`

`+ 0.10 × risk_persistence`

`+ 0.10 × technical_confirmation`

All components are transformed into relative 0–1 risk scores.

The final score is therefore a **ranking score, not a probability**.


### Operational interpretation tiers

Because the final score is ordinal and the challenge requires exactly 15 selected gateways, fixed global score bands would be easy to misinterpret. For manager-facing communication, the selected list is therefore divided by relative weekly rank:

| Rank within selected 15 | Tier | Interpretation |
|---:|---|---|
| 1–5 | Critical | strongest relative evidence |
| 6–10 | High | strong evidence within the constrained capacity |
| 11–15 | Medium | selected, but weaker relative evidence |

These labels are descriptive only. They are not calibrated probabilities and do not replace the required rule of ranking the eligible gateways and selecting the top 15.

### Missing and delayed meter observations

When there is no newer meter observation, `success_change` is treated as **unknown rather than zero deterioration**.

The ranking assigns this missing deterioration signal a neutral relative risk of 0.5 instead of inventing a measured 0% change.

For later challenge weeks, the latest known meter-read performance is carried forward while newer telemetry continues to update the technical picture.

This distinction is deliberate because an absent observation and an observed zero change do not mean the same thing.

---

# 7. Why This Production Model?

The final model was not selected because it has more components.

It was selected after comparison against simpler alternatives.

### Historical leakage-safe comparison

Compared with current-success-only ranking, the combined approach achieved:

- higher average Recall@15;
- higher average Precision@15;
- lower historical proxy cost.

In the two-week leakage-safe comparison:

**Simple current-success ranking**

- Recall@15: 35.2%
- Precision@15: 71.2%
- Cost: €675,900

**Combined Data Science score**

- Recall@15: 37.6%
- Precision@15: 75.7%
- Cost: €657,300

Estimated historical proxy saving:

> **€18,600**

The feature-ablation study later showed an even slightly better challenger:

> success + deterioration only

with an estimated historical proxy cost of €656,100.

However, the current production score performed best at the one-week horizon, which is most closely aligned with the actual operational decision of whom to visit next week.

Therefore I retained the production model rather than changing it purely to minimize one retrospective metric.

---


# 7A. Statistical Evidence Supporting the Ranking

The Data Science decision is supported by descriptive and comparative
statistical analysis before selecting the final production score.

## Descriptive comparison: repair vs no-error

Among leakage-safe historical visits with sufficient pre-visit history,
four-week meter-read success was:

- average success for repair cases: **65.8%**;
- average success for no-error cases: **80.7%**.

For the latest available pre-visit success observation:

- repair cases: **54.6%**;
- no-error cases: **79.4%**.

This shows a clear directional relationship: historically repaired gateways
tended to have worse recent meter-read performance than gateways that resulted
in no error found.

The relationship is not treated as a causal statement and is not treated as
the company's hidden ground truth.

## Threshold breakdown

The same historical analysis showed that low meter-read success was more common
among repaired visits.

| Pre-visit success threshold | Repair rate | No-error rate |
|---:|---:|---:|
| < 50% | 23.7% | 10.9% |
| < 60% | 32.5% | 14.9% |
| < 70% | 46.5% | 20.6% |
| < 80% | 69.3% | 28.6% |
| < 90% | 91.2% | 51.4% |

These are historical descriptive rates among previously visited gateways.
They support the use of current meter-read performance as a business-risk
signal, but they do not imply that the threshold itself defines a fault.

## Why this is useful for the final score

The descriptive evidence supports the ordering:

1. current business performance;
2. recent deterioration;
3. persistence;
4. technical confirmation.

The subsequent feature-ablation experiments then test whether adding those
signals improves the actual top-15 decision rather than relying only on
descriptive relationships.

## Statistical uncertainty and caution

The historical field-visit data is selection-biased because visits were not
randomly assigned across the fleet.

Therefore:

- group differences are evidence of association, not causation;
- historical rates are not fleet-wide failure probabilities;
- the 60% proxy is an analytical evaluation definition;
- the hidden LPDG ground truth is separate;
- later challenge weeks have stale business observations.

The purpose of the statistical analysis is therefore **evidence for ranking
design**, not proof of physical failure.

# 8. Horizon Robustness

The combined approach was compared with current-success-only across different future horizons.

| Horizon | Simple Recall@15 | Combined Recall@15 | Simple Precision | Combined Precision | Combined cost saving |
|---|---:|---:|---:|---:|---:|
| 1 week | 40.1% | **41.3%** | 77.3% | **79.2%** | €4,200 |
| 2 weeks | 35.9% | **37.4%** | 67.2% | **70.8%** | €16,200 |
| 3 weeks | 32.5% | **33.8%** | 58.4% | **62.0%** | €24,600 |

This supports the conclusion that adding information beyond current performance is useful, although the size of the improvement is modest.

These are historical proxy evaluations and should not be read as the hidden final score.

---

# 9. Early-Detection Evidence

I identified historical failure episodes using the transparent proxy:

`success_rate < 60%`

There were 162 historical episodes in the evaluation.

The episode analysis showed:

| Strategy | Caught before episode | Caught by episode start | Never selected | Estimated missed cost |
|---|---:|---:|---:|---:|
| Current success only | 26 | 16.0% | 49 | €177,000 |
| Success + deterioration | **37** | **37.0%** | **14** | **€67,800** |
| Current production score | 34 | 33.3% | 21 | €82,800 |

The success+deterioration design showed the strongest early-intervention behaviour in this proxy evaluation.

This supports keeping deterioration as an important part of the Data Science reasoning.

---

# 10. Threshold Decision

The challenge requires exactly 15 submitted gateways per week, so a threshold does not replace the top-15 ranking.

A threshold is instead useful as an **operational sensitivity control**.

The tested 2-week historical results were:

| Score threshold | Average gateways above threshold | Recall@15 | Precision@15 | Total proxy cost |
|---:|---:|---:|---:|---:|
| 0.40 | 181.0 | 37.6% | 75.7% | €657,300 |
| 0.50 | 132.7 | 37.6% | 75.7% | €657,300 |
| 0.60 | 84.2 | 37.6% | 75.7% | €657,300 |
| 0.70 | 46.2 | 37.6% | 75.7% | €657,300 |
| 0.80 | 25.0 | 36.6% | 73.6% | €665,700 |
| 0.90 | 11.8 | 28.9% | 58.4% | €728,700 |

### Interpretation

Thresholds from 0.40 through 0.70 produced the same top-15 dispatch in the tested historical setup.

Once the threshold became very high, the eligible pool became too restrictive and historical missed-failure cost increased.

Therefore I do **not** claim that one arbitrary threshold such as 0.60 is the "correct" threshold.

The evidence instead supports:

> a broad stable operating range, with very high thresholds becoming increasingly conservative.

Because the submission requires exactly 15 gateways, the operational dispatch rule remains **rank the eligible gateways and take the top 15**. The threshold is a sensitivity control rather than the final dispatch rule.

---

# 11. Data Cutoff and Leakage Prevention

For a decision labelled with a Monday `week_start`, only information that existed **before that Monday** is permitted to influence the ranking.

The evaluation experiments explicitly apply:

`week_start < decision_week`

This prevents using observations from the week being predicted.

For example, information recorded on or after Monday 2026-02-09 cannot be used to rank the 2026-02-09 decision.

The same principle applies to telemetry, meter-read data, field visits and engineer review.

---

# 12. Data Freshness Limitation

The supplied meter-read dataset ends earlier than the telemetry data.

Telemetry continues into February and March 2026.

For later challenge weeks:

- the latest known meter-read business performance is carried forward;
- newer telemetry continues to update the technical picture;
- when no new meter observation exists, deterioration is treated as unknown rather than as observed zero change.

Therefore later predictions should be interpreted as:

> **latest known business performance + newer technical telemetry**

rather than as fresh meter-read observations for every later week.

This limitation is documented rather than hidden.

---

# 13. Repeated Recommendations and Operational State

The current ranking is **stateless across weeks**: it ranks each week's eligible gateways from the information available at that decision point.

This means a gateway can remain high-priority and be selected repeatedly.

That can be inefficient within the same continuous fault episode because later visits cost €380 while only the earliest visit receives the episode's additional fault-credit under the challenge accounting.

I did not impose a blanket no-repeat rule because a later selection can be justified when it belongs to a separate fault episode.

A production system with feedback from field visits should therefore track previous recommendations and episode state, then suppress or adjust repeat recommendations when they represent the same unresolved problem.

---

# 14. Uncertainty and Known Failure Modes

The model can be wrong in several ways.

### False positive

A gateway looks risky but does not need a physical intervention.

This can waste the €380 visit.

### False negative

A faulty gateway is not selected.

This can incur €600 per week while the fault remains unattended.

### Temporary deterioration

A short-lived performance drop may recover without field intervention.

### Telemetry ambiguity

Missing or abnormal telemetry can have multiple causes and should not automatically be interpreted as physical failure.

### Historical selection bias

Historical technician visits are biased toward gateways that were already suspected.

### Proxy target limitation

The 60% future-failure definition is an analytical proxy, not the company's hidden ground truth.

### Data freshness limitation

For later challenge weeks, the business signal is partly stale because the supplied meter-read data ends on 2026-01-26. New telemetry reduces this gap for technical signals but cannot create missing business observations.

---

# 15. What the Model Can Do

The solution can:

- rank 15 gateways for each required challenge week;
- combine business and technical evidence;
- emphasize recent deterioration;
- account for persistent poor performance;
- produce manager-readable explanations;
- incorporate newer telemetry into later challenge-week decisions;
- provide a repeatable ranking process;
- quantify historical proxy cost and sensitivity.

---

# 16. What the Model Cannot Do

The solution cannot:

- prove that a gateway is physically broken;
- identify the exact failed hardware component;
- replace technician inspection;
- produce a calibrated probability of failure;
- create new meter-read observations that are not in the supplied data;
- guarantee that every selected visit is necessary;
- guarantee a successful repair;
- guarantee performance against LPDG's hidden ground truth before that ground truth is observed.

---

# 17. What I Would Change With One More Week of Data

I would prioritize the next steps as follows:

1. **First:** collect the next week's field outcomes and use them to evaluate and, if justified, recalibrate the 15-gateway decision.
2. **Second:** examine whether repeated recommendations correspond to the same continuous fault episode or to genuinely new episodes.
3. **Third:** reassess the practical operating threshold after new outcomes are observed, including the cost of changing it in either direction.
4. **Fourth:** test whether technical signals add more incremental value when fresher business observations are available.

The objective would be to learn from new evidence rather than automatically increase model complexity.

The most valuable next week is not simply "more data"; it is one more week that gives a new field outcome, reveals whether repeat recommendations represent the same episode, and allows the €380/€600 trade-off to be re-tested against fresh evidence.

---

# 17A. Cost-to-Decision Translation

The challenge provides two relevant prices:

| Event | Cost |
|---|---:|
| Dispatched visit | €380 |
| Faulty gateway remaining for one week | €600 |

At the required capacity of 15 visits per week:

`15 × €380 = €5,700` fixed visit cost per week.

Across the eight challenge weeks:

`120 × €380 = €45,600` fixed dispatched-visit cost.

These visit costs are the planned field-service spend represented by the submission. The €600 figure is a recurring missed-fault cost used to reason about prioritisation; it is not inferred from `predictions.csv`.

This leads to the operational trade-off:

> False positives spend limited visit capacity; false negatives can create recurring cost when a faulty gateway remains unattended.

The purpose of the ranking is therefore to spend the 15 available slots on the gateways with the strongest available evidence.

# 17B. One Concrete Failure Case to Explain

The ranking should be presented as fallible.

A useful review example is any historical gateway that was selected as high priority but ended with a no-error visit, or any faulty episode that the ranking failed to select early.

For that case, the explanation should follow this structure:

1. What signals were available before the decision?
2. Why did those signals push the gateway upward?
3. What happened at the field outcome?
4. Which evidence was missing or misleading?
5. What would one more week of data have added?

This demonstrates that the analysis is not being defended as perfect. It is being inspected for where the decision process can fail.

# 18. Final Decision

The final production approach remains the four-component Data Science score:

`0.45 current success risk`

`+ 0.35 deterioration risk`

`+ 0.10 persistence risk`

`+ 0.10 technical confirmation`

The model is selected because it provides a strong and explainable short-term operational ranking while remaining close to the simpler success+deterioration challenger over longer horizons.

The key conclusion from the experiments is not that every feature is equally important.

It is:

> **Current business performance is the strongest signal, recent deterioration adds meaningful early-warning value, and persistence/technical evidence can provide additional short-term confirmation without fundamentally changing the top-priority population.**

---

# 19. Reproducibility

The prediction pipeline reads from the top-level `data/` directory and can be rerun without internet access.

The main prediction command is:

```bash
python -m src.generate_predictions --data data --out predictions.csv
```
