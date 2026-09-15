# Statistical Evidence — NEXORA 2026

This is a documentation-ready statistical evidence layer for the Data Science
submission.

The evidence is descriptive and leakage-safe. It is used to support feature
selection and ranking design, not to claim causal relationships or hidden
ground truth.

Key measured comparisons:

- Average 4-week meter-read success:
  - repair: 65.8%
  - no-error: 80.7%
- Latest pre-visit success:
  - repair: 54.6%
  - no-error: 79.4%

Historical repair-rate breakdown by pre-visit success:

| Threshold | Repair | No-error |
|---:|---:|---:|
| <50% | 23.7% | 10.9% |
| <60% | 32.5% | 14.9% |
| <70% | 46.5% | 20.6% |
| <80% | 69.3% | 28.6% |
| <90% | 91.2% | 51.4% |

Interpretation:

Lower recent meter-read success is associated with a higher historical repair
rate in the visited-gateway population.

Caution:

- historical visits are selection-biased;
- these rates are not fleet-wide probabilities;
- the 60% threshold is an analytical proxy;
- LPDG's hidden ground truth is separate;
- the evidence supports ranking design but does not prove physical failure.
