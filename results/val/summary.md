# Results Summary

_Generated automatically from results.csv by make_plots.py._

_All numbers are conditional on the data and hyperparameters described in README.md._


## Table A — Accuracy by Encoder × Method × Shots (balanced queries, split: val)

| encoder | method | shots=1 | shots=5 | shots=10 | shots=20 |
|---|---|---|---|---|---|
| clip_b16 | logreg | 0.416 ± 0.013 | 0.652 ± 0.008 | 0.749 ± 0.007 | 0.818 ± 0.005 |
| clip_b16 | majority | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 |
| clip_b16 | proto | 0.395 ± 0.013 | 0.575 ± 0.008 | 0.627 ± 0.008 | 0.652 ± 0.007 |
| clip_b16 | random | 0.129 ± 0.005 | 0.129 ± 0.005 | 0.129 ± 0.005 | 0.129 ± 0.005 |
| dinov2_s | logreg | 0.439 ± 0.012 | 0.734 ± 0.007 | 0.830 ± 0.006 | 0.889 ± 0.004 |
| dinov2_s | majority | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 |
| dinov2_s | proto | 0.396 ± 0.013 | 0.669 ± 0.008 | 0.743 ± 0.006 | 0.792 ± 0.006 |
| dinov2_s | random | 0.129 ± 0.005 | 0.129 ± 0.005 | 0.129 ± 0.005 | 0.129 ± 0.005 |
| resnet50 | logreg | 0.416 ± 0.012 | 0.673 ± 0.009 | 0.768 ± 0.006 | 0.833 ± 0.005 |
| resnet50 | majority | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 |
| resnet50 | proto | 0.401 ± 0.012 | 0.618 ± 0.009 | 0.678 ± 0.007 | 0.718 ± 0.006 |
| resnet50 | random | 0.129 ± 0.005 | 0.129 ± 0.005 | 0.129 ± 0.005 | 0.129 ± 0.005 |


## Table B — Transductive Gain vs Prototype (balanced queries)

| encoder | method | shots | Δacc (mean±CI) | win_rate | wilcoxon_p |
|---|---|---|---|---|---|
| clip_b16 | logreg | 1 | 0.021 ± 0.004 | 0.79 | 0.0000 |
| clip_b16 | logreg | 5 | 0.078 ± 0.006 | 1.00 | 0.0000 |
| clip_b16 | logreg | 10 | 0.122 ± 0.006 | 1.00 | 0.0000 |
| clip_b16 | logreg | 20 | 0.166 ± 0.006 | 1.00 | 0.0000 |
| dinov2_s | logreg | 1 | 0.043 ± 0.008 | 0.83 | 0.0000 |
| dinov2_s | logreg | 5 | 0.065 ± 0.007 | 0.98 | 0.0000 |
| dinov2_s | logreg | 10 | 0.087 ± 0.005 | 1.00 | 0.0000 |
| dinov2_s | logreg | 20 | 0.097 ± 0.005 | 1.00 | 0.0000 |
| resnet50 | logreg | 1 | 0.015 ± 0.004 | 0.77 | 0.0000 |
| resnet50 | logreg | 5 | 0.055 ± 0.005 | 0.98 | 0.0000 |
| resnet50 | logreg | 10 | 0.091 ± 0.005 | 1.00 | 0.0000 |
| resnet50 | logreg | 20 | 0.116 ± 0.005 | 1.00 | 0.0000 |


## Table C — Accuracy Under Imbalanced Queries (shots=5)

| encoder | method | balanced | α=1.0 | α=0.3 |
|---|---|---|---|---|
| clip_b16 | logreg | 0.652 ± 0.008 | — | — |
| clip_b16 | majority | 0.125 ± 0.000 | — | — |
| clip_b16 | proto | 0.575 ± 0.008 | — | — |
| clip_b16 | random | 0.129 ± 0.005 | — | — |
| dinov2_s | logreg | 0.734 ± 0.007 | — | — |
| dinov2_s | majority | 0.125 ± 0.000 | — | — |
| dinov2_s | proto | 0.669 ± 0.008 | — | — |
| dinov2_s | random | 0.129 ± 0.005 | — | — |
| resnet50 | logreg | 0.673 ± 0.009 | — | — |
| resnet50 | majority | 0.125 ± 0.000 | — | — |
| resnet50 | proto | 0.618 ± 0.009 | — | — |
| resnet50 | random | 0.129 ± 0.005 | — | — |


## Narrative Findings (auto-generated)

- **clip_b16, 5-shot**: `logreg` improved over `proto` by 0.078 ± 0.006 acc pts (p<0.05).
- **clip_b16, 10-shot**: `logreg` improved over `proto` by 0.122 ± 0.006 acc pts (p<0.05).
- **dinov2_s, 5-shot**: `logreg` improved over `proto` by 0.065 ± 0.007 acc pts (p<0.05).
- **dinov2_s, 10-shot**: `logreg` improved over `proto` by 0.087 ± 0.005 acc pts (p<0.05).
- **resnet50, 5-shot**: `logreg` improved over `proto` by 0.055 ± 0.005 acc pts (p<0.05).
- **resnet50, 10-shot**: `logreg` improved over `proto` by 0.091 ± 0.005 acc pts (p<0.05).