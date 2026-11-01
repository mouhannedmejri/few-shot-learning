# Results Summary

_Generated automatically from results.csv by make_plots.py._

_All numbers are conditional on the data and hyperparameters described in README.md._


## Table A — Accuracy by Encoder × Method × Shots (balanced queries, split: test)

| encoder | method | shots=1 | shots=5 | shots=10 | shots=20 |
|---|---|---|---|---|---|
| clip_b16 | logreg | 0.407 ± 0.007 | 0.649 ± 0.005 | 0.744 ± 0.004 | 0.816 ± 0.003 |
| clip_b16 | majority | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 |
| clip_b16 | proto | 0.386 ± 0.007 | 0.562 ± 0.005 | 0.611 ± 0.004 | 0.638 ± 0.004 |
| clip_b16 | random | 0.126 ± 0.002 | 0.126 ± 0.002 | 0.126 ± 0.002 | 0.126 ± 0.002 |
| clip_b16 | tim | 0.460 ± 0.008 | 0.669 ± 0.005 | 0.752 ± 0.004 | 0.812 ± 0.003 |
| clip_b16 | tim_cond | 0.269 ± 0.008 | 0.642 ± 0.005 | 0.741 ± 0.004 | 0.810 ± 0.003 |
| dinov2_s | logreg | 0.433 ± 0.007 | 0.729 ± 0.004 | 0.819 ± 0.003 | 0.878 ± 0.003 |
| dinov2_s | majority | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 |
| dinov2_s | proto | 0.397 ± 0.007 | 0.660 ± 0.005 | 0.734 ± 0.004 | 0.778 ± 0.004 |
| dinov2_s | random | 0.126 ± 0.002 | 0.126 ± 0.002 | 0.126 ± 0.002 | 0.126 ± 0.002 |
| dinov2_s | tim | 0.531 ± 0.010 | 0.782 ± 0.005 | 0.847 ± 0.003 | 0.888 ± 0.003 |
| dinov2_s | tim_cond | 0.159 ± 0.005 | 0.673 ± 0.009 | 0.827 ± 0.004 | 0.881 ± 0.003 |
| resnet50 | logreg | 0.415 ± 0.007 | 0.673 ± 0.005 | 0.764 ± 0.004 | 0.829 ± 0.003 |
| resnet50 | majority | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 | 0.125 ± 0.000 |
| resnet50 | proto | 0.400 ± 0.007 | 0.614 ± 0.005 | 0.671 ± 0.004 | 0.707 ± 0.003 |
| resnet50 | random | 0.126 ± 0.002 | 0.126 ± 0.002 | 0.126 ± 0.002 | 0.126 ± 0.002 |
| resnet50 | tim | 0.475 ± 0.009 | 0.715 ± 0.006 | 0.787 ± 0.004 | 0.834 ± 0.003 |
| resnet50 | tim_cond | 0.252 ± 0.007 | 0.627 ± 0.008 | 0.765 ± 0.005 | 0.831 ± 0.003 |


## Table B — Transductive Gain vs Prototype (balanced queries)

| encoder | method | shots | Δacc (mean±CI) | win_rate | wilcoxon_p |
|---|---|---|---|---|---|
| clip_b16 | logreg | 1 | 0.022 ± 0.002 | 0.85 | 0.0000 |
| clip_b16 | logreg | 5 | 0.087 ± 0.004 | 0.99 | 0.0000 |
| clip_b16 | logreg | 10 | 0.132 ± 0.004 | 1.00 | 0.0000 |
| clip_b16 | logreg | 20 | 0.178 ± 0.004 | 1.00 | 0.0000 |
| clip_b16 | tim | 1 | 0.074 ± 0.006 | 0.93 | 0.0000 |
| clip_b16 | tim | 5 | 0.108 ± 0.005 | 0.99 | 0.0000 |
| clip_b16 | tim | 10 | 0.141 ± 0.004 | 1.00 | 0.0000 |
| clip_b16 | tim | 20 | 0.175 ± 0.004 | 1.00 | 0.0000 |
| clip_b16 | tim_cond | 1 | -0.117 ± 0.007 | 0.02 | 0.0000 |
| clip_b16 | tim_cond | 5 | 0.081 ± 0.004 | 0.98 | 0.0000 |
| clip_b16 | tim_cond | 10 | 0.130 ± 0.004 | 1.00 | 0.0000 |
| clip_b16 | tim_cond | 20 | 0.172 ± 0.004 | 1.00 | 0.0000 |
| dinov2_s | logreg | 1 | 0.037 ± 0.004 | 0.84 | 0.0000 |
| dinov2_s | logreg | 5 | 0.069 ± 0.004 | 0.98 | 0.0000 |
| dinov2_s | logreg | 10 | 0.086 ± 0.003 | 1.00 | 0.0000 |
| dinov2_s | logreg | 20 | 0.100 ± 0.003 | 1.00 | 0.0000 |
| dinov2_s | tim | 1 | 0.134 ± 0.008 | 0.98 | 0.0000 |
| dinov2_s | tim | 5 | 0.122 ± 0.005 | 1.00 | 0.0000 |
| dinov2_s | tim | 10 | 0.114 ± 0.004 | 1.00 | 0.0000 |
| dinov2_s | tim | 20 | 0.110 ± 0.003 | 1.00 | 0.0000 |
| dinov2_s | tim_cond | 1 | -0.238 ± 0.007 | 0.00 | 0.0000 |
| dinov2_s | tim_cond | 5 | 0.013 ± 0.007 | 0.58 | 0.0002 |
| dinov2_s | tim_cond | 10 | 0.093 ± 0.004 | 1.00 | 0.0000 |
| dinov2_s | tim_cond | 20 | 0.103 ± 0.003 | 1.00 | 0.0000 |
| resnet50 | logreg | 1 | 0.015 ± 0.003 | 0.70 | 0.0000 |
| resnet50 | logreg | 5 | 0.059 ± 0.003 | 0.98 | 0.0000 |
| resnet50 | logreg | 10 | 0.093 ± 0.003 | 1.00 | 0.0000 |
| resnet50 | logreg | 20 | 0.123 ± 0.003 | 1.00 | 0.0000 |
| resnet50 | tim | 1 | 0.075 ± 0.006 | 0.91 | 0.0000 |
| resnet50 | tim | 5 | 0.101 ± 0.004 | 1.00 | 0.0000 |
| resnet50 | tim | 10 | 0.116 ± 0.004 | 1.00 | 0.0000 |
| resnet50 | tim | 20 | 0.128 ± 0.003 | 1.00 | 0.0000 |
| resnet50 | tim_cond | 1 | -0.148 ± 0.007 | 0.01 | 0.0000 |
| resnet50 | tim_cond | 5 | 0.013 ± 0.006 | 0.61 | 0.0000 |
| resnet50 | tim_cond | 10 | 0.093 ± 0.004 | 0.99 | 0.0000 |
| resnet50 | tim_cond | 20 | 0.125 ± 0.003 | 1.00 | 0.0000 |


## Table C — Accuracy Under Imbalanced Queries (shots=5)

| encoder | method | balanced | α=1.0 | α=0.3 |
|---|---|---|---|---|
| clip_b16 | logreg | 0.649 ± 0.005 | 0.650 ± 0.008 | 0.636 ± 0.011 |
| clip_b16 | majority | 0.125 ± 0.000 | 0.340 ± 0.010 | 0.543 ± 0.019 |
| clip_b16 | proto | 0.562 ± 0.005 | 0.566 ± 0.009 | 0.552 ± 0.013 |
| clip_b16 | random | 0.126 ± 0.002 | 0.123 ± 0.003 | 0.126 ± 0.003 |
| clip_b16 | tim | 0.669 ± 0.005 | 0.531 ± 0.008 | 0.388 ± 0.011 |
| clip_b16 | tim_cond | 0.642 ± 0.005 | 0.687 ± 0.010 | 0.729 ± 0.014 |
| dinov2_s | logreg | 0.729 ± 0.004 | 0.728 ± 0.007 | 0.722 ± 0.010 |
| dinov2_s | majority | 0.125 ± 0.000 | 0.340 ± 0.010 | 0.543 ± 0.019 |
| dinov2_s | proto | 0.660 ± 0.005 | 0.661 ± 0.008 | 0.653 ± 0.012 |
| dinov2_s | random | 0.126 ± 0.002 | 0.123 ± 0.003 | 0.126 ± 0.003 |
| dinov2_s | tim | 0.782 ± 0.005 | 0.614 ± 0.009 | 0.439 ± 0.012 |
| dinov2_s | tim_cond | 0.673 ± 0.009 | 0.763 ± 0.011 | 0.841 ± 0.013 |
| resnet50 | logreg | 0.673 ± 0.005 | 0.671 ± 0.008 | 0.664 ± 0.010 |
| resnet50 | majority | 0.125 ± 0.000 | 0.340 ± 0.010 | 0.543 ± 0.019 |
| resnet50 | proto | 0.614 ± 0.005 | 0.612 ± 0.007 | 0.610 ± 0.011 |
| resnet50 | random | 0.126 ± 0.002 | 0.123 ± 0.003 | 0.126 ± 0.003 |
| resnet50 | tim | 0.715 ± 0.006 | 0.576 ± 0.009 | 0.424 ± 0.012 |
| resnet50 | tim_cond | 0.627 ± 0.008 | 0.706 ± 0.012 | 0.770 ± 0.015 |


## Narrative Findings (auto-generated)

- **clip_b16, 5-shot**: `tim` improved over `proto` by 0.108 ± 0.005 acc pts (p<0.05).
- **clip_b16, 5-shot**: `logreg` improved over `proto` by 0.087 ± 0.004 acc pts (p<0.05).
- **clip_b16, 10-shot**: `tim` improved over `proto` by 0.141 ± 0.004 acc pts (p<0.05).
- **clip_b16, 10-shot**: `logreg` improved over `proto` by 0.132 ± 0.004 acc pts (p<0.05).
- **dinov2_s, 5-shot**: `tim` improved over `proto` by 0.122 ± 0.005 acc pts (p<0.05).
- **dinov2_s, 5-shot**: `logreg` improved over `proto` by 0.069 ± 0.004 acc pts (p<0.05).
- **dinov2_s, 10-shot**: `tim` improved over `proto` by 0.114 ± 0.004 acc pts (p<0.05).
- **dinov2_s, 10-shot**: `logreg` improved over `proto` by 0.086 ± 0.003 acc pts (p<0.05).
- **resnet50, 5-shot**: `tim` improved over `proto` by 0.101 ± 0.004 acc pts (p<0.05).
- **resnet50, 5-shot**: `logreg` improved over `proto` by 0.059 ± 0.003 acc pts (p<0.05).
- **resnet50, 10-shot**: `tim` improved over `proto` by 0.116 ± 0.004 acc pts (p<0.05).
- **resnet50, 10-shot**: `logreg` improved over `proto` by 0.093 ± 0.003 acc pts (p<0.05).