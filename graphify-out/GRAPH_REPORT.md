# Graph Report - fewshots  (2026-10-31)

## Corpus Check
- 13 files · ~10,078 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 10 file(s) not represented in the graph (top: .npy 7, (none) 1, .ipynb 1)

## Summary
- 225 nodes · 440 edges · 11 communities (7 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c1be4a97`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_fewshot.py
- run_experiments.py
- numpy
- few-shot-medical-transductive
- extract_features.py
- CLAUDE.md — few-shot-medical-transductive
- make_ep
- compute_metrics
- TestRunExperimentsSafety
- .claude/CLAUDE.md

## God Nodes (most connected - your core abstractions)
1. `CLAUDE.md — few-shot-medical-transductive` - 18 edges
2. `sample_episode()` - 16 edges
3. `make_ep()` - 16 edges
4. `compute_metrics()` - 15 edges
5. `predict_tim()` - 13 edges
6. `predict_proto()` - 12 edges
7. `Episode` - 11 edges
8. `run_experiment()` - 11 edges
9. `TestSampler` - 11 edges
10. `predict_logreg()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `make_ep()` --calls--> `sample_episode()`  [EXTRACTED]
  tests/test_fewshot.py → fewshot.py
- `get_method_preds()` --calls--> `predict_proto()`  [EXTRACTED]
  run_experiments.py → fewshot.py
- `get_method_preds()` --calls--> `predict_tim()`  [EXTRACTED]
  run_experiments.py → fewshot.py
- `run_episodes()` --calls--> `predict_tim()`  [EXTRACTED]
  tune.py → fewshot.py
- `get_method_preds()` --calls--> `predict_tim_cond()`  [EXTRACTED]
  run_experiments.py → fewshot.py

## Import Cycles
- None detected.

## Communities (11 total, 4 thin omitted)

### Community 0 - "test_fewshot.py"
Cohesion: 0.07
Nodes (11): aggregate_episodes(), ci95(), Episode, paired_improvement(), predict_majority(), predict_proto(), predict_random(), predict_tim() (+3 more)

### Community 1 - "run_experiments.py"
Cohesion: 0.11
Nodes (14): predict_logreg(), sample_episode(), get_method_preds(), load_features(), load_hparams(), main(), make_smoke_features(), make_smoke_hparams() (+6 more)

### Community 2 - "numpy"
Cohesion: 0.17
Nodes (11): ci95_series(), fig_acc_vs_shots(), fig_confusion_matrix(), fig_imbalance(), fig_transductive_gain(), fmt(), main(), make_summary() (+3 more)

### Community 3 - "few-shot-medical-transductive"
Cohesion: 0.07
Nodes (27): Caveats, Data, Encoders (all frozen, `.eval()`, `torch.no_grad()`), Episode Protocol, Experiment A — Accuracy by Encoder × Method × Shots (balanced queries), Experiment B — Transductive Gain vs Prototype, Experiment C — Imbalanced Query Sets, few-shot-medical-transductive (+19 more)

### Community 4 - "extract_features.py"
Cohesion: 0.10
Nodes (6): build_pool_indices(), extract_features(), load_encoder(), main(), MedMNISTFeatureDataset, _run_smoke()

### Community 5 - "CLAUDE.md — few-shot-medical-transductive"
Cohesion: 0.11
Nodes (18): Caveats the README must state, CLAUDE.md — few-shot-medical-transductive, Commands, Data, Encoders (all frozen, `.eval()`, `torch.no_grad()`), Episode protocol, Figures, Golden rules (never violate) (+10 more)

### Community 6 - "make_ep"
Cohesion: 0.18
Nodes (4): make_ep(), make_pool(), make_query(), TestSampler

## Knowledge Gaps
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `sample_episode()` connect `run_experiments.py` to `test_fewshot.py`, `numpy`, `make_ep`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Should `test_fewshot.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07400555041628122 - nodes in this community are weakly interconnected._
- **Why does `compute_metrics()` connect `compute_metrics` to `test_fewshot.py`, `run_experiments.py`, `numpy`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Should `run_experiments.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1140819964349376 - nodes in this community are weakly interconnected._
- **Why does `make_ep()` connect `make_ep` to `test_fewshot.py`, `run_experiments.py`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Should `few-shot-medical-transductive` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._
- **Should `extract_features.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10461538461538461 - nodes in this community are weakly interconnected._