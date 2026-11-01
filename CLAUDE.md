# CLAUDE.md — few-shot-medical-transductive

## Project in one paragraph
Reproducible empirical study: **Transductive Few-Shot Classification of Medical Images with Frozen Foundation-Model Features.**
Question: (1) does transductive inference help few-shot medical image classification when every encoder is frozen, and (2) does it degrade when the query set is class-imbalanced (Dirichlet-skewed)?
We apply *known* methods (prototype, logistic-regression probe, TIM-style). **Do not claim novelty** anywhere (README, report, code comments).
Negative or mixed results are acceptable and must be reported as-is.

## Golden rules (never violate)
1. **No fabricated numbers.** Every number in README / report / summary comes from `results/results.csv` through a script. If a run did not happen, write "not run". Never paste expected or remembered values.
2. **No test leakage.** Support sets come from the train pool only. Hyperparameters are tuned on val (val-drawn episodes) and then frozen in `configs/best_hparams.json`. The test split is touched only by the final run (`run_experiments.py --final`).
3. **No invented citations.** Papers are listed in `REFERENCES.md` with status `UNVERIFIED` until the user checks Google Scholar. Never invent venues, pages, DOIs or numbers from papers.
4. **Paired design.** The same episodes (same support set, same query set) are evaluated by every method. Method choice must never change the RNG stream.
5. **Frozen encoders.** No fine-tuning and no episodic training in the core study. (Optional extras only after everything else is done; see Phase 6.)
6. Keep the repo small. Don't add frameworks, configs systems, or abstractions the plan doesn't need.

## Repo layout
```
few-shot-medical-transductive/
├── README.md
├── CLAUDE.md
├── REFERENCES.md
├── requirements.txt
├── extract_features.py      # encoders -> features/*.npy (+ pool indices)
├── fewshot.py               # sampler, prototype, logreg, TIM-style, metrics
├── tune.py                  # val-only hyperparameter search -> configs/best_hparams.json
├── run_experiments.py       # loops encoder/method/shots/imbalance/episodes -> results/results.csv
├── make_plots.py            # tables (markdown) + figures + results/summary.md
├── tests/                   # pytest
├── notebooks/demo.ipynb     # Colab-runnable end to end
├── configs/best_hparams.json
├── features/                # {dataset}_{encoder}_{pool|val|test}_{X|Y}.npy, pool_idx.npy (gitignored if large)
├── results/results.csv, summary.md
└── figures/
```

## Data
- `medmnist` (v3+), MedMNIST+ with `size=224`. **Check download size first** (`medmnist` INFO / HTTP HEAD / free disk). If too large or slow, fall back to `size=128` via `--size 128` and record it in the README. For big files use `mmap_mode='r'` if supported.
- BloodMNIST (8 classes) first. PathMNIST (9 classes) only after BloodMNIST is complete. PathMNIST test comes from a different clinical center: report the val→test accuracy gap as the domain-shift angle.
- **Pool**: 500 images per class sampled once from train with a fixed seed; save indices to `features/{dataset}_pool_idx.npy`. Never extract the full train set. If a class has <500 train images, use all of them and log it.
- Val and test: extract fully.

## Encoders (all frozen, `.eval()`, `torch.no_grad()`)
| key | model | feature | per-encoder preprocessing |
|---|---|---|---|
| `resnet50` | torchvision ResNet50, ImageNet weights | global-avg-pool (2048-d) | ImageNet mean/std |
| `dinov2_s` | `facebookresearch/dinov2` `dinov2_vits14` | CLS (384-d) | ImageNet mean/std |
| `clip_b16` | OpenCLIP or HF CLIP ViT-B/16 (OpenAI weights) | image embedding (512-d) | **CLIP mean/std** `[0.4815,0.4578,0.4082]/[0.2686,0.2613,0.2758]` — do NOT reuse ImageNet stats |
| `biomedclip` (optional) | `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-224-vit-b-16` via open_clip | image embedding | its own preprocess |

- Images are already 224×224 (or 128→resize to 224). Convert grayscale to RGB if needed.
- L2-normalize features before saving (float32). Save `X` and `Y` per split.
- CPU is acceptable for extraction (~9k images per encoder for BloodMNIST: pool 4000 + val + test). Use GPU if available, `--device` flag, batch size flag, fp16 autocast optional on GPU.
- Extraction is idempotent: skip existing files unless `--force`.

## Episode protocol
- An **episode** = (support seed, query set). Default **300 episodes** (flag `--episodes`, ≥300 for final).
- Support: K ∈ {1,5,10,20} shots per class drawn **without replacement** from the pool with `np.random.default_rng(seed)`.
- Query: `n_total = C × 30` fixed across imbalance levels (so query size doesn't change). Drawn from the query split (val during tuning, test for final) using an RNG derived from `(episode_id, imbalance_level)` that is **independent of method and shots**.
  - Balanced: 30 per class.
  - Imbalanced: `p ~ Dirichlet(a·1_C)`, `counts ~ Multinomial(n_total, p)`; levels `a ∈ {1.0, 0.3}` (optionally 0.1). Sample **without replacement** inside a class; use `replace=True` only if a class has fewer images than requested (log how often).
- Sanity baselines, reported as rows: **random** (1/C expected, also simulate) and **majority-class** (predict the most frequent class of that query set).

## Methods (inputs: L2-normalized features)
1. `proto` — nearest prototype by cosine (mean of support features, re-normalized). No hyperparameters.
2. `logreg` — sklearn multinomial `LogisticRegression` fit on support only; tune `C ∈ {0.01,0.1,1,10,100}` on val.
3. `tim` — simplified TIM (cosine classifier with temperature `tau`, prototypes optimized with Adam):
   `loss = CE(support) − λ·(H(Y) − α·H(Y|X))`, `steps=50`, `lr=1e-2`, prototypes initialized from support means and re-normalized each step.
4. `tim_cond` (**ablation, diagnoses Experiment C**) — same as `tim` but marginal-entropy term removed (`loss = CE + λ·α·H(Y|X)`). If degradation under imbalance mostly disappears here, the marginal term is the cause. Report this honestly either way.

Reference TIM implementation (keep semantics, may be vectorized):
```python
def tim_predict(zs, ys, zq, C, tau=20., lam=0.5, alpha=1.0, steps=50, lr=1e-2, use_marginal=True):
    P = torch.stack([zs[ys==c].mean(0) for c in range(C)]).clone().requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)
    for _ in range(steps):
        Pn = F.normalize(P, dim=1)
        ce = F.cross_entropy(tau * zs @ Pn.T, ys)
        pq = (tau * zq @ Pn.T).softmax(1)
        cond = -(pq * (pq + 1e-8).log()).sum(1).mean()          # H(Y|X) minimize
        pm = pq.mean(0)
        marg = -(pm * (pm + 1e-8).log()).sum()                  # H(Y) maximize
        loss = ce - lam * ((marg if use_marginal else 0.) - alpha * cond)
        opt.zero_grad(); loss.backward(); opt.step()
    return (zq @ F.normalize(P, dim=1).T).argmax(1)
```
Do not meta-train anything. Do not change method definitions after seeing test results.

## Hyperparameter tuning (`tune.py`, val only)
- Per (dataset, encoder, shots): 100 val episodes, balanced val queries, support from the train pool.
- Grids: `tau ∈ {5,10,20,30}`, `λ ∈ {0.1,0.3,0.5,1,2}`, `α ∈ {0.5,1}`, logreg `C` as above. Select by mean accuracy. Ties → smaller λ.
- Write `configs/best_hparams.json` (with a timestamp and the val-episode seeds). `run_experiments.py --final` must refuse to run if this file is missing.
- Hyperparameters tuned on balanced val are used for all imbalance levels (that is the realistic deployment assumption). Optional, clearly labeled ablation: re-tune on imbalanced val.

## Metrics (own vectorized implementation, unit-tested against sklearn)
- Accuracy, **macro-F1**, **balanced accuracy**.
- Under imbalance some classes may be absent from a query set: compute macro-F1 and balanced accuracy **over classes present in that query set's ground truth** (state this in the README).
- Aggregation: mean and **95% CI** over episodes (t-interval, or bootstrap; pick one and say which).
- **Improvement metrics** (Experiment B/C), computed per episode then aggregated (paired):
  - `Δacc = acc(method) − acc(proto)`, `ΔmacroF1`, `Δbalacc`, with paired 95% CI;
  - relative error reduction `(err_proto − err_method)/err_proto`;
  - win rate (% episodes where method > proto) and a paired test (Wilcoxon signed-rank) — report p-values but don't over-interpret;
  - degradation under imbalance: `metric(balanced) − metric(a)` per method, plus `Δ vs proto` at each imbalance level (does the transductive gain shrink or flip sign?).

## Results format
`results/results.csv` is **long format, one row per (dataset, encoder, method, shots, imbalance, episode)** with columns:
`dataset,encoder,method,shots,imbalance,episode,split,n_query,acc,macro_f1,bal_acc`.
Everything else (tables, plots, summary) is derived from it. Include the sanity rows (`random`, `majority`).
`make_plots.py` writes: Table A (encoder × method × shots, mean ± CI), Table B (gain per shot, paired CI), Table C (imbalance), figures, and `results/summary.md` which states findings **conditionally and quantitatively** ("TIM improved 5-shot DINOv2 accuracy by X ± Y pts" only if the numbers say so).

## Figures
1. Accuracy and macro-F1 vs shots per encoder/method (95% CI error bars).
2. Bar chart: transductive gain over prototype at each shot.
3. Line plot: accuracy (and gain) vs imbalance level per method — the key figure.
4. One confusion-matrix pair (DINOv2, 5-shot, `tim` vs `proto`).
PNG, readable fonts, labeled axes, no seaborn dependency required.

## Testing (pytest, runs in <1 min on CPU with synthetic features)
Required tests:
- sampler: support ⊂ pool and has exactly K per class; query indices belong to the intended split; Dirichlet counts sum to `n_total`; same seed → identical episodes; changing method doesn't change episodes.
- metrics vs `sklearn` on random labels (including absent-class case).
- proto reaches ~100% on well-separated synthetic Gaussians; `tim` ≥ `proto` on a synthetic set where the query cloud is shifted from support; `tim` with `λ=0` equals CE-only optimization.
- `run_experiments.py --final` refuses without `best_hparams.json`; tuning code path never opens test files (assert on split names).
- Smoke run: `python run_experiments.py --smoke` uses synthetic features and finishes in seconds.
Run `pytest -q` after every meaningful change; don't proceed to the next phase with failing tests.

## Commands
```
pip install -r requirements.txt
python extract_features.py --dataset bloodmnist --encoders resnet50 dinov2_s clip_b16 --size 224 --device cuda
python tune.py --dataset bloodmnist
python run_experiments.py --dataset bloodmnist --final --episodes 300
python make_plots.py
pytest -q
```

## Phases and acceptance checks
- **Phase 0** scaffold, `requirements.txt`, tests for sampler/metrics, smoke mode. ✅ when `pytest` passes.
- **Phase 1** feature extraction BloodMNIST (DINOv2 + ResNet50 first). ✅ files exist, shapes/class counts logged, features are unit-norm.
- **Phase 2** proto + logreg, val tuning, first accuracy-vs-shots table.
- **Phase 3** TIM + `tim_cond`, tune λ/τ on val, Experiment B.
- **Phase 4** Experiment C, CLIP, confusion matrices; PathMNIST if time allows.
- **Phase 5** README (setup, protocol, results tables pulled from CSV, honest caveats), 1–2 page report draft, demo notebook.
- **Phase 6 (optional, only if everything above is done)** full fine-tuning row for one encoder; PCA efficiency table.
Minimum viable deliverable: BloodMNIST, DINOv2 + ResNet50, 3 methods, balanced queries, one table, two plots, README.

## Caveats the README must state
- Frozen-feature, known-method empirical study; no novelty claim.
- Results may be negative or mixed; DINOv2 vs CLIP ranking on stain-based histology can surprise.
- Hyperparameters tuned on balanced val and not re-tuned for imbalance (unless the labeled ablation was run).
- Metric conventions under absent classes; query size fixed; pool size 500/class; image size used (224 or 128).

## Style
Python 3.10+, type hints on public functions, short docstrings, seeds exposed as flags, no global state, logging via `print`/`logging` (no silent failures). Commit-sized changes; summarize what changed and what was run at the end of each phase.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
