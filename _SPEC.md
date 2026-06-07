# Specification: Continuous-Feature Representation Experiment (v3)

## 1. Overview

This experiment extends the pluto-toy project to test whether richer numerical
feature representations — and larger model capacity — can produce stable
*category strain* (tension) in a small transformer. The existing v2 sweep uses
a coarse 5-level ordinal feature space (0–4 mapped to words like "small",
"large"). This project replaces that with continuous numerical features in two
formats (digit strings and spelled-out words), adds a model-size sweep up to
~1M parameters, and measures whether genuine tension (both "planet" and "dwarf"
labels simultaneously active for P10) becomes a stable outcome rather than a
rare tail event.

The hypothesis under test: coarse ordinal features may make category boundaries
too easy to resolve, collapsing tension into clean label swaps. Continuous
features create genuinely graded similarity, potentially sustaining tension as a
stable equilibrium.

## 2. Users and context

**Primary users:** Two collaborators (Norman and co-worker) running experiments
on laptops and Google Colab. Eventually shared publicly as a teaching artifact
and supplement to a methods paper.

**Context:** This is the third iteration of the pluto-toy experiment (v1: three
probes, v2: dual-label probe + rote control, v3: continuous features). It runs
alongside the existing v2 code — parallel scripts, shared site and notebooks.

## 3. Functional requirements

### 3.1 Continuous feature generation

**3.1.1 Feature distributions.** Replace the ordinal 0–4 system with
continuous-valued features drawn from parameterized Gaussian distributions.
Each category prototype has a mean and standard deviation per axis:

| Category  | Mass (kg)     | Diameter (km)   | Orbit (AU)      |
|-----------|---------------|-----------------|-----------------|
| planet    | N(350, 50)    | N(12000, 2000)  | N(5.0, 2.0)     |
| asteroid  | N(0.5, 0.2)   | N(50, 20)       | N(3.0, 1.0)     |
| comet     | N(0.01, 0.005)| N(10, 5)        | N(40, 15)       |
| moon      | N(5, 2)       | N(1500, 500)    | N(0.5, 0.2)     |

These are illustrative defaults. All distribution parameters (means and
standard deviations per category per axis) must be configurable via CLI flags
or a config dict so they can be swept over.

**3.1.2 P10 parameterization.** P10's mass-axis mean is the primary edge
parameter (analogous to the existing `--p10-edge`). Default: mass drawn from
N(40, 5), placing P10 in the ambiguous zone between planet bulk (350) and
E*-region (~38). Diameter and orbit pinned to planet-edge values (large
diameter, distant orbit) as in the current design.

**3.1.3 E* placement.** Same two-corner design as v2:
- **Overlap region:** E* features drawn near P10's neighborhood (mass ~38,
  diameter ~planet-edge, orbit ~distant).
- **Disjoint rote-control corner:** E* features drawn from a region far from
  all canonical prototypes, matching v2's direction: mass=huge, diameter=tiny,
  orbit=near (e.g., mass ~500, diameter ~5, orbit ~0.01). Future work may
  explore more extreme separations.
- `n_eris_rote` controls the corner-mix split, as in v2.

**3.1.4 Two representation formats.** Each continuous value is rendered as text
in one of two formats:

- **Digit format:** `"P10 has mass 38.08 kg diameter 11500 km orbit 39.2 AU ."`
  Numbers are rounded to a configurable number of decimal places (default: 2
  for mass, 0 for diameter, 1 for orbit).
- **Spelled-out format:** `"P10 has mass thirty-eight point zero eight
  kilograms diameter eleven thousand five hundred kilometers orbit thirty-nine
  point two astronomical units ."` Uses a deterministic number-to-words library
  (e.g., `num2words`). Hyphenated compounds (e.g., "thirty-eight") are split
  into separate tokens ("thirty", "eight") so the tokenizer uses a fixed set of
  English number words rather than creating unique compound tokens per value.

The representation format is a sweep axis (`--representation ordinal|digit|spelled`).
The ordinal format is retained as a control condition matching v2.

**3.1.5 Units.** Each axis has a unit label (kg, km, AU) appended in both
formats. This provides semantic grounding for the numbers. Units are fixed per
axis (not swept).

### 3.1a Tokenization strategy

**3.1a.1 Character-level digit tokenization.** The existing v2 word-level
tokenizer produces one token per unique whitespace-delimited word. Continuous
numbers would cause vocabulary explosion (every unique sampled value becomes
its own token), making the embedding matrix dominate parameter count and
producing sparse gradients for most number tokens.

Instead, the v3 tokenizer uses **character-level tokenization for numeric
strings**: the digit string "38.08" is tokenized as `["3", "8", ".", "0", "8"]`.
Non-numeric tokens (entity names, category words, units, structural words like
"has", "is", "a") remain word-level. This produces a small, fixed vocabulary:
digits 0–9, the decimal point, plus the existing word tokens (~70–80 total
tokens across all conditions).

**3.1a.2 Spelled-out tokenization.** For the spelled-out format, each word in
the number-words output is a separate token. Hyphenated forms are split (e.g.,
"thirty-eight" becomes two tokens: "thirty", "eight"). The spelled-out
vocabulary adds ~30 English number words (zero through nineteen, twenty, thirty,
..., hundred, thousand, point) to the base vocabulary. Total vocabulary for
spelled-out: ~100 tokens.

**3.1a.3 Vocabulary consistency across conditions.** To enable fair
parameter-count comparison, the tokenizer vocabulary is fixed across all three
representation formats for a given sweep. The vocabulary is the *union* of
tokens needed by all formats: digits 0–9, ".", all English number words, all
unit strings, plus the shared word tokens. This means every condition uses
the same embedding matrix size (~100 tokens), and parameter-count matching
(§3.3.2) is not confounded by vocabulary variation.

**3.1a.4 Implementation.** A new `TokenizerV3` class wraps the existing
`WordTokenizer` with a preprocessing step that splits digit strings into
characters. The existing `WordTokenizer` is not modified. `TokenizerV3` is used
by all v3 scripts; v2 scripts continue to use `WordTokenizer`.

### 3.2 Context-window isolation

Spelled-out numbers produce longer sequences than digit or ordinal formats.
To isolate the effect of representation from the effect of sequence length and
model capacity, two isolation strategies are implemented:

**3.2.1 Strategy A — Pad to match (default).** All conditions use the same
`block_size` (set to whatever the spelled-out condition requires, likely
128–256). Shorter sequences (digit, ordinal) are not explicitly padded at the
sentence level. Instead, the training loop works as in v2: the corpus is
concatenated into a flat token stream, and `get_batch` samples random
contiguous windows of `block_size` tokens. Shorter representations simply
produce a denser token stream (more sentences per window). The model
architecture (including positional embedding size) is identical across
conditions. This is the default isolation strategy.

**3.2.2 Strategy B — Match parameter count (optional extension).** Let
`block_size` vary per condition (64 for ordinal, ~96 for digit, ~192 for
spelled-out). Adjust `n_embd` and `n_layer` so total parameter count is held
approximately constant (within +/-10%) across conditions. Since vocabulary is
fixed across conditions (§3.1a.3), parameter-count matching only needs to
account for positional embeddings and transformer body. This isolates capacity
but not attention pattern length.

The isolation strategy is a sweep axis (`--isolation pad|match-params`).
Strategy A (pad) is the default. Strategy B is run as a secondary analysis
after the primary sweep completes. Pilot cells (5–10) comparing the two
strategies should be run before committing to a full factorial cross of
Strategy B to assess whether the axis is informative.

**3.2.3 Changes to shared training code.** Supporting variable `block_size`
requires changes to `train.py` (accepting `block_size` as a parameter rather
than only from CLI args) and `model.py` (no changes needed — `block_size` is
already a config parameter). The backward-compatibility contract: existing v2
CLI invocations with existing flags must produce bit-identical results. New
code paths are only activated when v3-specific flags are passed.

### 3.3 Model-size sweep

**3.3.1 Size tiers.** Three tiers, with the largest subject to a feasibility
check:
- **Small:** ~50K params (n_embd=32, n_layer=4) — matches v2.
- **Medium:** ~200K params (n_embd=64, n_layer=6) — matches v2.
- **Large:** ~1M params (n_embd=128, n_layer=8) — new. Feasibility determined
  by wall-clock time on the user's machine and Google Colab (target: single
  cell under ~10 minutes).

**3.3.2 Parameter-count matching.** Under isolation Strategy B (§3.2.2), the
`n_embd`/`n_layer` values for each tier are adjusted per representation format
to hold total parameters approximately constant (within +/-10%). The parameter
count formula for this model (with weight tying) is:

```
params = vocab_size * n_embd           # token embeddings (shared with LM head)
       + block_size * n_embd           # positional embeddings
       + n_layer * (                   # per transformer block:
           3 * n_embd * n_embd         #   QKV projection
           + n_embd * n_embd           #   output projection
           + 4 * n_embd * n_embd       #   MLP fc
           + 4 * n_embd * n_embd       #   MLP proj
           + 4 * n_embd               #   2 LayerNorm + 2 MLP biases
         )
       + n_embd                        # final LayerNorm
```

Since vocabulary is fixed across conditions (§3.1a.3), the only variable is
`block_size`. The implementation searches over `n_embd` values (constraining
`n_embd % n_head == 0`) to find the closest match to the target parameter count
for each (block_size, n_layer) pair, and logs the actual count.

### 3.4 Numerical-comprehension diagnostic

**3.4.1 Purpose.** A co-measured diagnostic (not a gate) that assesses whether
the model has learned numerical proximity. Reported alongside every sweep cell
so readers can correlate comprehension with tension outcomes.

**3.4.2 Design.** For each trained checkpoint, probe whether the model assigns
higher P(same_category) to numerically-close held-out entities than to
numerically-distant ones. Concretely:

- Select one canonical planet (e.g., P9) and one canonical asteroid (e.g., A1)
  as test entities. These entities are *not* held out from training — they
  appear in the phase 1 corpus as normal. The diagnostic tests whether the
  model's category assignments are consistent with feature proximity, not
  whether it generalizes to unseen entities.
- After training, prompt `"{test_name} is a"` and record P(planet) and
  P(asteroid).
- A model that understands numerical proximity should assign high P(planet) to
  P9 (whose features are close to other planets) and low P(planet) to A1.
- Report a **comprehension score**: `P(correct_label | P9) -
  P(correct_label | A1)`. Ranges from -1 (inverted) to +1 (perfect);
  0 = chance.

**3.4.2a Representation-neutral alternative.** Because the label-probability
diagnostic tests different things across representations (ordinal proximity is
categorical, continuous proximity is numerical), a second diagnostic is
reported alongside it: the **embedding centroid distance** between P10's
learned embedding and the centroid of each category's embeddings. This is
representation-neutral and provides a complementary signal about whether the
model's internal geometry reflects category structure regardless of how
features are rendered as text.

**3.4.3 Relationship to tension results.** The comprehension score is reported
in the results table alongside tension classification and tension index. No
sweep cells are excluded based on comprehension score. If tension arises in
cells with low comprehension, this is flagged as an observation requiring a
mechanistic explanation — that explanation is deferred to analysis, not
pre-specified here.

### 3.5 Sweep design

**3.5.1 Axes.** The full factorial sweep crosses:

| Axis              | Values                                      | Count |
|-------------------|---------------------------------------------|-------|
| representation    | ordinal, digit, spelled                     | 3     |
| mode              | unlabeled, dwarf, planet                    | 3     |
| schedule          | canon-only, curriculum, mixed               | 3     |
| isolation         | pad (default); match-params (extension)     | 1–2   |
| model size        | small, medium, (large if feasible)          | 2–3   |
| n_eris_rote       | 0, 2, 4, 6 (with n_eris=6)                 | 4     |
| seed              | 0–9                                         | 10    |

**Ordinal control cells.** When representation=ordinal, the isolation axis is
collapsed (ordinal always uses block_size=64, the v2 default). This avoids
redundant cells.

**Default sweep (pad isolation only, 2 size tiers):**
- Non-ordinal: 2 repr × 3 mode × 3 sched × 1 iso × 2 size × 4 rote × 10 seed = 1440
- Ordinal control: 1 repr × 3 mode × 3 sched × 1 iso × 2 size × 4 rote × 10 seed = 720
- Default total: 2160 cells.

**Extended sweep (adds match-params isolation):**
- Non-ordinal match-params: 2 repr × 3 mode × 3 sched × 1 iso × 2 size × 4 rote × 10 seed = 1440
- Extended total: 3600 cells.

**Quick mode (`--quick`):** 3 seeds instead of 10, small model only,
pad isolation only. Total: 3 repr × 3 mode × 3 sched × 1 size × 4 rote × 3
seed = 324 cells. For pilot runs and development iteration.

**3.5.2 Wall-clock budget.** At ~3 minutes per small-model cell and ~8 minutes
per medium-model cell on CPU, the default sweep (2160 cells) would take ~150
hours sequentially on CPU. This is not a single-session run. Execution
strategies:

- **Quick mode on laptop:** 324 cells × ~3 min = ~16 hours. Feasible as an
  overnight run.
- **Default sweep on Colab GPU:** ~3–5x speedup expected. ~30–50 hours, spread
  across multiple sessions.
- **Phased execution:** Run representation conditions independently (each is
  ~720 cells) and aggregate. Each condition fits in a ~8-hour Colab session.
- **Target: no single invocation exceeds 8 hours.** The sweep script supports
  `--representations`, `--sizes`, and `--seeds` flags to run subsets.

**3.5.3 Reproducibility.** Every sweep cell logs: seed, all hyperparameters,
actual parameter count, wall-clock time, corpus hash, and Python/PyTorch
versions. The corpus hash is SHA-256 of `phase1.txt` and `phase2.txt`
concatenated (in that order), plus a separate hash of `entities.json`. Hashes
are stored in the sweep manifest per cell.

### 3.6 Probing

**3.6.1 Dual-label tension probe.** The existing tension probe (§ tension_probe.py)
is adapted for continuous features. The prompt format changes to match the
representation:

- Ordinal: `"P10 has mass small diameter large orbit distant . P10 is a"`
- Digit: `"P10 has mass 38.08 kg diameter 11500 km orbit 39.2 AU . P10 is a"`
- Spelled: `"P10 has mass thirty-eight point zero eight kilograms diameter
  eleven thousand five hundred kilometers orbit thirty-nine point two
  astronomical units . P10 is a"`

Calibration thresholds, 2×2 classification grid, and tension_index formula are
unchanged from v2.

**3.6.2 Baseline classifiers.** The existing four baselines (logreg, GNB, kNN,
hand-Bayes) are run on the continuous feature values (not the text
representations). This provides a representation-independent reference: what
would a shallow classifier predict given the underlying numbers? Because
baselines see raw numbers and not text, their results are **identical across
representation conditions** for the same (seed, mode, n_eris_rote) tuple.
Baselines are computed once per unique (seed, mode, n_eris_rote) combination
and shared across representation/isolation/model-size conditions to avoid
redundant computation.

**3.6.3 Comprehension diagnostic.** As specified in §3.4.

### 3.7 Aggregation and reporting

**3.7.1 Summary JSON.** Aggregate results into a JSON file matching the schema
of `results/tension_summary_2026-05-07.json`, extended with fields for
`representation`, `isolation`, `comprehension_score`, and `block_size`.

**3.7.2 Figures.** At minimum:
- Tension classification grid (representation × mode × schedule), analogous to
  the existing grid but with representation as a new axis.
- Corner-mix curves per representation format, comparable to the existing v2
  figure.
- Comprehension score vs. tension index scatter plot.
- Baseline classifier comparison, extended with continuous-feature baselines.

**3.7.3 Interactive site.** Extend `docs/index.html` with a new section or tab
for v3 results. The existing v2 content is preserved. New data is embedded via
the same `build_site.py` mechanism.

**3.7.4 Colab notebook.** Extend `tutorial-tension.ipynb` (or create a
companion `tutorial-continuous.ipynb` if the existing notebook becomes too
long) with cells demonstrating the continuous-feature pipeline, the
comprehension diagnostic, and the headline comparison figures.

### 3.8 Parallel codebase

All new scripts are separate from the existing v2 scripts:

| New file                    | Purpose                                         |
|-----------------------------|--------------------------------------------------|
| `generate_corpus_v3.py`     | Continuous-feature corpus generation              |
| `tokenizer_v3.py`           | Character-level digit tokenizer (TokenizerV3)     |
| `sweep_v3.py`               | v3 sweep orchestration                            |
| `tension_probe_v3.py`       | Adapted tension probe for continuous features     |
| `comprehension_probe.py`    | Numerical-comprehension diagnostic                |
| `aggregate_tension_v3.py`   | v3 aggregation                                    |
| `plot_tension_v3.py`        | v3 figures                                        |
| `baseline_classifier_v3.py` | Baselines on continuous features                  |
| `build_site_v3.py`          | Site builder incorporating v3 data                |

Shared infrastructure (`model.py`, `tokenizer.py`, `train.py`) is reused
without modification where possible. If modifications are needed (e.g.,
supporting variable `block_size`), they are backward-compatible.

## 4. Concrete examples

### 4.1 Corpus generation

**Input:**
```bash
python3 generate_corpus_v3.py --out-dir data_v3 --mode dwarf \
    --representation digit --n-eris 6 --n-eris-rote 2 --seed 42
```

**Output (data_v3/phase1.txt excerpt):**
```
P1 has mass 362.41 kg diameter 13102 km orbit 4.8 AU .
P1 is a planet .
...
P10 has mass 38.08 kg diameter 11500 km orbit 39.2 AU .
P10 is a planet .
the planets are P1 and P2 and P3 and P4 and P5 and P6 and P7 and P8 and P9 and P10 .
```

**Output (data_v3/phase2.txt excerpt):**
```
E1 has mass 37.22 kg diameter 10800 km orbit 38.5 AU .
E1 is a dwarf .
...
E5 has mass 0.001 kg diameter 105000 km orbit 0.01 AU .
E5 is a dwarf .
```

(E1–E4 in overlap region near P10; E5–E6 in disjoint rote-control corner.)

### 4.2 Spelled-out format

The same P10 entity in spelled-out format:
```
P10 has mass thirty-eight point zero eight kilograms diameter eleven thousand five hundred kilometers orbit thirty-nine point two astronomical units .
```

### 4.3 Tension probe output

```json
{
  "entity": "P10",
  "P_planet": 0.34,
  "P_dwarf": 0.28,
  "thr_planet": 0.31,
  "thr_dwarf": 0.25,
  "classification": "TENSION",
  "tension_index": 1.097,
  "comprehension_score": 0.72
}
```

### 4.4 Comprehension diagnostic output

```json
{
  "held_out_planet": "P9",
  "P_planet_for_planet": 0.61,
  "held_out_asteroid": "A1",
  "P_planet_for_asteroid": 0.08,
  "comprehension_score": 0.53
}
```

## 5. Failure modes and error handling

**5.1 Numerical tokenization failures.** If `num2words` produces unexpected
output for edge-case numbers (e.g., very small decimals like 0.001), the corpus
generator should raise an error with the offending value rather than silently
producing garbled text. Unit tests cover boundary values.

**5.2 `num2words` dependency.** `num2words` is required only for the
spelled-out representation. It is a hard dependency listed in
`requirements.txt`. If not installed, `generate_corpus_v3.py` raises
`ImportError` with a clear message at import time (not at first use). The digit
and ordinal representations do not require it, but the unified tokenizer
vocabulary (§3.1a.3) includes number words regardless, so `num2words` is needed
to build the vocabulary even for non-spelled conditions. If this coupling
proves inconvenient, it can be relaxed by hardcoding the number-word list.

**5.3 Context-window overflow.** If a spelled-out feature description exceeds
`block_size`, the corpus generator raises an error identifying the entity and
the sequence length. The fix is to increase `block_size` or reduce decimal
precision — not to silently truncate.

**5.4 Training divergence.** A training cell is considered failed if: (a) loss
becomes NaN at any step, or (b) loss exceeds 20.0 after step 500. The check is
implemented in `sweep_v3.py` by reading the final loss from the training log.
Failed cells are excluded from aggregation but reported in the manifest with
the failure reason.

**5.5 Sweep failures.** If a single cell fails (training divergence, probe
error, or any uncaught exception), the sweep logs the failure with full
hyperparameters and continues. Failed cells are excluded from aggregation but
reported in the manifest.

**5.6 Incompatible shared code.** If modifications to `model.py`, `tokenizer.py`,
or `train.py` break the existing v2 pipeline, this is a blocking bug. The v2
Quick Start (`python3 generate_corpus.py ... && python3 train.py ...`) must
continue to work. Backward compatibility is verified by running the v2 smoke
test (one canon-only cell) after any shared-code change.

**5.7 Irreproducible results.** If a sweep cell produces different results
across runs with the same seed, the corpus hash comparison will detect the
divergence. This should be investigated (likely a non-deterministic operation)
rather than ignored.

## 6. Non-functional requirements

**6.1 Performance.** A single cell (generate + train + probe) should complete
in under 3 minutes on CPU for the small model tier. The full default sweep
(2160 cells) is not a single-session run — see §3.5.2 for execution
strategies. No single invocation should exceed 8 hours.

**6.2 Dependencies.** New dependencies (e.g., `num2words` for spelled-out
format) are added to a `requirements.txt` or `pyproject.toml`. The project
uses `uv` for environment management.

**6.3 Compatibility.** Python 3.10+. PyTorch 2.0+. Must run on CPU, MPS
(Apple Silicon), and CUDA.

**6.4 Reproducibility.** Every sweep cell is fully reproducible given its seed
and hyperparameters. Corpus files are regenerated (not cached across seeds)
to ensure the hash check works. All random state flows through explicit
`random.Random(seed)` / `torch.manual_seed(seed)` calls.

## 7. Out of scope

- **Knowledge-graph representation.** See §10 (Future work / TODO).
- **Models larger than ~1M parameters.** If ~1M is feasible, it's the ceiling.
  Multi-GPU or HPC training is out of scope.
- **Automated mechanistic interpretation.** If tension arises in cells with low
  comprehension scores, the spec calls for flagging it — not for building an
  automated explanation pipeline.
- **Modifying existing v2 scripts.** v2 code (`generate_corpus.py`, `sweep.py`,
  `tension_probe.py`, etc.) is not modified. Shared infrastructure (`model.py`,
  `tokenizer.py`, `train.py`) may receive changes, but the backward-
  compatibility contract applies: existing v2 CLI invocations with existing
  flags must produce bit-identical results (§3.2.3).
- **New category types.** The category ontology (planet, asteroid, comet, moon,
  dwarf) is unchanged from v2.
- **Prompt engineering.** The probe prompt format is a direct adaptation of v2.
  Systematic prompt-format sweeps are not in scope.

## 8. Verification plan

**8.1 Ordinal control reproduces v2.** Run the ordinal condition through the
v3 sweep pipeline and compare tension classifications and tension indices
against the existing `results/tension_summary_2026-05-07.json`. Criterion:
same tension classification label (canon/swap/tension/nothing) for at least 90%
of matched cells, and tension_index within +/-0.15 for all matched cells.
Differences are investigated; if they stem from tokenizer changes (§3.1a), they
are documented but not blocking.

**8.2 Digit format learns categories.** Train a small model on digit-format
corpus with mode=planet (easiest condition). Probe P1: P(planet) should be
>0.5. If not, the model isn't learning from digit features at all.

**8.3 Spelled-out format learns categories.** Same as 8.2 but with spelled-out
format. This is the harder test — if it fails, the spelled-out condition is
still included in the sweep (per §3.4.3) but flagged.

**8.4 Context-window isolation produces different results.** Compare pad vs.
match-params isolation for the same representation and model-size tier. If
results are identical, the isolation axis is not informative (fine — report it).
If they differ, the isolation strategy matters and both must be reported.

**8.5 Comprehension diagnostic correlates with expectations.** For ordinal
(control) condition, comprehension score should be above 0.3 (the model can
learn ordinal categories from the existing v2-style corpus). If it falls below
0.3, the diagnostic itself may be miscalibrated and should be investigated
before interpreting cross-representation comparisons.

**8.6 End-to-end smoke test.** A single command runs one cell of each
representation format through the full pipeline (generate → train → probe →
aggregate) and produces a summary JSON. This should complete in under 5 minutes
on CPU.

## 9. Open questions

**9.1** What decimal precision best balances feature discriminability against
sequence length? Default is 2/0/1 (mass/diameter/orbit) but this may need
tuning. Start with defaults and adjust if comprehension diagnostics are
uniformly low.

**9.2** Should the feature distributions (§3.1.1) be calibrated to real solar
system values, or is the current illustrative parameterization sufficient? Real
values might improve ecological validity but add complexity. Current decision:
start with illustrative values; consider real values as a follow-up.

**9.3** The 3600-cell sweep (§3.5.1) may be too large for a single laptop run.
If wall-clock exceeds the 4-hour budget, which axes should be prioritized for
the default sweep vs. deferred to an extended run?

**9.4** For the site and notebook extensions (§3.7.3, §3.7.4): if the notebook
grows too long, when should we split into a companion notebook vs. extending
the existing one? Tentative threshold: if adding v3 content exceeds ~30 cells,
split.

## 10. Future work / TODO: Knowledge-graph representation (Idea 1)

Instead of flat feature descriptions, represent the corpus as a knowledge graph
with relational triples (e.g., "P10 --has_mass--> small", "P10 --member_of-->
planets"). Phase 2 would add/modify edges.

### Axis 1: Likelihood of producing interesting findings

**Moderate-to-high, but for a *different* reason than you might expect.**

The current setup encodes category membership in two ways: feature descriptions
("P10 has mass small...") and explicit labels ("P10 is a planet"). Tension
requires both signals to be simultaneously active and conflicting. The problem
is that a flat feature string doesn't give the model much *structure* to be
conflicted *about* — it can just overwrite the label token's distribution
without disturbing the feature representation.

A knowledge graph could help because it makes category membership *structurally
entangled* with other facts. If "P10 --member_of--> planets" is connected to
"planets --have--> large_mass" and "P10 --has_mass--> small", the conflict is
embedded in the graph topology itself, not just in a label token. Phase 2 could
add edges like "E1 --similar_to--> P10" and "E1 --member_of--> dwarfs",
creating a path that structurally links P10 to dwarfs *through* shared features
rather than just co-occurring in the training data.

The risk: you're changing so many variables simultaneously (data format,
inductive bias, what "overlap" even means) that if you *do* find tension, it's
hard to attribute it to the graph structure specifically versus the dozen other
things that changed.

### Axis 2: Methodological rigor

**This is where it gets hard.**

Several serious challenges:

1. **Linearization problem.** A decoder-only transformer eats sequences, not
   graphs. You'd need to serialize the graph into text (e.g., triples,
   adjacency descriptions, or natural-language paraphrases of edges). The
   choice of linearization format is itself a major experimental variable —
   different serializations produce very different learning dynamics, and
   there's a small literature on this (e.g., graph-to-text for GNNs vs. LLMs)
   but no consensus for tiny models.

2. **Comparability breaks.** The existing probes (logprob, embedding geometry,
   dual-label) are calibrated against the current corpus format. A graph
   representation would need a new probing framework, which means you can't
   compare results to the existing 160-cell sweep. You'd be starting a new
   experiment, not extending the current one.

3. **Confound: is it tension or is it graph structure?** If you find tension
   with a KG representation and not with flat features, you can't tell whether
   the graph *created* tension or just made the model worse at resolving
   conflicts (i.e., it might be forgetting, not tension, but in a more complex
   form).

4. **Scale concern.** Graph reasoning is hard even for large models. A
   50K-parameter transformer with a 64-token context window would struggle to
   learn meaningful graph structure. You'd likely need to scale up
   substantially, which changes the experiment's identity (it's no longer
   "runnable on a laptop in 25 minutes").

**Bottom line:** Intellectually interesting and could produce genuine tension
through structural entanglement, but it's essentially a *new project* rather
than an extension of this one. The methodological overhead is high and the
attribution problem (why did tension arise?) becomes much murkier.

## 11. Phasing recommendation

### Phase 1: Foundation (walking skeleton)

Build the continuous-feature corpus generator (`generate_corpus_v3.py`) with
digit format only, and the character-level digit tokenizer (`TokenizerV3`,
§3.1a). The tokenizer design is a Phase 1 deliverable because it is a
first-order architectural decision that affects every downstream component.
Verify the corpus + tokenizer produce a valid token stream that the existing
`train.py` / `model.py` can consume. Run one cell end-to-end (generate →
tokenize → train → probe) and confirm the pipeline works.

Deliverables: `generate_corpus_v3.py`, `TokenizerV3` (in a new
`tokenizer_v3.py` or as an extension in `tokenizer.py`), smoke test passing,
one trained checkpoint with probe results.

### Phase 2: Spelled-out format + context-window isolation

Add spelled-out representation and the two isolation strategies (pad,
match-params). Implement `block_size` adaptation in training. Verify both
formats train successfully.

Deliverables: spelled-out support in `generate_corpus_v3.py`, isolation logic,
verification that both formats produce trained models.

### Phase 3: Probes and diagnostics

Implement `tension_probe_v3.py` (adapted for continuous features),
`comprehension_probe.py`, and `baseline_classifier_v3.py`. Verify probes
produce valid output on checkpoints from Phase 2.

Deliverables: all three probe/diagnostic scripts, verified on at least one
checkpoint per representation format.

### Phase 4: Sweep infrastructure

Implement `sweep_v3.py` and `aggregate_tension_v3.py`. Run the ordinal control
condition and verify it reproduces v2 results. Run a reduced sweep (e.g., 1
seed, small model only) to validate the full pipeline.

Deliverables: sweep and aggregation scripts, ordinal-control validation,
reduced-sweep results.

### Phase 5: Full sweep + model-size feasibility

Run the full sweep with 10 seeds. Time the ~1M model tier and decide whether to
include it. Generate `tension_summary_v3.json`.

Deliverables: full sweep results, model-size feasibility determination, summary
JSON.

### Phase 6: Visualization and reporting

Implement `plot_tension_v3.py` and `build_site_v3.py`. Extend the interactive
site and Colab notebook with v3 content. Write results report.

Deliverables: figures, extended site, extended notebook, dated results report.

## 12. Directory structure

```
pluto-toy/
├── # Existing v2 files (unchanged)
├── generate_corpus.py
├── sweep.py
├── tension_probe.py
├── ...
│
├── # New v3 files
├── generate_corpus_v3.py        # Continuous-feature corpus generation
├── tokenizer_v3.py              # Character-level digit tokenizer
├── sweep_v3.py                  # v3 sweep orchestration
├── tension_probe_v3.py          # Adapted tension probe
├── comprehension_probe.py       # Numerical-comprehension diagnostic
├── aggregate_tension_v3.py      # v3 aggregation
├── plot_tension_v3.py           # v3 figures
├── baseline_classifier_v3.py    # Baselines on continuous features
├── build_site_v3.py             # Site builder with v3 data
│
├── # Shared (backward-compatible modifications only)
├── model.py
├── tokenizer.py
├── train.py
│
├── # Data and results
├── data_v3/                     # Generated corpora (not checked in)
├── sweep_v3/                    # Sweep output (not checked in)
├── results/
│   ├── tension_summary_v3_YYYY-MM-DD.json
│   └── continuous-features-YYYY-MM-DD.md
│
├── # Site and notebooks
├── docs/
│   ├── template.html            # Extended with v3 section
│   └── index.html               # Rebuilt
└── tutorial-tension.ipynb        # Extended with v3 content
```
