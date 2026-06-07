# Phase 2 Implementation Plan: Spelled-out Format + Context-Window Isolation

**Spec reference:** `_SPEC.md` §3.1a, §3.2, §3.3.2, §4.2, §5.1–5.3, §11 Phase 2
**Goal:** Add spelled-out number representation to the corpus generator and
probe, implement context-window isolation strategies (pad to match + match-params
calculator), and verify the full pipeline works end-to-end with all three
representation formats.

**Scope extension beyond spec:** The spec places probe work in Phase 3, but the
user requested that Phase 2 also implement the spelled-out probe prompt
(removing the `NotImplementedError` in `tension_probe_v3.py`). This parallels
Phase 1's decision to pull the probe forward for walking-skeleton verification.

**Dependency addition:** `num2words` is added as a hard dependency per spec
§5.2 and §6.2. It is required for spelled-out corpus generation.

**Strategy B (match-params) scope:** Phase 2 implements the parameter-count
calculator and tests it in isolation. The full n_embd/n_layer search is not yet
wired into training or sweep — Strategy B requires cross-condition comparison to
be meaningful, so the wiring is deferred to Phase 4 (sweep infrastructure).
Phase 2 delivers a tested `compute_matched_config()` function in `isolation.py`.

**Spec note:** §4.2 shows hyphens in the spelled-out example ("thirty-eight"),
but §3.1a.1 explicitly says "Hyphenated compounds are split into separate tokens
('thirty', 'eight')." We follow §3.1a.1 — hyphens are always split. The §4.2
example is inconsistent with §3.1a.1 and should be updated separately.

**Param formula note:** The spec's formula in §3.3.2 undercounts per-layer bias
parameters (says 4E per layer, actual model has 9E). The `param_count()` function
in this plan uses the corrected formula derived from `model.py`. The
`test_param_count_formula_matches_model` test validates against the real model.

---

## Step 0: Add num2words dependency

**What:** Add `num2words` as a runtime dependency in `pyproject.toml`.

**Files modified:**
- `pyproject.toml` — add `num2words` to `[project] dependencies`

**Verification:** `uv sync` installs num2words without error.
`python -c "import num2words; print(num2words.num2words(38.08))"` prints
"thirty-eight point zero eight".

**Best practice note:** Adding the dependency first so all subsequent steps can
import it. The spec (§5.2) declares num2words a hard dependency — no lazy import
guard.

---

## Step 1: Spelled-out number conversion

**What:** Add a `_number_to_words()` helper and integrate spelled-out format
into `describe_continuous()` in `generate_corpus_v3.py`. This is the core new
logic: converting float feature values to English words with proper decimal
handling, hyphen splitting, and full unit names.

**Files modified:**
- `generate_corpus_v3.py` — add `_number_to_words()` helper, `_UNIT_FULL` and
  `_DECIMAL_PRECISION` constants, update `describe_continuous()` to handle
  `representation="spelled"`
- `tokenizer_v3.py` — add spelled-out unit names ("kilograms", "kilometers",
  "astronomical", "units") to the union vocabulary so vocab is consistent
  across all three representation formats per §3.1a.3
- `tests/test_generate_corpus_v3.py` — add tests for spelled-out format

**Test first:**

```
test_describe_spelled_format
  Given name="P10", feats=(38.08, 11500.0, 39.2)
  When describe_continuous("P10", feats, representation="spelled") is called
  Then returns "P10 has mass thirty eight point zero eight kilograms
       diameter eleven thousand five hundred kilometers orbit thirty nine
       point two astronomical units ."
  (hyphens split: "thirty-eight" -> "thirty eight")

test_describe_spelled_integer_no_point
  Given name="P1", feats=(350.0, 12000.0, 5.0)
  When describe_continuous("P1", feats, representation="spelled") is called
  Then mass part contains "three hundred fifty" (not "three hundred fifty
       point zero")
  And result does not contain "point"

test_describe_spelled_small_decimal
  Given name="C1", feats=(0.001, 8.0, 42.5)
  When describe_continuous("C1", feats, representation="spelled") is called
  Then mass part contains "zero point zero zero"
  (handles small decimals correctly after rounding to 2 decimal places)

test_describe_spelled_units_are_full_words
  Given any spelled-out description
  Then result contains "kilograms", "kilometers", "astronomical units"
  And result does NOT contain "kg", "km", or "AU" as standalone tokens

test_describe_spelled_no_hyphens
  Given name="P10", feats=(38.08, 11500.0, 39.2)
  When describe_continuous("P10", feats, representation="spelled") is called
  Then result contains no hyphens
  (hyphens are always split into separate tokens per section 3.1a.1)

test_describe_spelled_no_and
  Given name="P1", feats=(350.0, 12000.0, 5.0)
  When describe_continuous("P1", feats, representation="spelled") is called
  Then result does NOT contain " and " between number parts
  (num2words may insert "and" in some locales; we strip it)

test_number_to_words_output_validation
  Given value=38.08, precision=2
  When _number_to_words is called
  Then result contains only lowercase ASCII letters and spaces
  (per spec section 5.1: num2words output must not contain unexpected characters)

test_number_to_words_rejects_garbled_output
  Given a value that would produce unexpected characters in num2words output
  When _number_to_words is called
  Then raises ValueError with the offending value and the raw num2words output
  (defensive check per spec section 5.1)

test_vocab_includes_spelled_unit_words
  Given a fitted TokenizerV3
  Then vocab contains "kilograms", "kilometers", "astronomical", "units"
  (union vocabulary per section 3.1a.3 — consistent across all representations)
```

**Test matrix (edge cases):**
- Given feats with very large mass (500.0), result uses "five hundred" (no
  "point zero")
- Given feats with zero-like orbit (0.01), result is well-formed
- Given representation="invalid", raises ValueError (existing behavior preserved)
- Given identical seeds, spelled-out corpus is deterministic

**Statistical validation test:**
```
test_describe_spelled_deterministic
  Given seed=42, representation="spelled"
  When build_phase1_continuous is called twice with same RNG state
  Then both corpus texts are byte-identical
  (validates that num2words conversion introduces no non-determinism)
```

**Implementation:**

Add to `generate_corpus_v3.py`:

```python
from num2words import num2words as _num2words

_UNIT_FULL = {"mass": "kilograms", "diameter": "kilometers",
              "orbit": "astronomical units"}
_DECIMAL_PRECISION = {"mass": 2, "diameter": 0, "orbit": 1}

def _number_to_words(value: float, precision: int) -> str:
    """Convert a number to English words, splitting hyphens into spaces.

    Validates output contains only lowercase ASCII + spaces per spec section 5.1.
    """
    rounded = round(value, precision)
    if precision == 0 or rounded == int(rounded):
        text = _num2words(int(rounded))
    else:
        text = _num2words(rounded)
    text = text.replace("-", " ").replace(",", "")
    text = " ".join(w for w in text.split() if w != "and")
    if not all(c.isascii() and (c.isalpha() or c == " ") for c in text):
        raise ValueError(
            f"num2words produced unexpected output for {value}: {text!r}"
        )
    return text
```

Also update `tokenizer_v3.py` to add spelled-out unit words to the union
vocabulary:

```python
_UNIT_WORDS = ["kg", "km", "AU",
               "kilograms", "kilometers", "astronomical", "units"]
```

This ensures `TokenizerV3.fit()` includes these tokens regardless of which
representation format generated the training corpus, maintaining vocabulary
consistency per §3.1a.3.

And a new branch in `describe_continuous()`:

```python
if representation == "spelled":
    parts = [f"{name} has"]
    for axis, val in [("mass", mass), ("diameter", diam), ("orbit", orbit)]:
        words = _number_to_words(val, _DECIMAL_PRECISION[axis])
        parts.append(f"{axis} {words} {_UNIT_FULL[axis]}")
    return " ".join(parts) + " ."
```

Also update the ValueError message to include "spelled" in the valid options.

**Line budget:** ~30 new lines. `generate_corpus_v3.py` goes from 230 to ~260
lines. The file was already over the 200-line target in Phase 1. Accepted as
tolerable for a core data module — factoring constants would add churn for
marginal benefit at this stage.

**Verification:** `uv run pytest tests/test_generate_corpus_v3.py -v -k spelled`
— all new tests pass. Manual inspection:
`python3 generate_corpus_v3.py --out-dir /tmp/test --mode dwarf --representation spelled --seed 42`
and inspect `phase1.txt`.

---

## Step 2: Spelled-out corpus assembly + CLI validation

**What:** Verify that `build_phase1_continuous` and `build_phase2_continuous`
work with `representation="spelled"`, and that the CLI accepts
`--representation spelled`. Also measure sentence token counts to inform
block_size selection in Step 3.

**Files modified:**
- `generate_corpus_v3.py` — update CLI `choices` to include `"spelled"`
- `tests/test_generate_corpus_v3.py` — add assembly and CLI tests

**Test first:**

```
test_build_phase1_spelled_produces_all_entities
  Given seed=42, representation="spelled"
  When build_phase1_continuous is called
  Then returned entities dict has same keys as digit format
  And text contains "kilograms" (unit words present)

test_build_phase1_spelled_sentence_length
  Given seed=42, representation="spelled"
  When build_phase1_continuous is called and all description sentences
       are preprocessed by TokenizerV3.preprocess
  Then max token count is recorded and < 60 tokens
  (FEASIBILITY CHECK: determines required block_size for Strategy A.
   If max exceeds 128 after headroom, Strategy A becomes expensive.)

test_cli_spelled_representation_produces_output
  Given --out-dir {tmp} --mode dwarf --representation spelled --seed 42
  When generate_corpus_v3.py is run as subprocess
  Then phase1.txt, phase2.txt, entities.json all exist
  And phase1.txt contains "kilograms"
  And entities.json contains "representation": "spelled"

test_cli_all_three_representations_accepted
  Given --representation digit, ordinal, spelled (three separate runs)
  When generate_corpus_v3.py is run
  Then all three exit 0 (no ArgumentError)
```

**Test matrix (edge cases):**
- Given --representation spelled --seed 42 run twice, output is byte-identical
- Given spelled format with n_eris=0, phase2.txt is empty
- Given spelled format, entities.json features are still numeric floats (text
  format does not affect stored data)

**Implementation:**

Update CLI in `generate_corpus_v3.py`:

```python
ap.add_argument(
    "--representation",
    choices=["digit", "ordinal", "spelled"],
    default="digit",
)
```

No other changes needed — `build_phase1_continuous` and
`build_phase2_continuous` already pass `representation` through to
`describe_continuous()`, which handles "spelled" from Step 1.

**Verification:** Run CLI with all three representations and inspect output.
Record max spelled-out sentence token count from the feasibility test.

---

## Step 3: Block-size adaptation + isolation Strategy A

**What:** Add a `required_block_size()` helper that computes the minimum
block_size for a given corpus. Implement Strategy A (pad to match): all
conditions use the same (max) block_size so the model architecture is identical
across representation formats.

Note: the isolation strategy is a training/sweep concern, not a corpus-generation
concern (the corpus text is identical regardless of strategy). The `--isolation`
flag and metadata recording belong in the sweep manifest (Phase 4), not in
`generate_corpus_v3.py` or `entities.json`.

**Files created:**
- `isolation.py` — `required_block_size()` function
- `tests/test_isolation.py` — tests for block-size computation

**Test first:**

```
test_required_block_size_ordinal
  Given corpus text from representation="ordinal", seed=42
  When required_block_size(corpus_text) is called
  Then returns 64 (ordinal sentences are short)

test_required_block_size_digit
  Given corpus text from representation="digit", seed=42
  When required_block_size(corpus_text) is called
  Then returns 64 (digit sentences fit within 64)

test_required_block_size_spelled
  Given corpus text from representation="spelled", seed=42
  When required_block_size(corpus_text) is called
  Then returns a value >= 128 (spelled sentences are longer)

test_required_block_size_is_power_of_2
  Given any corpus text
  When required_block_size is called
  Then result is a power of 2 (64, 128, 256, ...)

test_strategy_a_uses_max_block_size
  Given three corpora (digit, ordinal, spelled)
  When required_block_size is called for each
  Then Strategy A block_size = max(results)
  And all three conditions would use this same value

test_spelled_corpus_trains_with_adapted_block_size
  Given spelled corpus, block_size from required_block_size(corpus)
  When train.py is called with --block-size {computed}
       for 200 steps with small model (n_embd=16, n_layer=2)
  Then training completes without error and loss decreases
  (CRITICAL: verifies spelled-out format actually trains)

test_digit_corpus_trains_with_padded_block_size
  Given digit corpus, block_size = required_block_size(spelled_corpus)
       (Strategy A: pad to match)
  When train.py is called with the padded block_size
  Then training completes and loss is comparable to block_size=64
  (verifies digit format doesn't degrade with larger block_size)
```

**Test matrix (edge cases):**
- Given an empty corpus, required_block_size raises ValueError
- Given block_size=256 with ordinal format, training still converges (wastes
  positional embedding capacity but doesn't break)

**Implementation:**

New file `isolation.py` (~40 lines):

```python
from tokenizer_v3 import TokenizerV3

def required_block_size(corpus_text: str) -> int:
    """Minimum power-of-2 block_size for a corpus, with headroom.

    Scans all lines, measures max token count after TokenizerV3 preprocessing,
    adds 50% headroom, and rounds up to the next power of 2. Floor is 64.

    The 50% headroom ensures training windows (which sample contiguous token
    spans) contain multiple sentences, not just one maximal sentence. The
    exact factor is calibrated to keep spelled-out format in the 128-256
    range per spec section 3.2.1.
    """
    max_tokens = max(
        len(TokenizerV3.preprocess(line).split())
        for line in corpus_text.strip().split("\n")
        if line.strip()
    )
    target = int(max_tokens * 1.5)
    block_size = 64
    while block_size < target:
        block_size *= 2
    return block_size
```

**Statistical validation test:**
```
test_block_size_monotonic_across_representations
  Given corpora for digit, ordinal, and spelled (same seed)
  When required_block_size is called for each
  Then ordinal <= digit <= spelled
  (longer text representations require larger or equal block_size)
```

**Verification:** `uv run pytest tests/test_isolation.py -v` — all tests pass.
Run:
```bash
python3 -c "
from generate_corpus_v3 import build_phase1_continuous
from isolation import required_block_size
import random
text, _ = build_phase1_continuous(random.Random(42), 'spelled')
print('spelled block_size:', required_block_size(text))
"
```
Confirm a reasonable value (expect 128 or 256).

---

## Step 4: Match-params parameter-count calculator (Strategy B stub)

**What:** Implement `param_count()` and `compute_matched_config()` in
`isolation.py`. These are the core of Strategy B: given a target parameter count
and a different block_size, find n_embd/n_layer values that hold total params
within +/-10%. The functions are tested in isolation but NOT wired into training
or sweep yet.

**Files modified:**
- `isolation.py` — add `param_count()`, `compute_matched_config()`
- `tests/test_isolation.py` — add match-params tests

**Test first:**

```
test_param_count_formula_matches_model
  Given GPTConfig(n_embd=32, n_layer=4, n_head=4, block_size=64, vocab_size=100)
  When param_count(vocab_size, n_embd, n_layer, block_size) is called
  Then result matches GPT(config).num_params() exactly
  (validates our formula against the actual model)

test_param_count_increases_with_block_size
  Given same config but block_size=128 vs block_size=64
  When param_count is called for both
  Then result with block_size=128 > result with block_size=64
  (positional embeddings scale with block_size)

test_compute_matched_config_within_tolerance
  Given target_params=50000, block_size=128, n_layer=4, n_head=4, vocab_size=100
  When compute_matched_config is called
  Then returned n_embd produces param count within +/-10% of target

test_compute_matched_config_n_embd_divisible_by_n_head
  Given n_head=4
  When compute_matched_config is called
  Then returned n_embd % n_head == 0

test_compute_matched_config_small_tier
  Given target ~50K params (small tier), block_size from spelled-out
  When compute_matched_config is called
  Then returns a valid n_embd with actual params within +/-10%

test_compute_matched_config_medium_tier
  Given target ~200K params (medium tier), block_size from spelled-out
  When compute_matched_config is called
  Then returns a valid n_embd within +/-10%

test_compute_matched_config_impossible_target_raises
  Given target_params=10 (impossibly small for any valid n_embd)
  When compute_matched_config is called
  Then raises ValueError with message about infeasible target
```

**Test matrix (edge cases):**
- Given block_size=64 (ordinal default), matched config for small tier returns
  n_embd=32 (the existing default — no adjustment needed)
- Given very large block_size (512), n_embd is reduced to compensate
- Given n_head=2 vs n_head=4, both produce valid configs

**Implementation:**

Add to `isolation.py`:

```python
def param_count(vocab_size: int, n_embd: int, n_layer: int,
                block_size: int) -> int:
    """Parameter count for our GPT model (weight tying, pre-LN blocks).

    Derived from model.py, NOT the spec's formula (which undercounts biases).
    Per block: qkv(3E²) + proj(E²) + fc(4E²+4E) + mlp_proj(4E²+E)
               + ln1(2E) + ln2(2E) = 12E² + 9E.
    Final ln_f: 2E. head shares weights with tok_emb.
    """
    E = n_embd
    return (
        vocab_size * E              # tok_emb (shared with head)
        + block_size * E            # pos_emb
        + n_layer * (
            12 * E * E              # attention + MLP weights
            + 9 * E                 # LN weights/biases + MLP biases
        )
        + 2 * E                     # ln_f weight + bias
    )
    # TODO (Phase 4): wire into sweep infrastructure for Strategy B isolation

def compute_matched_config(
    target_params: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    vocab_size: int,
) -> int:
    """Find n_embd that holds param count within +/-10% of target.

    Searches n_embd values divisible by n_head (ascending), returns the
    value whose param_count is closest to target. Raises ValueError if
    no candidate achieves +/-10%.
    """
    best_n_embd = None
    best_distance = float("inf")
    for n_embd in range(n_head, 512, n_head):
        count = param_count(vocab_size, n_embd, n_layer, block_size)
        distance = abs(count - target_params)
        if distance < best_distance:
            best_distance = distance
            best_n_embd = n_embd
        if count > target_params * 1.5:
            break
    if best_n_embd is None or best_distance > target_params * 0.1:
        raise ValueError(
            f"No n_embd achieves +/-10% of target {target_params} "
            f"with block_size={block_size}, n_layer={n_layer}"
        )
    return best_n_embd
```

**Line budget:** `isolation.py` total ~80 lines (required_block_size from Step 3
+ param_count + compute_matched_config). Under the 200-line limit.

**Verification:** `uv run pytest tests/test_isolation.py -v -k match` — all
tests pass. Cross-check:

```python
from model import GPT, GPTConfig
cfg = GPTConfig(n_embd=32, n_layer=4, n_head=4, block_size=64, vocab_size=100)
model = GPT(cfg)
print(f"Formula: {param_count(100, 32, 4, 64)}, Model: {model.num_params()}")
```

Both values should be identical.

---

## Step 5: Spelled-out probe prompt

**What:** Remove the `NotImplementedError` in `tension_probe_v3.py` and
implement the spelled-out probe prompt format. The probe uses the same
calibration and classification logic — only the prompt text changes.

**Files modified:**
- `tension_probe_v3.py` — update `feature_prompt_v3()` to handle
  `representation="spelled"` (remove the guard clause)
- `tests/test_tension_probe_v3.py` — add spelled-out probe tests

**Test first:**

```
test_feature_prompt_spelled_format
  Given name="P10", feats=(38.08, 11500.0, 39.2), representation="spelled"
  When feature_prompt_v3 is called
  Then returns "P10 has mass thirty eight point zero eight kilograms
       diameter eleven thousand five hundred kilometers orbit thirty nine
       point two astronomical units . P10 is a"
  (uses describe_continuous with spelled format, appends " {name} is a")

test_feature_prompt_spelled_no_hyphens
  Given any entity and feats, representation="spelled"
  When feature_prompt_v3 is called
  Then result contains no hyphens

test_probe_spelled_prompt_fits_block_size
  Given spelled-out probe prompt for P10 with typical features
  When preprocessed by TokenizerV3.preprocess and tokenized
  Then token count < required_block_size for spelled corpus
  (probe prompt must fit within the model's context window)

test_spelled_probe_produces_valid_output
  Given a model trained on spelled-out corpus (fixture with adapted block_size)
  When probe_v3 is called with representation="spelled"
  Then output dict has expected schema (thresholds, targets, classifications)
  And P10 has a valid classification in {TENSION, canon, swap, nothing}

test_spelled_probe_classification_logic_unchanged
  Given fixed P_planet=0.4, P_dwarf=0.3, thr_planet=0.31, thr_dwarf=0.25
  When classify() is called
  Then classification is "TENSION" regardless of representation
  (classification logic is representation-agnostic — shared with v2)
```

**Test matrix (edge cases):**
- Given mode="unlabeled" (no dwarf anchors), probe handles empty dwarf anchor
  set gracefully (existing behavior, verify not broken)
- Given a model with random weights (no training), probe returns valid JSON
  with classifications (doesn't crash on low-confidence outputs)

**Implementation:**

In `tension_probe_v3.py`, replace lines 37-38:

```python
if representation == "spelled":
    raise NotImplementedError("Spelled-out probe prompts require Phase 2")
```

with nothing — remove the guard. The function already calls
`describe_continuous()` on line 39, which now handles all three formats after
Step 1. The rest of the probe logic (calibration, thresholds, classification)
is representation-agnostic and needs no changes.

Add a fixture to `tests/conftest.py` for a spelled-out trained model (or
parameterize the existing `trained_small_model` fixture to accept
representation and block_size).

**Verification:** `uv run pytest tests/test_tension_probe_v3.py -v -k spelled`
— all new tests pass.

---

## Step 6: End-to-end smoke tests for spelled-out pipeline

**What:** Full pipeline tests for spelled-out format: generate -> train -> probe.
Both as function calls and CLI subprocess calls. Verify all three representation
formats produce valid results.

**Files modified:**
- `tests/test_smoke_e2e.py` — add spelled-out pipeline tests
- `tests/conftest.py` — add or extend fixtures for spelled-out training

**Test first:**

```
test_full_pipeline_spelled_format
  Given seed=42, representation="spelled", mode="dwarf", schedule="curriculum"
  When generate -> fit tokenizer -> train (adapted block_size) -> probe
  Then probe output JSON exists with valid schema
  And P10 has a classification

test_full_pipeline_spelled_canon_only
  Given representation="spelled", schedule="canon-only"
  When full pipeline runs
  Then training completes and probe produces output

test_full_pipeline_spelled_cli_subprocess
  Given the full v3 CLI pipeline with --representation spelled:
    python3 generate_corpus_v3.py --out-dir {tmp}/data --mode dwarf
        --representation spelled --seed 42
    python3 -c "from tokenizer_v3 import TokenizerV3; from pathlib import Path;
        tok = TokenizerV3.fit(Path('{tmp}/data/phase1.txt').read_text()
        + '\n' + Path('{tmp}/data/phase2.txt').read_text());
        tok.save('{tmp}/data/tokenizer.json')"
    python3 train.py --data-dir {tmp}/data --out-dir {tmp}/run
        --schedule curriculum --tokenizer {tmp}/data/tokenizer.json
        --block-size {computed} --phase1-steps 100 --phase2-steps 50
        --n-embd 16 --n-layer 2
    python3 tension_probe_v3.py --ckpt {tmp}/run/ckpt.pt
        --entities {tmp}/data/entities.json --out {tmp}/run/tension.json
        --representation spelled
  Then all commands exit 0
  And tension.json has valid schema

test_all_three_representations_produce_valid_probes
  Given representations=["digit", "ordinal", "spelled"], same seed
  When full pipeline runs for each
  Then all three produce valid probe JSON with P10 classification
  (cross-format validation: same underlying data, different text rendering)

test_spelled_format_model_learns_categories
  Given spelled corpus, mode="planet", schedule="canon-only",
        n_embd=16, n_layer=2, 1000 steps, adapted block_size
  When trained and probed
  Then P(planet | P1) > 0.15
  (same threshold as Phase 1's digit-format test)
```

**Test matrix (edge cases):**
- Given spelled format, reproducibility test: same seed produces same probe
  output within tolerance (atol=1e-6)
- Given spelled format with larger block_size, test completes within timeout
  (training is slower with larger context window, but should stay under 2 min
  for small model)

**Implementation:**

Follow the pattern in existing `test_smoke_e2e.py`. The main difference is
passing `--block-size` to `train.py` with the value from
`required_block_size()`, and passing `--representation spelled` to both the
corpus generator and probe.

For the spelled-out training fixture, either:
(a) Add a `trained_small_model_spelled` session-scoped fixture to
    `conftest.py`, or
(b) Parameterize the existing `trained_small_model` fixture with
    `representation` and `block_size` parameters.

Option (b) is cleaner if Phase 3 tests also need different representations.
Option (a) is simpler for now. Use (a) — a second fixture is ~15 lines and
avoids re-running Phase 1 tests with spelled-out format.

**Verification:** `uv run pytest tests/test_smoke_e2e.py -v` — all tests pass,
including new spelled-out tests. Total time under 5 minutes.

---

## Step 7: v2 backward-compatibility verification

**What:** Confirm that all Phase 2 changes have not broken the v2 pipeline or
the existing v3 digit/ordinal pipeline from Phase 1.

**Files modified:**
- None — existing tests should pass without changes.

**Test first:**

No new tests. Run the full existing suite:

```
test_v2_corpus_generation_unchanged — existing, must pass
test_v2_train_pipeline_unchanged — existing, must pass
test_v2_tension_probe_unchanged — existing, must pass
test_v2_quick_start_command — existing, must pass
```

Plus the entire Phase 1 test suite (90 tests).

**Verification:**

```bash
uv run pytest tests/ -v
```

All 110+ tests pass. No regressions. v2 Quick Start command from CLAUDE.md
works unchanged:

```bash
python3 generate_corpus.py --out-dir data --mode unlabeled
python3 train.py --out-dir runs/canon_only --schedule canon-only --data-dir data
```

---

## Exit checklist

All criteria must pass before Phase 2 is considered complete:

- [ ] `uv run pytest` — all tests pass (Phase 1 + Phase 2)
- [ ] `uv run pytest --co` — test count >= 110
- [ ] `num2words` listed as runtime dependency in `pyproject.toml`
- [ ] `generate_corpus_v3.py --representation spelled --mode dwarf --seed 42`
      produces valid corpus with English number words and full unit names
- [ ] Spelled-out corpus contains no hyphens (split per §3.1a.1)
- [ ] Spelled-out corpus contains no "and" in number words
- [ ] `required_block_size()` on spelled corpus returns power of 2 >= 128
- [ ] `required_block_size()` on digit and ordinal corpora returns 64
      (verify empirically; if > 64, investigate whether sentences grew)
- [ ] Spelled corpus trains with adapted block_size, loss decreases
- [ ] Digit corpus trains with padded block_size (Strategy A), loss comparable
      to block_size=64
- [ ] `param_count()` formula matches `GPT(config).num_params()` for all
      test configs
- [ ] `compute_matched_config()` produces configs within +/-10% of target for
      small and medium tiers
- [ ] `tension_probe_v3.py` accepts `representation="spelled"` (no
      NotImplementedError)
- [ ] Spelled probe produces valid JSON with P10 classification
- [ ] Full CLI pipeline works for all three representations
- [ ] All three representations produce valid probe output with same seed
- [ ] v2 backward-compatibility tests pass (no regressions)
- [ ] v2 Quick Start from CLAUDE.md works unchanged
- [ ] No new files exceed 200 lines
- [ ] All new functions have type hints and brief docstrings
- [ ] `isolation.py` exists with `required_block_size()`, `param_count()`, and
      `compute_matched_config()`
- [ ] TODO in `isolation.py`: "Strategy B wiring deferred to Phase 4"
