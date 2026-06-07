# Phase 1 Implementation Plan: Foundation (Walking Skeleton)

**Spec reference:** `_SPEC.md` §3.1, §3.1a, §11 Phase 1
**Goal:** Build the continuous-feature corpus generator (digit format + ordinal
control), the character-level digit tokenizer, and verify the full pipeline
end-to-end (generate -> tokenize -> train -> probe).

**Note on scope:** The spec places probes in Phase 3, but `tension_probe_v3.py`
is pulled forward into Phase 1 because the digit-format probe prompt differs
from v2's and we need it to verify the walking skeleton end-to-end. It is a
functional stub, not the full Phase 3 implementation.

**Deferred to Phase 4:** Configurable distribution parameters (§3.1.1 says
"must be configurable via CLI flags or a config dict"). Phase 1 uses hardcoded
defaults; a `--prototypes-json` CLI flag will be added in Phase 4 when sweep
infrastructure needs it. TODO is marked in `generate_corpus_v3.py`.

---

## Step 0: Project infrastructure

**What:** Set up test framework, dependency management, and project tooling.

**Files created:**
- `tests/conftest.py` — shared fixtures
- `pyproject.toml` — dependency management via `uv`

**Details:**
- Install pytest, pytest-cov as dev dependencies.
- Add torch as a runtime dependency. (`num2words` is NOT added in Phase 1 —
  the union vocabulary uses a hardcoded word list instead. `num2words` is added
  in Phase 2 when spelled-out generation needs it.)
- Create `tests/` directory with `conftest.py` providing:
  - `tmp_corpus_dir` fixture (tmp_path-based)
  - `seeded_rng` fixture returning `random.Random(42)`
  - `small_model_cfg` fixture returning a GPTConfig with n_embd=16, n_layer=2,
    n_head=2, block_size=64 (fast tests)
  - `trained_small_model` fixture that generates a digit-format corpus, fits
    a `TokenizerV3`, trains for 100 steps, and returns (model, tokenizer,
    entities_dict). Used by Steps 4, 5, and 6 to avoid duplicating setup.

**Verification:** `uv run pytest --co` collects zero tests (no test files yet)
but exits cleanly.

**Best practice note:** We're establishing the test infrastructure first so
every subsequent step can follow TDD (test first, watch it fail, implement).

---

## Step 1: Continuous-feature corpus generator — core data model

**What:** Build the core entity generation logic for continuous features:
distribution sampling, feature formatting, and the entity data model. No file
I/O or CLI yet.

**Files created:**
- `generate_corpus_v3.py` — core functions only (no `if __name__`)
- `tests/test_generate_corpus_v3.py`

**Line budget:** ~120 lines for Step 1 (constants + 6 functions). Step 2 adds
~60 lines (build_phase1, build_phase2, CLI). Total ~180 lines, under the
200-line repo limit. If it creeps over, factor PROTOTYPES and ORDINAL_BINS
into a compact block at module top.

**Test first (write these before implementation):**

```
test_sample_canonical_returns_correct_count_and_shape
  Given category="planet", n=9, seed=42
  When sample_canonical_continuous is called
  Then returns list of 9 tuples, each with 3 positive floats

test_sample_canonical_respects_distribution
  Given category="planet", n=1000, seed=42
  When sample_canonical_continuous is called
  Then mean mass is within 3 SE of 350, mean diameter within 3 SE of 12000
  (Statistical validation: seed is fixed so this is deterministic, but 3 SE
   provides margin if seeds change. SE(mass) = 50/sqrt(1000) ~ 1.58,
   so 3 SE ~ 4.74. The bound is a sanity check, not a flaky statistical test.)

test_sample_edge_places_p10_at_configured_mass
  Given p10_edge=40.0, seed=42, n=100 samples
  When sample_edge_continuous is called 100 times
  Then mean mass is within 3 SE of 40.0

test_sample_edge_at_prototype_mean
  Given p10_edge=350.0, seed=42, n=100 samples
  When sample_edge_continuous is called 100 times
  Then mean mass is within 3 SE of 350.0
  (edge case: P10 positioned at planet prototype center, should look canonical)

test_sample_disjoint_matches_v2_direction
  Given seed=42, n=100 samples
  When sample_disjoint_continuous is called 100 times
  Then mean mass > 400 (huge), mean diameter < 50 (tiny), mean orbit < 0.5 (near)
  (validates §3.1.3: rote corner matches v2 direction — mass=huge, diameter=tiny, orbit=near)

test_describe_digit_format
  Given name="P10", feats=(38.08, 11500.0, 39.2)
  When describe_continuous("P10", feats, representation="digit") is called
  Then returns "P10 has mass 38.08 kg diameter 11500 km orbit 39.2 AU ."

test_describe_digit_no_scientific_notation
  Given name="C1", feats=(0.001, 8.0, 42.5)
  When describe_continuous("C1", feats, representation="digit") is called
  Then returns "C1 has mass 0.00 kg diameter 8 km orbit 42.5 AU ."
  (no "1e-03" or similar)

test_describe_ordinal_format
  Given name="P10", feats=(38.08, 11500.0, 39.2)
  When describe_continuous("P10", feats, representation="ordinal") is called
  Then returns "P10 has mass small diameter large orbit distant ."
  (ordinal bins the continuous value into 0-4 via ORDINAL_BINS, maps to v2 words)

test_describe_ordinal_corpus_has_no_units
  Given name="P1", feats=(350.0, 12000.0, 5.0), representation="ordinal"
  When describe_continuous is called
  Then result does not contain "kg", "km", or "AU"

test_preprocess_ordinal_sentence_unchanged
  Given text "P10 has mass small diameter large orbit distant ."
  When TokenizerV3.preprocess is called (from Step 3 — cross-validates)
  Then returns the same string unchanged (no numeric tokens to split)

test_label_format_matches_v2
  Given name="E1", category="dwarf"
  When label("E1", "dwarf") is called
  Then returns "E1 is a dwarf ."
```

**Test matrix (edge cases):**
- Given mass=-5.0 (negative from Gaussian tail), sampling clips to floor > 0
- Given n_eris_rote > n_eris, raises ValueError
- Given representation="invalid", raises ValueError
- Given identical seeds, two calls produce identical feature tuples (reproducibility)

**Implementation:**

Core functions in `generate_corpus_v3.py`:

```python
PROTOTYPES = {
    "planet":   {"mass": (350, 50), "diameter": (12000, 2000), "orbit": (5.0, 2.0)},
    "asteroid": {"mass": (0.5, 0.2), "diameter": (50, 20), "orbit": (3.0, 1.0)},
    "comet":    {"mass": (0.01, 0.005), "diameter": (10, 5), "orbit": (40, 15)},
    "moon":     {"mass": (5, 2), "diameter": (1500, 500), "orbit": (0.5, 0.2)},
}
# TODO (Phase 4): make configurable via --prototypes-json CLI flag per §3.1.1

ORDINAL_BINS = {
    "mass": [0.1, 1.0, 50, 200],
    "diameter": [20, 100, 5000, 10000],
    "orbit": [0.3, 1.0, 3.0, 10.0],
}
```

Functions:
- `sample_canonical_continuous(category, n, rng, sigma_scale=1.0)` — returns
  list of (mass, diameter, orbit) floats from N(mu, sigma) clipped to > 0
- `sample_edge_continuous(p10_edge, rng, sigma=5.0)` — P10 features: mass from
  N(p10_edge, sigma), diameter from planet-edge, orbit pinned high (~39 AU).
  CLI flag is `--p10-edge` (matches v2 naming; "mass" is implicit since only
  mass varies per §3.1.2)
- `sample_disjoint_continuous(rng, sigma_scale=1.0)` — rote-control corner:
  mass ~N(500, 50), diameter ~N(5, 2), orbit ~N(0.01, 0.005). Matches v2
  direction (huge, tiny, near).
- `describe_continuous(name, feats, representation)` — format features as text.
  `representation="digit"`: round per axis (2 decimals for mass, 0 for
  diameter, 1 for orbit), append units, use f-string formatting (not repr())
  to avoid scientific notation. `representation="ordinal"`: bin into 0-4 using
  ORDINAL_BINS, map to v2 vocabulary words; no units.
- `label(name, category)` — identical to v2: `"{name} is a {category} ."`
- `to_ordinal(value, axis)` — helper to bin a continuous value

**Verification:** `uv run pytest tests/test_generate_corpus_v3.py -v` — all
tests pass including the statistical validation tests.

---

## Step 2: Continuous-feature corpus generator — corpus assembly + CLI

**What:** Build the full corpus assembly (phase 1 + phase 2 text) and CLI
interface, matching v2's output file structure.

**Files modified:**
- `generate_corpus_v3.py` — add `build_phase1`, `build_phase2`, CLI main
- `tests/test_generate_corpus_v3.py` — add assembly + CLI tests

**Test first:**

```
test_build_phase1_produces_all_entities
  Given seed=42, representation="digit"
  When build_phase1_continuous is called
  Then returned entities dict has keys P1..P10 + 6 asteroids + 6 comets + 6 moons
  And P10's category is "planet"

test_build_phase1_text_contains_roster_lines
  Given seed=42, representation="digit"
  When build_phase1_continuous is called
  Then text contains "the planets are P1 and P2 and P3 and P4 and P5 and P6 and P7 and P8 and P9 and P10 ."
  (exact match on roster format — "and" between every member, matching v2)

test_build_phase1_digit_sentence_fits_block_size
  Given seed=42, representation="digit"
  When build_phase1_continuous is called and all description + label line pairs
       are preprocessed by TokenizerV3.preprocess
  Then max token count of any single sentence < 40 (leaves room within block_size=64)
  (PRIMARY FEASIBILITY CHECK: if digit-format sentences don't fit in a 64-token
   window alongside at least one other sentence, the approach needs rethinking)

test_build_phase2_overlap_entities_near_p10
  Given mode="dwarf", n_eris=6, n_eris_rote=0, seed=42
  And P10 features from build_phase1_continuous with same seed
  When build_phase2_continuous is called
  Then all E* mass values are within 20 of P10's actual mass from phase1
  (uses P10's actual sampled mass, not the assumed mean of 40)

test_build_phase2_rote_entities_in_disjoint_corner
  Given mode="dwarf", n_eris=6, n_eris_rote=3, seed=42
  When build_phase2_continuous is called
  Then first 3 E* have mass > 400 (disjoint corner)
  And last 3 E* have mass < 100 (overlap region)

test_build_phase2_mode_unlabeled_no_labels
  Given mode="unlabeled"
  When build_phase2_continuous is called
  Then text contains no "is a" lines for E* entities

test_build_phase2_mode_dwarf_has_labels
  Given mode="dwarf"
  When build_phase2_continuous is called
  Then text contains "E1 is a dwarf ." for each E*

test_entities_json_schema_matches_v2
  Given a complete corpus generation
  When entities.json is written
  Then it has "phase1", "phase2", "phase2_mode", "n_eris", "n_eris_rote",
       "p10_edge", "representation" keys
  And phase1 entries have "category" and "features" (3-tuple of floats)

test_cli_produces_output_files
  Given --out-dir {tmp} --mode dwarf --representation digit --seed 42
  When generate_corpus_v3.py is run as subprocess
  Then {tmp}/phase1.txt, {tmp}/phase2.txt, {tmp}/entities.json all exist
  And phase1.txt contains "kg" (digit format units)

test_ordinal_representation_matches_v2_format
  Given --representation ordinal --mode dwarf --seed 42
  When corpus is generated
  Then phase1.txt lines match v2 format ("has mass small diameter large orbit distant .")
  And no "kg", "km", "AU" appear in text
```

**Test matrix (edge cases):**
- Given --seed 42 run twice, output files are byte-identical (reproducibility)
- Given --n-eris 0, phase2.txt is empty, entities.json phase2 dict is empty
- Given --mode planet, E* labels are "planet" not "dwarf"
- Given invalid --representation value, argparse exits with error

**Implementation:**

- `build_phase1_continuous(rng, representation, repeats=8, p10_edge=40.0)`
  — generates 9 planets + P10 + 18 others. Returns (text, entities_dict).
  Text structure matches v2: descriptions, labels, roster lines, repeated
  `repeats` times with shuffled entity order per repeat. Roster format:
  `"the planets are P1 and P2 and ... and P10 ."` (every member separated
  by "and", matching v2 exactly).
- `build_phase2_continuous(rng, representation, mode, n_eris=6,
  n_eris_rote=0, repeats=6, p10_feats=None)` — generates E1..En. Corner-mix
  split via n_eris_rote. `p10_feats` tuple is passed from phase1 so overlap
  E* are placed near P10's *actual* sampled features, not a fixed mean.
  Default n_eris=6 (matches spec §3.5.1, differs from v2's default of 5).
- CLI mirrors v2's interface plus `--representation` flag. Flag names use
  kebab-case matching v2: `--p10-edge`, `--n-eris`, `--n-eris-rote`,
  `--out-dir`, `--seed`, `--mode`, `--representation`.
- Output files: `phase1.txt`, `phase2.txt`, `entities.json` with schema
  matching v2 extended with `"representation"` field.

**Verification:** Run CLI with `--representation digit` and `--representation
ordinal`, inspect output files. Ordinal output should be structurally
identical to v2 (same entity names, same sentence patterns), though feature
values differ because v3 samples continuous then bins.

---

## Step 3: Character-level digit tokenizer

**What:** Build `TokenizerV3` — a tokenizer that splits numeric strings into
individual characters while keeping non-numeric tokens as whole words.

**Files created:**
- `tokenizer_v3.py` — `TokenizerV3` class
- `tests/test_tokenizer_v3.py`

**Test first:**

```
test_numeric_string_split_into_characters
  Given text "P10 has mass 38.08 kg"
  When TokenizerV3.preprocess is called
  Then returns "P10 has mass 3 8 . 0 8 kg"

test_non_numeric_tokens_preserved
  Given text "P10 is a planet ."
  When TokenizerV3.preprocess is called
  Then returns "P10 is a planet ." (unchanged)

test_mixed_alphanumeric_tokens_not_split
  Given text "P10 E1 A3"
  When TokenizerV3.preprocess is called
  Then returns "P10 E1 A3" (entity names are not purely numeric strings)

test_preprocess_ordinal_corpus_unchanged
  Given text "P10 has mass small diameter large orbit distant ."
  When TokenizerV3.preprocess is called
  Then returns same string unchanged

test_preprocess_leading_zero_decimal
  Given text "0.001"
  When TokenizerV3.preprocess is called
  Then returns "0 . 0 0 1"

test_preprocess_integer_no_decimal
  Given text "11500"
  When TokenizerV3.preprocess is called
  Then returns "1 1 5 0 0"

test_fit_produces_expected_vocab_size
  Given a digit-format corpus
  When TokenizerV3.fit is called
  Then vocab_size is approximately 100 (digits 0-9, ".", ~30 number words,
       ~60 word tokens)

test_vocab_identical_across_representations
  Given a digit-format corpus and an ordinal-format corpus (different seeds)
  When TokenizerV3.fit is called on each independently
  Then vocab_size is identical for both
  (union vocabulary ensures consistent parameter count per §3.1a.3)

test_encode_decode_roundtrip
  Given text "P10 has mass 3 8 . 0 8 kg diameter 1 1 5 0 0 km orbit 3 9 . 2 AU ."
  When encode then decode is called
  Then output matches input

test_vocab_includes_number_words
  Given a fitted TokenizerV3
  Then vocab contains "thirty", "eight", "hundred", "thousand", "point"
  (hardcoded union vocabulary per §3.1a.3)

test_save_load_roundtrip
  Given a fitted TokenizerV3 saved to JSON
  When loaded from that JSON
  Then vocab is identical and encode/decode produce same results

test_save_includes_class_field
  Given a fitted TokenizerV3 saved to JSON
  When JSON is loaded as raw dict
  Then it contains "tokenizer_class": "v3"
  (enables auto-detection when loading — see Step 4)

test_pad_token_is_zero
  Given a fitted TokenizerV3
  Then token_to_id["<pad>"] == 0

test_encode_unknown_token_raises_informative_error
  Given a fitted TokenizerV3 and text containing "UNKNOWN_TOKEN"
  When encode is called
  Then raises KeyError with message containing "UNKNOWN_TOKEN" and a
       truncated excerpt of the input text for debugging context
```

**Implementation:**

`TokenizerV3` class in `tokenizer_v3.py`:

- Same interface as `WordTokenizer` (fit, encode, decode, save, load,
  vocab_size property) but separate class.
- `preprocess(text: str) -> str` — static method. Splits on whitespace. For
  each token, if it matches `^-?\d+\.?\d*$` (purely numeric string), replaces
  with space-separated characters. Rejoins.
- `fit(cls, text: str, include_number_words: bool = True)` — class method.
  Accepts a single string (matching `WordTokenizer.fit` signature) or a list
  of strings (joined with newlines). Preprocesses, splits on whitespace,
  builds vocab from unique tokens. If `include_number_words=True`, adds a
  hardcoded list of ~30 English number words: zero through nineteen, twenty,
  thirty, forty, fifty, sixty, seventy, eighty, ninety, hundred, thousand,
  million, point. These are the tokens Phase 2's spelled-out format will need.
- `encode(text: str) -> List[int]` — preprocesses then looks up each token.
  On KeyError, re-raises with context: `f"Unknown token '{t}' in: '{text[:80]}...'"`.
- `decode(ids: List[int]) -> str` — joins tokens with spaces. Returns the
  character-separated form (no numeric reconstruction).
- `save(path: str)` — JSON with `{"tokenizer_class": "v3", "token_to_id": {...}}`.
- `load(cls, path: str)` — loads JSON, auto-detects class from
  `"tokenizer_class"` field.

**Verification:** `uv run pytest tests/test_tokenizer_v3.py -v` — all tests
pass. Manually inspect vocabulary dump to confirm ~100 tokens.

---

## Step 4: Integration — tokenizer + corpus with train.py

**What:** Verify that a v3 corpus + TokenizerV3 can be consumed by `train.py`.
Add a `--tokenizer` flag to `train.py` for passing a pre-fitted tokenizer path.

**Files modified:**
- `train.py` — add `--tokenizer` flag (path to pre-fitted tokenizer JSON).
  When provided, load the tokenizer instead of fitting a new one. Auto-detect
  class from the JSON's `"tokenizer_class"` field (absent = WordTokenizer,
  "v3" = TokenizerV3). When not provided, behavior is unchanged (fit
  WordTokenizer on corpus text). This preserves full backward compatibility.

**Files created:**
- `tests/test_integration_train.py`

**Tokenizer dispatch protocol:** The `--tokenizer` flag is the integration
point between v3 corpus generation and training. The v3 pipeline works as:
1. `generate_corpus_v3.py` produces phase1.txt, phase2.txt
2. The caller (sweep script or user) fits `TokenizerV3` and saves it
3. `train.py --tokenizer tokenizer.json` loads it and skips fitting
4. `tension_probe_v3.py` loads the same tokenizer from the run directory

Both `TokenizerV3` and `WordTokenizer` save/load to JSON. The `"tokenizer_class"`
field in the JSON determines which class is instantiated on load.

**Test first:**

```
test_v3_corpus_trains_without_error
  Given a digit-format corpus, TokenizerV3 fitted and saved to JSON
  When train.py is called with --tokenizer tokenizer.json for 50 steps
       with a small model (n_embd=16, n_layer=2)
  Then training completes without error and final loss < 10.0
  (setup: generate corpus, fit tokenizer, save, call train_phase directly)

test_v3_training_loss_decreases
  Given a digit-format corpus and small model
  When trained for 200 steps
  Then loss at step 200 < loss at step 10
  (validates that the model learns from character-level numeric tokens)

test_v2_invocation_still_works
  Given v2 corpus generated by generate_corpus.py (ordinal format)
  When train.py is called WITHOUT --tokenizer flag
  Then training completes using WordTokenizer (default behavior unchanged)
  And checkpoint is valid

test_tokenizer_flag_loads_correct_class
  Given a TokenizerV3 saved to JSON with "tokenizer_class": "v3"
  When train.py loads it via --tokenizer
  Then the loaded tokenizer is a TokenizerV3 instance

test_checkpoint_saves_tokenizer
  Given training completes with --tokenizer flag
  When checkpoint directory is inspected
  Then tokenizer.json exists and contains "tokenizer_class": "v3"
  (so tension_probe_v3.py can load it later)

test_checkpoint_contains_expected_keys
  Given training completes
  When checkpoint is loaded
  Then it contains "model_state", "config", "schedule", "args"
  And config.vocab_size matches TokenizerV3.vocab_size

test_v3_ordinal_corpus_trains_like_v2
  Given ordinal-format corpus from generate_corpus_v3.py, TokenizerV3 fitted
  When trained with same hyperparameters as v2
  Then final loss is within same order of magnitude as v2 training

test_digit_sentence_length_within_block_size
  Given a digit-format corpus preprocessed by TokenizerV3
  When max token count of any description sentence is measured
  Then max < 40 tokens (well within block_size=64 default)
  (FEASIBILITY: ensures training windows contain multiple sentences)

test_empty_phase2_canon_only_succeeds
  Given --n-eris 0 (empty phase2.txt), schedule="canon-only"
  When training runs
  Then completes without error (phase2 data is not used)

test_empty_phase2_curriculum_raises_error
  Given --n-eris 0 (empty phase2.txt), schedule="curriculum"
  When training reaches phase2
  Then raises ValueError with message about empty phase2 data
  (fail loudly rather than crashing on randint(0, -1))
```

**Statistical validation test:**
```
test_training_converges_on_digit_corpus
  Given digit-format corpus, small model (n_embd=16, n_layer=2), 500 steps
  When trained with 5 different seeds
  Then all 5 final losses < 8.0
  (validates character-level tokens don't prevent convergence.
   Thresholds should be calibrated empirically after initial implementation —
   run once, record actual losses, set threshold at max + 3*std.)
```

**Implementation approach:**

Add to `train.py`:
- `--tokenizer` optional CLI arg (default: None). When provided, load
  tokenizer from JSON path instead of fitting. Detection logic:
  ```python
  if args.tokenizer:
      data = json.loads(Path(args.tokenizer).read_text())
      if data.get("tokenizer_class") == "v3":
          from tokenizer_v3 import TokenizerV3
          tokenizer = TokenizerV3.load(args.tokenizer)
      else:
          tokenizer = WordTokenizer.load(args.tokenizer)
  else:
      tokenizer = WordTokenizer.fit(text)  # existing behavior
  ```
- Add empty-data check before phase2 training: if `len(data_p2) == 0` and
  schedule is "curriculum" or "mixed", raise `ValueError("Phase 2 data is
  empty — cannot train with schedule '{schedule}'")`.
- Save the tokenizer JSON to the output directory alongside the checkpoint.

**Verification:** Run v2 Quick Start command from CLAUDE.md and confirm
identical behavior. Then run v3 corpus with `--tokenizer` flag.

---

## Step 5: Tension probe stub for continuous features

**What:** Create `tension_probe_v3.py` with the adapted prompt format for
continuous features, reusing the existing classification logic. Handles digit
and ordinal formats; spelled-out format raises NotImplementedError (Phase 2).

**Files created:**
- `tension_probe_v3.py`
- `tests/test_tension_probe_v3.py`

**Test first:**

```
test_feature_prompt_digit_format
  Given name="P10", feats=(38.08, 11500, 39.2), representation="digit"
  When feature_prompt_v3 is called
  Then returns "P10 has mass 38.08 kg diameter 11500 km orbit 39.2 AU . P10 is a"

test_feature_prompt_ordinal_format
  Given name="P10", feats=(38.08, 11500, 39.2), representation="ordinal"
  When feature_prompt_v3 is called
  Then returns "P10 has mass small diameter large orbit distant . P10 is a"

test_feature_prompt_spelled_raises
  Given representation="spelled"
  When feature_prompt_v3 is called
  Then raises NotImplementedError

test_classify_tension_cell
  Given P_planet=0.34, P_dwarf=0.28, thr_planet=0.31, thr_dwarf=0.25
  When classify is called
  Then returns "tension"

test_classify_canon_cell
  Given P_planet=0.5, P_dwarf=0.05, thr_planet=0.31, thr_dwarf=0.25
  When classify is called
  Then returns "canon"

test_classify_swap_cell
  Given P_planet=0.05, P_dwarf=0.4, thr_planet=0.31, thr_dwarf=0.25
  When classify is called
  Then returns "swap"

test_classify_nothing_cell
  Given P_planet=0.05, P_dwarf=0.05, thr_planet=0.31, thr_dwarf=0.25
  When classify is called
  Then returns "nothing"

test_tension_index_at_threshold
  Given P_planet=thr_planet, P_dwarf=thr_dwarf
  When tension_index is called
  Then returns 1.0

test_tension_index_below_threshold
  Given P_planet=thr_planet/2, P_dwarf=thr_dwarf
  When tension_index is called
  Then returns 0.5

test_tension_index_zero_threshold_returns_none
  Given thr_planet=0.0
  When tension_index is called
  Then returns None

test_label_probabilities_returns_valid_distribution
  Given the trained_small_model fixture
  When label_probabilities_v3 is called with a digit-format prompt
  Then returned probabilities are in [0, 1] and keys include "planet" and "dwarf"

test_probe_prompt_fits_block_size
  Given a digit-format probe prompt for P10
  When preprocessed and tokenized
  Then token count < block_size (read from checkpoint config)
  (spec §5.3: raise error if prompt exceeds block_size)

test_calibration_with_no_dwarf_anchors
  Given mode="unlabeled" (no E* labels in training), no dwarf anchors available
  When calibration_thresholds_v3 is called
  Then thr_dwarf is None and P10's classification uses only thr_planet
  And cell is "n/a" for dwarf-related metrics

test_output_json_schema
  Given a full probe run via CLI
  When output JSON is loaded
  Then it matches schema: thresholds, anchors, targets with per-target
       P_planet, P_dwarf, cell, tension_index
  And contains "representation" field
```

**Statistical validation tests:**
```
test_classification_logic_matches_v2
  Given identical inputs (P_planet, P_dwarf, thresholds)
  When v3 classify and v2 tension_probe.classify are called
  Then results are identical for all 4 quadrants of the 2x2 grid

test_tension_index_monotonic_in_probabilities
  Given fixed thresholds and P_dwarf=thr_dwarf
  When P_planet increases from 0.01 to 2*thr_planet in 20 steps
  Then tension_index increases monotonically
  (validates the min(ratio_p, ratio_d) formula behaves as expected)

test_tension_index_symmetric
  Given P_planet/thr_planet = P_dwarf/thr_dwarf = r
  When tension_index is called for r in [0.1, 0.5, 1.0, 1.5, 2.0]
  Then tension_index == r for all values
  (validates the min() formula when both ratios are equal)
```

**Implementation:**

- Import `classify` and `tension_index` from `tension_probe.py` (these are
  pure functions with no v2-specific dependencies).
- `feature_prompt_v3(name, feats, representation)` — builds probe prompt.
  Uses `describe_continuous` from `generate_corpus_v3.py` for the feature
  portion, appends ` {name} is a`. Checks token count against block_size
  (from checkpoint config) and raises ValueError if it exceeds.
- `label_probabilities_v3(model, tokenizer, prompt, device)` — encodes with
  `TokenizerV3`, forward pass, extract softmax probabilities for label tokens.
- `calibration_thresholds_v3(model, tokenizer, ents, representation, device)`
  — anchors: P1..P9 for planet, E* with labels for dwarf. If no dwarf anchors
  exist (mode=unlabeled), thr_dwarf=None and classifications are "n/a".
- `load_model_and_tokenizer(ckpt_path, device)` — loads checkpoint, reads
  tokenizer.json from same directory, auto-detects class.
- CLI: `--ckpt`, `--entities`, `--out`, `--representation`, `--device`,
  `--targets`.

**Verification:** Run on checkpoint from Step 4's `trained_small_model`.
Output JSON has valid structure and probabilities.

---

## Step 6: End-to-end smoke test

**What:** Integration tests running the full pipeline both as function calls
(fast) and as CLI subprocesses (validates the user-facing commands).

**Files created:**
- `tests/test_smoke_e2e.py`

**Test first:**

```
test_full_pipeline_digit_format
  Given seed=42, representation="digit", mode="dwarf", schedule="curriculum"
  When:
    1. generate_corpus_v3 produces phase1.txt, phase2.txt, entities.json
    2. TokenizerV3.fit on phase1 + phase2 text, saved to tokenizer.json
    3. train_phase on phase1 (100 steps), then phase2 (50 steps)
    4. tension_probe_v3 runs on the checkpoint
  Then:
    - Training completes without error
    - Final loss < 8.0
    - Probe output JSON exists and has valid schema
    - P10 has a classification (canon, swap, tension, or nothing)
    - All probabilities are in [0, 1]

test_full_pipeline_ordinal_format
  Given seed=42, representation="ordinal", mode="dwarf", schedule="curriculum"
  When same pipeline is run
  Then same validation passes

test_full_pipeline_unlabeled_mode
  Given seed=42, representation="digit", mode="unlabeled", schedule="curriculum"
  When pipeline is run
  Then training completes
  And probe handles missing dwarf anchors gracefully (thr_dwarf=None)

test_full_pipeline_canon_only
  Given schedule="canon-only"
  When pipeline is run
  Then training uses only phase1 data, probe still runs

test_full_pipeline_mixed
  Given schedule="mixed"
  When pipeline is run
  Then training concatenates phase1+phase2, single training phase

test_pipeline_reproducibility
  Given seed=42, representation="digit"
  When full pipeline is run twice on CPU with torch.use_deterministic_algorithms(True)
  Then probe output JSON values match within tolerance (atol=1e-6)
  (float comparison with tolerance, not byte-identical, to handle platform diffs)

test_full_pipeline_cli_subprocess
  Given seed=42, representation="digit", mode="dwarf", schedule="curriculum"
  When the full v3 pipeline is run as sequential subprocess calls:
    python3 generate_corpus_v3.py --out-dir {tmp}/data --mode dwarf
        --representation digit --seed 42
    python3 -c "fit tokenizer, save to {tmp}/data/tokenizer.json"
    python3 train.py --data-dir {tmp}/data --out-dir {tmp}/run
        --schedule curriculum --tokenizer {tmp}/data/tokenizer.json
        --phase1-steps 100 --phase2-steps 50 --n-embd 16 --n-layer 2
    python3 tension_probe_v3.py --ckpt {tmp}/run/ckpt.pt
        --entities {tmp}/data/entities.json --out {tmp}/run/tension.json
        --representation digit
  Then all commands exit 0
  And {tmp}/run/tension.json exists and has valid schema
  (CLI-level e2e test — catches argument-name mismatches between scripts)
```

**Statistical validation test:**
```
test_digit_format_model_learns_categories
  Given digit-format corpus, mode="planet", schedule="canon-only",
        n_embd=16, n_layer=2, 1000 steps
  When trained and probed
  Then P(planet | P1) > 0.15
  (spec §8.2 verification. Threshold is above uniform 1/vocab_size ~ 0.01.
   Set conservatively for a tiny model; calibrate empirically after first run.)
```

**Implementation:**

Function-call tests use the `trained_small_model` fixture from `conftest.py`
for speed (100/50 steps, n_embd=16, n_layer=2). Each test completes in
< 30 seconds. The CLI subprocess test (`test_full_pipeline_cli_subprocess`)
runs the actual user-facing commands end-to-end and is the definitive walking
skeleton check.

**Verification:** `uv run pytest tests/test_smoke_e2e.py -v` — all tests
pass. Total time < 3 minutes.

---

## Step 7: v2 backward-compatibility check

**What:** Verify that changes to `train.py` have not broken the v2 pipeline.

**Files created:**
- `tests/test_v2_compat.py`

**Test first:**

```
test_v2_corpus_generation_unchanged
  Given v2 generate_corpus.py with --seed 42 --mode dwarf
  When run as subprocess
  Then phase1.txt and phase2.txt are produced without error

test_v2_train_pipeline_unchanged
  Given v2 corpus from above
  When train.py is called WITHOUT --tokenizer flag, --schedule canon-only
  Then training completes with WordTokenizer (default, unchanged behavior)
  And checkpoint is produced

test_v2_tension_probe_unchanged
  Given v2 checkpoint from above
  When tension_probe.py (v2) is called
  Then probe output is valid JSON with expected schema

test_v2_quick_start_command
  Given the exact command from CLAUDE.md Quick Start:
    python3 generate_corpus.py --out-dir {tmp}/data --mode unlabeled
    python3 train.py --out-dir {tmp}/runs/canon_only --schedule canon-only
                     --data-dir {tmp}/data
  When run as subprocess
  Then both commands exit 0 and checkpoint exists
```

**Implementation:** All tests use subprocess calls (not imports) to verify
the exact CLI contract. Training uses minimal steps (100) for speed.

**Verification:** `uv run pytest tests/test_v2_compat.py -v` — all pass.

---

## Exit checklist

All criteria must pass before Phase 1 is considered complete:

- [ ] `uv run pytest` — all tests pass (Steps 0–7)
- [ ] `uv run pytest --co` — confirms test count >= 35
- [ ] `generate_corpus_v3.py --representation digit --mode dwarf --seed 42`
      produces phase1.txt, phase2.txt, entities.json with digit-format features
- [ ] `generate_corpus_v3.py --representation ordinal --mode dwarf --seed 42`
      produces ordinal-format output structurally matching v2
- [ ] `TokenizerV3.fit()` on a digit corpus produces vocab_size ~100
- [ ] Vocabulary is identical regardless of which representation generated the
      corpus (union vocabulary per §3.1a.3)
- [ ] A digit-format corpus trains successfully with `train.py --tokenizer
      tokenizer.json` for 500 steps, loss < 5.0
- [ ] Max digit-format sentence length < 40 tokens after preprocessing
      (fits within block_size=64)
- [ ] `tension_probe_v3.py` produces valid JSON with classifications for P10
- [ ] Full pipeline CLI smoke test passes (generate -> train -> probe via
      subprocess calls)
- [ ] v2 backward-compatibility tests pass (no v2 regressions)
- [ ] v2 Quick Start from CLAUDE.md works unchanged
- [ ] No new files exceed 200 lines (repo convention)
- [ ] All new functions have type hints and brief docstrings
- [ ] Statistical validation tests cover: distribution sampling (within 3 SE),
      classification logic (matches v2), tension_index monotonicity,
      training convergence (loss decreases)
- [ ] TODO for `--prototypes-json` is present in `generate_corpus_v3.py`
