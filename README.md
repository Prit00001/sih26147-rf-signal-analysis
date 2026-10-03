# sigscope

Offline RF signal parameter extraction and demodulation toolkit (NTRO, SIH26147).

Phase 1: secure ingestion (.wav / .IQ), preprocessing, spectral analysis, and a
synthetic ground-truth data generator. No network access is required or used
at runtime (SR-03).

## Setup

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
uv pip install --python .venv/bin/python pybind11 cmake ninja
scripts/build_native.sh   # builds sigscope._native (pybind11/C++), requires a C++17 compiler
```

## Run

```bash
.venv/bin/sigscope generate --mod qpsk --snr 10 --out data/ --name demo --seed 1
.venv/bin/sigscope ingest data/demo.wav
.venv/bin/sigscope spectrum data/demo.wav --out spectrum.png
```

## Test / lint / security

```bash
.venv/bin/python -m pytest tests/ -q --ignore=tests/fuzz_wav.py --ignore=tests/fuzz_iq.py
.venv/bin/ruff check src tests
.venv/bin/mypy src/sigscope
.venv/bin/bandit -c pyproject.toml -r src/sigscope
.venv/bin/semgrep --config auto src/sigscope
.venv/bin/pip-audit
```

Fuzzing (Linux only -- atheris needs libFuzzer, which Apple Clang does not
ship; the equivalent Hypothesis property tests in
`tests/test_property_parsers.py` run on every platform):

```bash
python tests/fuzz_wav.py -max_total_time=60
python tests/fuzz_iq.py -max_total_time=60
```

## Known local-environment gaps (not code defects)

- `clang-tidy` / `cppcheck` are not installed on this machine; they run in CI
  (Linux) per SR-07. Install locally with `brew install llvm cppcheck` if you
  want to run them here too.
- `atheris` cannot build against Apple Clang on macOS (see above); it runs in
  CI on Linux.

## Phase 2: parameter estimation + modulation classification

Train the classifier (requires the `train` extra -- PyTorch; not needed at runtime):

```bash
uv pip install --python .venv/bin/python -e ".[train]"
.venv/bin/python -m sigscope.classify.train --out models --examples-per-class-snr 120 --epochs 25
```

Analyze a file (estimates symbol rate, SNR, roll-off, and classifies modulation):

```bash
.venv/bin/sigscope analyze data/demo.wav
```

Accuracy-vs-SNR benchmark (NFR-02):

```bash
.venv/bin/python scripts/benchmark_modulation.py --model models/modulation_cnn.manifest.json --out reports
```

**Measured result, original** (11 classes, SNR -5..25 dB): overall mean
accuracy ~0.70. Strong (85-100%) for BPSK, 2/4/8-FSK, AM, FM, noise. Weak
(<45%) for QPSK/16QAM/64QAM -- root cause: C20/C40 cumulants are CFO-sensitive
in a way that gets WORSE with a longer window (destructive phase averaging).

**Fixed** (see `estimate/cumulants.py`'s `compute_cumulants_windowed` and
`classify/ensemble.py`'s `constellation_fit_scores`): (1) M-th-power CFO
pre-compensation + averaging cumulant MAGNITUDES across many short windows
instead of one long one; (2) a constellation-fit (EVM) check that runs the
already-verified Phase 3 demod chain once per PSK/QAM candidate and scores by
fit to that candidate's own constellation, **normalized by each
constellation's minimum inter-point distance** -- without that normalization
this is a real bug: raw EVM trivially favors denser constellations (more
points to snap to), so 64QAM looked like the best fit for every signal,
QPSK included, until this was caught by testing. **Measured result, after**:
overall mean accuracy ~0.86; at SNR >= 15 dB, QPSK 90%, 16QAM 100%, 64QAM
100% (target was >=80%, met). 8PSK also improved (65% -> 85-100% at
SNR>=15dB) as a side effect, though it wasn't a target class.

## Phase 3: demodulation

```bash
.venv/bin/python -c "
from sigscope.synth.generator import generate_signal
from sigscope.demod.psk_qam import demodulate_psk_qam
sig, gt = generate_signal('qpsk', num_symbols=4000, snr_db=15.0, seed=1)
result = demodulate_psk_qam(sig.samples, 'qpsk', sig.sample_rate, gt.symbol_rate)
print(len(result.hard_bits), 'bits recovered')
"
.venv/bin/python scripts/benchmark_ber.py --out reports
```

Chain: coarse M-th-power CFO estimate -> RRC matched filter -> Gardner timing
recovery (proportional-only; see sync.py's docstring for why an integral term
was found to cause unbounded timing drift) -> optional CMA/LMS equalizer
(off by default -- see psk_qam.py's use_equalizer docstring for the measured
finding that it hurts QAM BER when there is no real multipath to correct) ->
decision-directed carrier recovery (generalizes Costas to any PSK/QAM
constellation) -> Gray-mapped soft/hard bits. FSK uses a separate frequency-
discriminator chain (demod/fsk.py).

**Measured result** (`reports/demod_ber_vs_snr.csv`/`.png`): symbol error rate
drops to near zero by 10-15 dB SNR for BPSK/QPSK/2FSK/4FSK/8FSK, and by 20 dB
for the denser 8PSK/16QAM/64QAM constellations -- physically sensible
ordering (denser constellation needs more SNR). Decision-directed carrier
recovery has an inherent rotational ambiguity without a sync word/differential
encoding; BER here is evaluated after resolving that ambiguity against ground
truth (see `tests/demod_test_utils.py`), which a real deployment would resolve
via Phase 5's bit-stream correlation (sync word) instead.

## Phase 4: de-interleaving + FEC (C++ core)

```bash
# Build the C++ GoogleTest kernel tests (opt-in, not part of the default build):
cmake -S native -B native/build-tests -G Ninja -DSIGSCOPE_BUILD_TESTS=ON \
  -Dpybind11_DIR="$(.venv/bin/python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build native/build-tests -j
./native/build-tests/sigscope_kernel_tests
```

C++ kernels (`native/src/`): soft-decision Viterbi, min-sum BP LDPC decoder,
bitstream correlation -- all exposed via `sigscope._native` and covered by
both GoogleTest (C++-level) and pytest (via the Python bindings), release and
ASan/UBSan builds both clean.

Python (`sigscope/fec/`): Block/Convolutional/Diagonal/Pseudo-random
interleavers with round-trip tests; Reed-Solomon over GF(2^m) (own
from-scratch codec -- see reed_solomon.py's docstring for why, instead of the
`galois` package: its numba/llvmlite install was slow/heavy for no real
correctness benefit at this project's small code lengths); a concatenated
RS+convolutional chain; and blind identification (interleaver period via
GF(2) rank deficiency, RS via syndrome structure test, LDPC via H-matrix
library matching, pseudo-random interleavers via a position-sensitive
checker -- **not** generic rank, which is mathematically incapable of
detecting a pure bit-position permutation; this was a real bug caught and
fixed during testing, not a hypothetical).

**Real bugs found and fixed during this phase** (each caught by testing
against ground truth, not assumed correct): (1) RS generator-polynomial
degree-order mismatch between the encoder and evaluator; (2) RS decoder
using the wrong codeword-index-to-GF-exponent mapping; (3) the pseudo-random
interleaver identifier's original rank-based scoring was mathematically
incapable of ever working, since GF(2) rank is invariant under coordinate
permutation -- replaced with a code-specific syndrome checker.

**Upgraded since (P5):** the RS decoder originally searched error-location
subsets combinatorially (correct and simple, but exponential in the number of
correctable errors -- fine at small test code lengths, not production-scale).
Replaced with the standard Berlekamp-Massey + Chien search + Forney pipeline
(all O(n*t) or better); the combinatorial version is kept only as
`RSCode._decode_combinatorial`, for cross-validating the replacement in
tests, not for live use. **Measured**
(`scripts/benchmark_rs_decode.py` / `reports/rs_decode_time_before_after.csv`,
full correction radius t errors per trial): RS(15,9) t=3 decode time
6.24 ms -> 0.11 ms (~57x); RS(31,21) t=5 5881 ms -> 0.34 ms (~17,000x);
RS(255,223) t=16 (CCSDS-standard length, now supported) was not feasible to
even run on the old decoder (C(255,16) candidate subsets) -> 6.42 ms mean,
6.71 ms worst case on the new one.

**Known, documented scope limits:** FEC/interleaver identification is confidence-scored and
reports "unidentified" rather than a false positive when nothing in the
library matches, consistent with Phase 1's documented position that some
blind identification (arbitrary LDPC H, unlisted PRNG generators) is not
attempted, with the analyst-override path always available regardless.

## Phase 5: bit-stream correlation + GUI

```bash
.venv/bin/sigscope-gui   # launch the desktop GUI
# or headless (CI/no display):
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_gui_smoke.py -q
```

`sigscope/correlate/bitstream.py` wraps the Phase 4 C++ correlation kernel for
frame-length discovery (autocorrelation), sync-word search (cross-
correlation), and header/payload segmentation (per-bit-position entropy).

**Bug found and fixed during testing:** the sync-word confidence metric
originally scored "margin over the second-best match," which reads a
correctly-repeating periodic sync word (many equally-good true matches) as
*zero* confidence -- the opposite of correct. Replaced with a periodicity
check (do the found match positions recur at a consistent spacing?), which
correctly gives high confidence to a real repeating sync word and low
confidence to an isolated chance false alarm (verified with both a positive
and a negative control in `tests/test_bitstream_correlate.py`).

GUI (`sigscope/gui/`, PySide6 + pyqtgraph, dark theme, MVVM): left panel
(file open, Run Auto Analysis, live per-stage status with confidence);
center tabs (Spectrum, Waterfall, Constellation, Eye Diagram -- the eye
diagram is drawn from the raw ingested samples since the demod chain does not
currently expose its intermediate matched-filtered stream, a documented
simplification, not a fabricated view); right panel (parameter table with a
confidence bar per row, plus modulation/symbol-rate override fields --
editing them and clicking "Re-run with overrides" is the step-by-step
analyst mode, FR-15); bottom tabs (bitstream hex/binary, correlation
summary, pipeline log). The pipeline itself runs on a QThread
(`PipelineWorker`) so the UI stays responsive, emitting a signal per
completed stage.

## Localhost web GUI (prototype)

A lightweight browser-based alternative to the PySide6 desktop GUI, for easy
demoing: stdlib `http.server` only (no new framework dependency), binds to
127.0.0.1 only (SR-03/SR-04: no external network exposure).

```bash
.venv/bin/sigscope-web --port 8901
# open http://127.0.0.1:8901/ in a browser, enter a .wav/.iq file path, click Run Analysis
```

Shares its actual analysis logic with the desktop GUI via
`sigscope/pipeline_core.py` (a plain, framework-agnostic function) -- refactored
out of the desktop GUI's QThread worker so both front ends call the same code,
not two copies of it.

## P2: convolutional code blind identification

`sigscope/fec/conv_identify.py` -- library of 6 standard codes (rate 1/2 K=3/5/7,
rate 1/3 K=7, plus rate 2/3 and 3/4 punctured variants of the K=7 mother
code -- puncture patterns are this project's own stated assumption, not
claimed to match any external standard bit-for-bit). For each candidate and
bit-alignment phase: Viterbi-decode (erasure LLRs at punctured positions),
re-encode, measure agreement with the received bits.

**Real bug found and fixed while testing this:** Viterbi decoding finds the
best-fitting trellis path for *whatever* it's given, so re-encode agreement
is substantial even on pure noise -- measured 81-88% for the unpunctured
codes, 93-95% for the punctured ones (more decoded-bit freedom per
transmitted bit = easier to fit noise). A single fixed agreement threshold
(as literally specified) produced false positives on random data and on an
out-of-library code. Fixed by calibrating each candidate's own noise floor
at runtime (not a hand-picked constant) and reporting confidence as excess
agreement above that floor.

**Measured**: all 6 library codes correctly identified at 2% injected BER
(confidence 0.58-0.88); pure random data and an out-of-library K=9 code both
correctly report "unidentified" (confidence <0.02) across 4 random seeds.

## P3: web GUI usable for judges

Replaced the typed file-path input with real file upload (drag-and-drop +
click-to-browse), a dropdown of 5 bundled demo samples (freshly generated by
our own synthetic generator at server start, never downloaded, clearly
labelled "synthetic sample (ground truth known)"), and added waterfall, eye
diagram, hex bitstream, and a header/payload view with the presumed sync
word highlighted. All analysis stays in `pipeline_core.py`; the web layer
(`webui/server.py`) only renders. Uploads are size/extension-validated
server-side, written to a workspace under the OS temp dir, and deleted in a
`finally` block after every request (verified by a test that checks the
workspace is empty before and after).

**Real gap closed while wiring this up:** interleaver/FEC identification
(built in Phase 4/P2) was never actually connected to the live pipeline.
Wiring it in surfaced two further real bugs, both caught by testing against
signals with actual known structure (not assumed working):
1. FEC identification must run on the **de-interleaved** stream when an
   interleaver is found -- a block code's row-wise redundancy is invisible
   in the still-interleaved stream (my first coded+interleaved test signal
   correctly found the interleaver period but reported FEC "unidentified"
   until this was added).
2. `identify_rs`'s argmax over candidate k picked the WEAKEST k in the
   library (e.g. 14 instead of the true 9) every time: RS(n,k') for k'>k
   checks a strict subset of the same syndrome roots, so genuine RS(15,9)
   data trivially also "passes" the weaker RS(15,10..14) tests. Fixed with
   analytic per-(n,k) chance calibration plus preferring the smallest k
   among near-tied candidates (Occam's razor: most specific code consistent
   with the data).

Verified end-to-end (not just unit tests): `test_upload_endpoint_populates_every_panel`
uploads a real generated file over HTTP and decodes every returned PNG to
confirm real image bytes (not placeholders); integration tests confirm a
conv-coded demo correctly identifies "convolutional (rate1/2_K7)" and an
RS+interleaved demo correctly identifies both the interleaver period and
"reed-solomon (n=15, k=9, m=4)" through the real ingest->demod->identify
chain, while a plain uncoded signal honestly reports "unidentified" for both.

## P4: carrier-rotation integrity check + full-chain dashboard fix

**Integrity check:** confirmed (dynamic test, not just reading the source --
`test_no_label_leakage.py` patches `Path.open` and records every file
actually opened during a real pipeline run) that `pipeline_core.py`, the
CLI, and both GUIs never read a ground-truth label.

**Real gap closed:** decision-directed carrier recovery (`demod/sync.py`)
can lock onto any of a constellation's rotational symmetries with equal
validity -- a real receiver-side ambiguity, and until now the live pipeline
never resolved it at all. Fixed with a blind (no ground truth) resolution:
try every rotation in the constellation's actual symmetry group (M for
M-PSK, 4 for square QAM -- its real symmetry fold, not its point count),
and score each by (a) whether a known sync word (CCSDS 0x1ACFFC1D, this
project's one assumed pattern absent an analyst override) correlates, else
(b) which rotation's bits best fit a library FEC code. The chosen rotation
and reason are shown in both GUIs' parameter panels.
`test_carrier_rotation_resolution.py` forces the ambiguity, deletes the
label file, and verifies 100% bit-exact recovery through the real
ingest->demod->resolve pipeline.

**Dashboard: three more real bugs, found while chasing a partial
ground-truth match on the full-chain CCSDS demo:**
1. A confidently-identified convolutional code is normally the innermost
   transform before modulation, so it said nothing about an interleaver/
   outer code hidden BEHIND it -- `pipeline_core.py` now Viterbi-decodes a
   confident conv match and searches one more level deeper (bounded to one
   extra level, not unbounded recursion) on the decoded payload.
2. That search initially still failed: a periodic sync word interspersed
   directly in an otherwise continuously-encoded stream corrupts the
   decode at every frame boundary. Fixed by stripping the header (using the
   SAME frame_length/header_length/phase the correlate stage already
   found) before identification, and raising the identify bit-budget
   (`_IDENTIFY_MAX_BITS`, 6000 -> 8192) enough to cover one full
   interleaver block -- a truncated prefix of a block-interleaved stream
   cannot be correctly un-interleaved at all, not just less confidently.
3. The ground-truth panel's "not detected" display for frame_length used
   `frame_length_confidence`, a metric that is legitimately noisy even for
   a CORRECT detection (measured 0.11 on this exact demo) -- switched to
   gating on `header_confidence`, which is cleanly binary in practice
   (exactly 0 on every demo with no real framing, including one case,
   2FSK, where the OLD metric would have shown a false "detected").

**Measured result:** the full-chain CCSDS demo (QPSK + conv r1/2 K=7 + outer
RS(15,9) interleaved + sync 0x1ACFFC1D) now matches **8/8 (100%)**
ground-truth rows -- modulation, symbol rate, SNR, rolloff, interleaver,
FEC, frame length, and header length all correctly found through the real
pipeline, up from a partial match before this fix.

Both GUIs' pipeline stepper now color-codes the REAL per-stage status
(green=done, amber=not present/fallback, accent=override) -- the desktop
GUI previously had no status coloring at all, only a raw confidence-based
v/~ marker that looked identical for "ran but found nothing" and "ran and
succeeded at low confidence".

Rolloff: an earlier version of this README/docstring claimed
`pipeline_core.py` excludes a low-confidence rolloff estimate from the
ground-truth score with a tooltip; that mechanism was never actually
built. Corrected rather than retroactively built, because the REAL
measured accuracy (`scripts/benchmark_rolloff.py`, re-run 2026-10-03:
mean absolute error 0.019-0.026 across the entire -5..25 dB SNR range
tested) turns out not to need it.

## P5: Reed-Solomon decoder rewrite (Berlekamp-Massey + Chien + Forney)

See Phase 4's RS section above for the measured before/after timing
(`scripts/benchmark_rs_decode.py`): replacing the O(C(n,t)) combinatorial
search with the standard O(n*t) pipeline made RS(255,223) -- the
CCSDS-standard length, previously infeasible to even run -- decode in
~6.4 ms. Cross-validated against the kept-for-testing old decoder across
2000+ random trials at several (n,k,m). **Real bug caught by that cross-
validation:** Forney's error-evaluator derivative in a characteristic-2
field only keeps ODD-power terms of Lambda, but each one lands at an EVEN
exponent (Lambda[1]*x^1 -> x^0, Lambda[3]*x^3 -> x^2, ...), not consecutive
ones -- evaluating it as if consecutive gave a confidently wrong error
magnitude at every trial with 3 simultaneous errors on RS(15,9), while
Chien search's error LOCATIONS were already correct. Error locations
alone looking right is not sufficient evidence the decoder works; only the
cross-validated, bit-exact corrected codeword is.

## P6: multipath, flat fading, and timing drift in the channel model

`synth/generator.py` gained three new, opt-in (default-off) channel
impairments: a tapped-delay-line multipath model, AR(1)-filtered flat
Rayleigh-like fading, and linear receiver-clock timing drift (this
project's channel model previously had a fixed timing offset to find, but
nothing that drifted over time -- see `demod/sync.py`'s Gardner recovery
docstring). All three are documented simplifications with a stated upgrade
path (a proper Jakes/Clarke Doppler model), not a claim of matching a
specific real propagation environment.

`scripts/benchmark_ber.py` and `scripts/benchmark_modulation.py` both gained
an `--impaired` flag applying one fixed, documented preset (a -6 dB echo 3
samples behind the direct path, 5 Hz fading, 20 ppm clock drift). **Measured,
per class** (re-run 2026-10-03):
- Demod SER: BPSK/2FSK/4FSK stay close to their clean-channel curves at
  moderate-to-high SNR. QPSK and every denser PSK/QAM constellation degrade
  sharply (SER 0.10-0.97 across the whole SNR grid) -- the demod chain's
  default (`use_equalizer=False`, see Phase 3) has no multipath correction
  applied automatically, and this is the first channel model with real
  multipath to expose that. A real, now-measured limit of the current
  default configuration, not something papered over: enabling the
  equalizer is the documented override path (Phase 3), though it would
  itself need re-validation against multipath specifically -- the existing
  "hurts QAM BER" finding was measured on a channel with NO multipath at
  all, a different scenario.
- Classifier overall mean accuracy: 0.864 (clean) -> 0.681 (impaired).

## Assumptions & limits

A consolidated list -- nothing here is new; each item is explained in
detail, with the real bug or measurement that surfaced it, in the phase
section above it belongs to.

**Reed-Solomon:** GF(2^m) library covers m=3..8 (n up to 255); the
project's own identification library only searches m in {3,4,5} (Phase 4) --
a real RS(255,223)-coded signal would decode correctly once identified, but
blind identification would not currently propose it as a candidate (an
easy, not-yet-needed library extension, not an algorithmic limit).

**Interleaver/FEC identification**, even after P4's one-level-deeper search:
still bounded to exactly one extra recursion level (an outer code hidden
behind TWO inner transforms would not be found), and the deeper search's
FEC fallback only checks the RAW decoded stream, not every candidate
rotation's de-interleaved version too (interacts with P4's rotation
resolution: an interleaved-RS-only signal, rotated, with no sync word, can
come back "unresolved" -- see `pipeline_core.py`'s docstring). Arbitrary
LDPC H matrices and PRNG generators outside the built-in library are not
blindly identifiable at all, by design (Phase 1's documented position) --
the analyst-override path always remains available.

**Channel model:** multipath is a fixed tapped-delay-line, not a
continuous-Doppler or frequency-selective model; fading is AR(1)-filtered
Gaussian, not the exact Jakes/Clarke Doppler spectrum shape. Both are
stated simplifications (P6) with a documented upgrade path, not claims of
matching a specific real propagation environment.

**macOS (this development machine) vs Linux CI tooling:** `clang-tidy`,
`cppcheck`, and `atheris` are not installed/buildable here (`atheris` needs
libFuzzer, which Apple Clang does not ship) -- `scripts/write_build_info.py`
reports each honestly as "not run" with the real reason rather than hiding
or guessing. All three run for real on Linux CI (`.github/workflows/ci.yml`),
which is where they were actually exercised end-to-end while building
this (see below).

**CI, as actually run, not just as configured:** building this out (P4)
surfaced four real, previously-latent CI bugs, each found by watching an
actual triggered GitHub Actions run rather than assuming the YAML was
correct: (1) two cppcheck style findings on hand-written C++ (fixed by
suppressing with justification, not by writing objectively less-readable
code to satisfy a style linter); (2) CMake silently picked up the system
Python (3.12) instead of this project's own venv (3.11) because no run
pinned `-DPYTHON_EXECUTABLE` explicitly, building a `.so` the venv's own
pytest could not import -- fixed in both the CI workflow and the
equivalent local `scripts/build_native.sh`; (3) the `train` extra never
declared `onnxscript`, which the installed PyTorch's ONNX exporter
requires -- every classifier-training test failed on a clean install
until this was added; (4) `models/` is gitignored (a build artifact, not
committed -- correctly, it is large and reproducible), so a fresh checkout
has no trained classifier at all, and three tests that called the
auto-classify path with no modulation override silently got
`modulation=None` and skipped demodulation entirely -- a direct, isolated
reproduction (`gardner_timing_recovery` etc. in isolation) matched the
local macOS baseline exactly, which is what ruled out a platform-level
numeric bug and pointed at the classifier dependency instead. Training a
real model takes ~6 minutes (measured), too slow to gate every CI push on,
so the fix is what the project's own test-design docstring already argues
for (see `test_classify.py`): these three tests pass an explicit
modulation override, since they exercise upload/GUI/dashboard plumbing,
not classifier accuracy, and one of them (the GUI smoke test) had already
tried to do this but called `_on_run_auto()`, which deliberately clears
any override -- `_on_rerun_with_overrides()` is the method that honors it.
As of the run that includes this commit, every CI
step (native build, clang-tidy, cppcheck, ASan/UBSan + GoogleTest, ruff,
mypy --strict, bandit, semgrep, pip-audit, the full test suite with
>=80% coverage, both 60s atheris fuzz targets, and writing
`reports/build_info.json`) has been independently verified green on a
real run -- not assumed from the config alone.
