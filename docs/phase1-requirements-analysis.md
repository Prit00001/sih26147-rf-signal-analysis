# Phase 1: Requirements Analysis
## SIH 2026 - Problem Statement SIH26147 (NTRO - RF Signal Parameter Extraction and Demodulation Tool)

Document version: 0.1 (Draft, awaiting approval to proceed to Phase 2)
Date: 2026-09-30
Status: DRAFT

---

## 1. Purpose and Scope

This document is the Phase 1 (Requirements Analysis) artifact of the SDLC for a GUI-based
tool that ingests .wav and .IQ recordings of HF/VHF/UHF signals, estimates signal
parameters, demodulates FSK/PSK/QAM, de-interleaves, performs FEC decoding, and
correlates the resulting bitstream to locate headers and payloads. It refines every
clause of the problem statement into testable requirements, defines a Requirements
Traceability Matrix (RTM), states assumptions and constraints (including the honest
limits of blind estimation), and produces a STRIDE threat model and risk register.

Every requirement ID below is treated as binding for the rest of the SDLC. No
requirement is dropped or silently relabelled "future work"; where fully blind automatic
estimation is not feasible, the requirement is still satisfied through the analyst
override path (manual entry or correction of parameters, re-run of downstream stages).

## 2. Module ID Legend

Used throughout the RTM and later phases.

| Module ID  | Name                                   | Primary FR/NFR/SR coverage |
|------------|----------------------------------------|-----------------------------|
| MOD-ING    | Ingestion (.wav, .IQ, SigMF)           | FR-01, FR-02, FR-03, FR-04 |
| MOD-PARAM  | Parameter Estimation                   | FR-05, FR-06, FR-07, FR-08, FR-09 |
| MOD-DEMOD  | Demodulation                           | FR-10 |
| MOD-DEINT  | De-interleaving                        | FR-11 |
| MOD-FEC    | FEC Decoding                           | FR-12 |
| MOD-CORR   | Bit-stream Correlation                 | FR-13 |
| MOD-GUI    | GUI (views, one-click, step-by-step)   | FR-14, FR-15, NFR-03 |
| MOD-DATA   | Synthetic data generator + dataset mgmt| FR-16 |
| MOD-CORE   | Core build/runtime (GNU Radio/Py/C++)  | FR-17, NFR-01, NFR-04, NFR-05 |
| MOD-REPORT | Export (JSON/PDF), batch               | FR-18 |
| MOD-SEC    | Security & audit infrastructure (cross-cutting) | SR-01..SR-12 |
| MOD-QA     | Test/benchmark harness & accuracy reporting | NFR-02, NFR-05 |
| MOD-PKG    | Packaging & release                    | NFR-04, SR-12 |

## 3. Problem-Statement Clause Tags (used in RTM column 2)

| Tag | Clause |
|-----|--------|
| PS-BG        | Background: manual analysis is insufficient for fine-grain parameter extraction |
| PS-DESC-BAND | Description: terrestrial signals in HF, VHF, UHF bands |
| PS-DESC-FMT  | Description: .wav and .IQ store raw info in different formats, need different processing |
| PS-DESC-VARY | Description: recordings from different sensors/locations, parameters vary |
| PS-DESC-TOOLS| Description: solution may use GNU Radio, Python, C++ |
| PS-DESC-SPEC | Description: spectral relationship from training data (.IQ and .wav) for parameter ID |
| PS-DESC-DEMOD| Description: expected solution should demodulate signals |
| PS-GUI-i     | GUI feature (i): identify signal parameters (Fs, modulation, FEC, interleaving) |
| PS-GUI-ii    | GUI feature (ii): demodulate FSK, QAM, PSK |
| PS-GUI-iii   | GUI feature (iii): de-interleaving (Block, Convolutional, Diagonal, Pseudo-random) |
| PS-GUI-iv    | GUI feature (iv): FEC (Viterbi conv, RS, concatenated, LDPC) |
| PS-GUI-v     | GUI feature (v): bit stream correlation |
| PS-EXP       | Expected Solution: GUI improves feature visibility (waterfall, constellation), automated analysis, header/payload identification via correlation |

---

## 4. Functional Requirements - Refined, with Acceptance Criteria

### FR-01 Ingest .wav
Parse RIFF/WAVE container: sample rate, bit depth (8/16/24/32 int, 32/64 float), channel
count, and treat 2-channel files as interleaved I/Q (ch0=I, ch1=Q) when the analyst
flags the file as IQ-in-WAV, otherwise treat as real-valued audio.
- AC1: Correctly extracts sample rate, bit depth, channel count for PCM16, PCM32, and
  IEEE-float WAV files.
- AC2: A 2-channel WAV explicitly marked "IQ" by the analyst is loaded as a complex64
  array with correct I/Q channel assignment.
- AC3: A non-WAVE file (bad RIFF magic) is rejected with a clear, typed error, not a
  crash or hang.
- AC4: Files up to a configured max size are loaded within bounded memory (see NFR-01);
  oversized files are rejected before full allocation.
- Test IDs: TC-ING-001, TC-ING-002, TC-ING-003, TC-ING-004

### FR-02 Ingest raw .IQ
Support int8, int16, float32, and complex64 interleaved I/Q; read SigMF `.sigmf-meta`
sidecar when present; otherwise require the analyst to specify format, sample rate, and
endianness before loading.
- AC1: Loads int8/int16/float32/complex64 raw IQ files given explicit or SigMF-derived
  format parameters, producing a complex64 in-memory array.
- AC2: When a valid `.sigmf-meta` file is present, `sample_rate`, `datatype`, and
  `frequency` fields are read and pre-filled as defaults (still analyst-editable).
- AC3: If no format is supplied and no SigMF metadata exists, the tool refuses to guess
  silently; it prompts the analyst for format (this is a documented "cannot proceed
  fully blind" case, satisfied via analyst input, not dropped).
- AC4: Malformed/truncated .IQ files (length not a multiple of the sample stride) are
  rejected with a clear error, not read out of bounds.
- Test IDs: TC-ING-010, TC-ING-011, TC-ING-012, TC-ING-013

### FR-03 Handle cross-sensor/location variation
Normalization (unit power scaling), resampling to a common analysis rate, DC-offset
removal, and IQ-imbalance (gain/phase) correction.
- AC1: DC offset in a synthetic test signal is reduced to within a configured residual
  threshold after correction.
- AC2: A synthetic IQ-imbalance impairment (known gain/phase error) is reduced by a
  measurable, reported amount after correction; residual imbalance is reported, not
  hidden.
- AC3: Resampling to a target rate preserves the signal's measured symbol rate within a
  documented tolerance.
- Test IDs: TC-PREP-001, TC-PREP-002, TC-PREP-003

### FR-04 Support HF/VHF/UHF recordings
Accept baseband recordings at any sample rate from the kHz to GHz-derived (post
down-conversion, baseband-sampled) range; no hardcoded band assumptions in ingestion or
estimation code paths.
- AC1: Synthetic test vectors at representative HF (~3 kHz-30 MHz equivalent baseband),
  VHF, and UHF baseband sample rates all pass through ingestion and parameter estimation
  without band-specific special-casing failures.
- Test IDs: TC-ING-020, TC-ING-021, TC-ING-022

### FR-05 Identify sampling frequency
Automatic: read from WAV header or SigMF metadata when present; when absent or flagged
unreliable by the analyst, estimate via spectral occupancy and cyclostationary analysis.
Analyst override: manual entry, with immediate re-run of all downstream stages.
- AC1: When Fs is present in header/metadata, it is used directly and reported with
  confidence = 1.0 (source: metadata).
- AC2: When Fs is absent, the blind estimator returns a value and a confidence score in
  [0,1]; estimation error vs ground truth on synthetic data is measured empirically in
  Phase 4, not asserted here.
- AC3: Analyst can override Fs; the change propagates to demodulation, de-interleaving,
  FEC, and correlation on re-run.
- Test IDs: TC-PARAM-001, TC-PARAM-002, TC-PARAM-003

### FR-06 Identify modulation type
FSK (2/4/8-FSK), PSK (BPSK/QPSK/8PSK), QAM (16/64-QAM), plus AM/FM/noise classes.
Automatic: higher-order cumulants (C20, C40, C42) combined with a CNN/transformer
classifier over raw IQ, ensembled with calibrated confidence. Analyst override: manual
modulation selection.
- AC1: Classifier reports one of the supported classes plus a calibrated confidence
  score for every input.
- AC2: Confusion matrix and accuracy-vs-SNR curve (-5 to 25 dB) are produced by the
  Phase 4 benchmark for every class in scope; no accuracy number is claimed prior to
  that measurement.
- AC3: Analyst override selection is honored by all downstream stages regardless of
  classifier output.
- Test IDs: TC-PARAM-010, TC-PARAM-011, TC-PARAM-012

### FR-07 Identify interleaving type and parameters
Block, Convolutional, Diagonal, Pseudo-random.
Automatic: GF(2) rank-deficiency analysis over candidate periods for block interleaving,
extended structural analysis for convolutional/diagonal; pseudo-random matched only
against a library of standard LFSR/PRNG permutations. Analyst override: manual type,
depth/period, and (for pseudo-random) a user-supplied permutation table or seed.
- AC1: Block interleaver depth/period is recovered on synthetic test vectors within a
  documented search range, with a reported confidence.
- AC2: When the interleaver is pseudo-random with an unknown, non-library generator, the
  automatic stage reports confidence = 0 / "unidentified" rather than a false positive;
  the analyst can supply a permutation table to proceed. This limitation is documented,
  not hidden (see Section 6).
- AC3: Analyst-entered parameters (any type) are consumed by MOD-DEINT on re-run.
- Test IDs: TC-PARAM-020, TC-PARAM-021, TC-PARAM-022, TC-PARAM-023

### FR-08 Identify FEC type and parameters
Convolutional (rate, constraint length, generator polynomials), RS (n, k), Concatenated,
LDPC. Automatic: rank-deficiency / dual-code (parity-check) recovery for convolutional;
GF(2^m) symbol-structure detection for RS; staged inner/outer detection for
concatenated; sparse parity-check matching against a library of standard LDPC codes for
LDPC. Analyst override: manual entry of all parameters, or a user-supplied H matrix /
generator polynomials.
- AC1: For a rate-1/2, K=7 convolutional code (thin end-to-end path), constraint length
  and rate are recovered on synthetic test vectors with reported confidence.
- AC2: For RS codes in a documented (n,k) search range, symbol size and code length are
  recovered with reported confidence.
- AC3: For LDPC, if the H matrix matches a library entry, it is identified with
  confidence; if it does not, the stage reports "no library match" and the analyst can
  supply an H matrix. Full blind reconstruction of an arbitrary unknown H matrix is out
  of scope for automatic mode and is explicitly documented as such (Section 6), not
  claimed.
- Test IDs: TC-PARAM-030, TC-PARAM-031, TC-PARAM-032, TC-PARAM-033

### FR-09 Additional parameters
Center frequency offset (CFO), bandwidth, symbol rate, SNR, roll-off.
- AC1: CFO is estimated and, if within the correctable range, compensated before
  demodulation; residual CFO is reported.
- AC2: Symbol rate is estimated via FFT of the squared envelope or cyclic
  autocorrelation, with confidence.
- AC3: SNR is estimated per-file and reported; used to annotate confidence of all
  upstream/downstream estimates.
- Test IDs: TC-PARAM-040, TC-PARAM-041, TC-PARAM-042, TC-PARAM-043

### FR-10 Demodulate FSK, PSK, QAM to soft and hard bits
Carrier recovery (Costas loop / decision-directed PLL), timing recovery (Gardner),
equalization (CMA/LMS); FSK via frequency discriminator or matched-filter bank.
- AC1: For the thin end-to-end path (QPSK), demodulated hard bits match the known
  transmitted bit sequence at high SNR (>15 dB) on synthetic data with BER within a
  documented tolerance.
- AC2: Soft bit (LLR) output is produced and consumed correctly by MOD-FEC's soft-decision
  Viterbi decoder.
- AC3: Eye diagram and constellation data are exposed for GUI display (FR-14).
- Test IDs: TC-DEMOD-001, TC-DEMOD-002, TC-DEMOD-003

### FR-11 De-interleave: Block, Convolutional, Diagonal, Pseudo-random
- AC1: Block and convolutional de-interleaving exactly invert the corresponding
  synthetic interleaver given correct parameters (bit-exact match).
- AC2: Pseudo-random de-interleaving supports both a library of standard LFSR/PRNG
  seeds and an analyst-supplied explicit permutation table; given the correct
  table/seed, output is bit-exact.
- Test IDs: TC-DEINT-001, TC-DEINT-002, TC-DEINT-003, TC-DEINT-004

### FR-12 FEC decode: Viterbi, RS, Concatenated, LDPC
Viterbi (soft-decision, short constraint length) and Berlekamp-Massey RS decoding, a
concatenated pipeline (RS outer + convolutional inner), and LDPC (min-sum / sum-product
belief propagation) against standard or user-supplied H matrices; performance-critical
kernels in C++.
- AC1: Viterbi decode of the thin-path K=7 rate-1/2 code corrects injected bit errors up
  to the code's documented error-correcting capability on synthetic data.
- AC2: RS decode corrects up to (n-k)/2 symbol errors on synthetic data.
- AC3: Concatenated pipeline (RS outer + convolutional inner) reduces BER relative to
  either code alone, measured on synthetic data across SNR.
- AC4: LDPC BP decoder converges (or reports non-convergence) within a bounded max
  iteration count; never hangs.
- Test IDs: TC-FEC-001, TC-FEC-002, TC-FEC-003, TC-FEC-004, TC-FEC-005

### FR-13 Bit-stream correlation
Autocorrelation, sync-word search, per-bit-position entropy for header/payload
segmentation.
- AC1: A known sync word inserted in synthetic data is located at the correct bit offset.
- AC2: Frame length is recovered from periodic sync-word spacing on synthetic
  multi-frame data.
- AC3: Header/payload boundary is reported with a confidence derived from entropy
  discontinuity.
- Test IDs: TC-CORR-001, TC-CORR-002, TC-CORR-003

### FR-14 GUI views
File input, spectrum, waterfall, constellation, eye diagram, parameter panel with
confidence scores, decoded bitstream view, correlation view.
- AC1: Each named view renders for the thin end-to-end path signal without error.
- AC2: The parameter panel shows, for every estimated parameter, its value, confidence,
  source (metadata / estimated / analyst-override), and an editable override control.
- Test IDs: TC-GUI-001 through TC-GUI-008 (one per view)

### FR-15 Automated + step-by-step analysis
One-click end-to-end pipeline run, and a step-by-step mode where the analyst inspects
and edits parameters between stages.
- AC1: One-click mode runs ingestion through report generation without analyst input on
  the thin end-to-end path.
- AC2: Step-by-step mode pauses after each stage, displays estimated parameters and
  confidence, and accepts analyst edits before the next stage runs.
- Test IDs: TC-GUI-010, TC-GUI-011

### FR-16 Use training data (.IQ and .wav) for spectral-relationship learning
- AC1: A documented synthetic data generator (GNU Radio flowgraphs) produces
  ground-truth-labeled .IQ and .wav pairs across modulations/interleavers/FEC/SNR/
  impairments.
- AC2: RadioML 2018.01A (or equivalent public dataset) is converted to both .IQ and .wav
  representations of the same underlying signal.
- AC3: Cross-format consistency test: the ML classifier trained on this data produces
  the same modulation label (within calibrated confidence tolerance) for the .IQ and
  .wav representation of the same signal.
- Test IDs: TC-DATA-001, TC-DATA-002, TC-DATA-003

### FR-17 Implementation uses GNU Radio, Python, C++
Performance-critical kernels (Viterbi, LDPC BP, correlation) in C++ via pybind11 or
GNU Radio OOT blocks.
- AC1: Build produces working Python bindings for the C++ kernels on the target
  platforms (NFR-04).
- AC2: C++ kernel unit tests run independently of the Python layer.
- Test IDs: TC-CORE-001, TC-CORE-002

### FR-18 Export results, batch processing
JSON + PDF report per file; batch processing of multiple files.
- AC1: JSON report validates against a published JSON Schema and contains every
  estimated/overridden parameter, confidence, and decoded-bitstream summary (not raw
  bitstream dump into logs - see SR-09).
- AC2: PDF report is human-readable and includes the same parameter table plus key
  plots (spectrum, constellation).
- AC3: Batch mode processes a directory of files sequentially/in parallel and produces
  one report per file plus a summary index, continuing past individual file failures
  (isolated, logged, not fatal to the batch).
- Test IDs: TC-REPORT-001, TC-REPORT-002, TC-REPORT-003

---

## 5. Non-Functional Requirements - Refined

### NFR-01 Performance
- AC1: Processing time per stage is measured and published (Phase 4/5) as a function of
  file duration and sample rate, for a documented reference machine spec.
- AC2: Files are processed in bounded-size chunks; peak memory use does not scale
  linearly with unbounded file size for streaming-capable stages (ingestion,
  preprocessing, demodulation).
- Test IDs: TC-PERF-001, TC-PERF-002

### NFR-02 Accuracy
- AC1: Classification accuracy vs. SNR (-5 to 25 dB) is reported per modulation class.
- AC2: BER after FEC decode is reported per code type vs. SNR.
- AC3: Parameter estimation error (Fs, symbol rate, CFO) is reported vs. SNR.
- Test IDs: TC-QA-001, TC-QA-002, TC-QA-003 (full benchmark matrix, Phase 4)

### NFR-03 Usability
- AC1: An analyst unfamiliar with the codebase can complete the one-click workflow
  using only the GUI and the user manual, without writing code.
- Test IDs: TC-GUI-020 (usability walkthrough checklist)

### NFR-04 Portability
- AC1: Runs on a documented Linux distribution/version.
- AC2: Runs on a documented Windows version.
- AC3: Packaging is reproducible from a pinned lockfile (same inputs -> same artifact
  hash, modulo documented non-determinism such as timestamps).
- Test IDs: TC-PKG-001, TC-PKG-002, TC-PKG-003

### NFR-05 Maintainability
- AC1: Public module interfaces are fully type-hinted (Python) / typed (C++).
- AC2: Core modules (MOD-PARAM, MOD-DEMOD, MOD-DEINT, MOD-FEC, MOD-CORR) reach >= 80%
  unit-test line coverage, measured by the CI coverage report.
- Test IDs: TC-QA-010

---

## 6. Security Requirements - Refined (mapped 1:1 to controls in Phase 2)

Each SR below is restated as a testable control; concrete design mechanisms are
produced in Phase 2, enforcement in Phase 3, verification evidence in Phase 4.

| SR | Restated control | Test IDs |
|----|-------------------|----------|
| SR-01 | Every parser validates header fields, enforces a configurable max file size before full allocation, and rejects malformed/truncated input with a typed error, never a crash | TC-SEC-001, TC-SEC-002 |
| SR-02 | No pickle/eval/exec on any external data; ML weights load only from local safetensors/ONNX files whose SHA-256 matches a recorded manifest, else load fails closed | TC-SEC-010, TC-SEC-011 |
| SR-03 | No runtime network sockets are opened by the application under normal operation; no telemetry; no auto-update mechanism exists | TC-SEC-020 |
| SR-04 | Application runs and installs without administrator/root privileges on both target OSes | TC-SEC-030 |
| SR-05 | C++ build enables -fstack-protector-strong and -D_FORTIFY_SOURCE=2 in release config; ASan/UBSan build variant exists and is run in CI | TC-SEC-040, TC-SEC-041 |
| SR-06 | Dependencies are pinned with hashes in a lockfile; pip-audit runs on every build with zero unresolved known-vulnerable pinned versions at release; CycloneDX SBOM is generated per release | TC-SEC-050, TC-SEC-051, TC-SEC-052 |
| SR-07 | bandit and semgrep (Python) and clang-tidy/cppcheck (C++) run in CI with zero high-severity findings at release | TC-SEC-060, TC-SEC-061 |
| SR-08 | Every file-format parser (WAV, IQ, SigMF, H-matrix, permutation-table loaders) has a fuzz harness (atheris for Python entry points, libFuzzer for C++ parsers) run for a documented minimum duration/corpus with no crashes | TC-SEC-070, TC-SEC-071 |
| SR-09 | No signal samples or decoded bitstream content are written outside the analyst-chosen workspace directory or appear in log output; temp files inside the workspace are deleted after use | TC-SEC-080, TC-SEC-081 |
| SR-10 | A local, append-only, hash-chained audit log records user identity, file hash, timestamp, and action for every analysis; chain integrity is verifiable | TC-SEC-090, TC-SEC-091 |
| SR-11 | Workspace and report encryption at rest (AES-256-GCM, user-supplied key) is available as an opt-in mode | TC-SEC-100 |
| SR-12 | Released artifacts are signed and published with checksums; verification instructions are provided | TC-SEC-110 |

---

## 7. Requirements Traceability Matrix (RTM)

| Req ID | PS Clause | Module | Test ID(s) |
|--------|-----------|--------|------------|
| FR-01 | PS-DESC-FMT | MOD-ING | TC-ING-001..004 |
| FR-02 | PS-DESC-FMT | MOD-ING | TC-ING-010..013 |
| FR-03 | PS-DESC-VARY | MOD-ING, MOD-CORE (preprocessing) | TC-PREP-001..003 |
| FR-04 | PS-DESC-BAND | MOD-ING, MOD-PARAM | TC-ING-020..022 |
| FR-05 | PS-BG, PS-GUI-i | MOD-PARAM | TC-PARAM-001..003 |
| FR-06 | PS-BG, PS-GUI-i | MOD-PARAM | TC-PARAM-010..012 |
| FR-07 | PS-BG, PS-GUI-iii | MOD-PARAM | TC-PARAM-020..023 |
| FR-08 | PS-BG, PS-GUI-iv | MOD-PARAM | TC-PARAM-030..033 |
| FR-09 | PS-BG, PS-EXP | MOD-PARAM | TC-PARAM-040..043 |
| FR-10 | PS-DESC-DEMOD, PS-GUI-ii | MOD-DEMOD | TC-DEMOD-001..003 |
| FR-11 | PS-GUI-iii | MOD-DEINT | TC-DEINT-001..004 |
| FR-12 | PS-GUI-iv | MOD-FEC | TC-FEC-001..005 |
| FR-13 | PS-GUI-v, PS-EXP | MOD-CORR | TC-CORR-001..003 |
| FR-14 | PS-EXP | MOD-GUI | TC-GUI-001..008 |
| FR-15 | PS-EXP | MOD-GUI | TC-GUI-010, TC-GUI-011 |
| FR-16 | PS-DESC-SPEC | MOD-DATA | TC-DATA-001..003 |
| FR-17 | PS-DESC-TOOLS | MOD-CORE | TC-CORE-001, TC-CORE-002 |
| FR-18 | PS-EXP | MOD-REPORT | TC-REPORT-001..003 |
| NFR-01 | PS-DESC-VARY | MOD-CORE | TC-PERF-001, TC-PERF-002 |
| NFR-02 | PS-BG | MOD-QA | TC-QA-001..003 |
| NFR-03 | PS-EXP | MOD-GUI | TC-GUI-020 |
| NFR-04 | (delivery constraint) | MOD-PKG | TC-PKG-001..003 |
| NFR-05 | (delivery constraint) | MOD-CORE, MOD-QA | TC-QA-010 |
| SR-01..SR-12 | (security constraint, all clauses) | MOD-SEC (cross-cutting) | TC-SEC-001..110 (see Section 6) |

Note: test case bodies (code) are authored in Phase 4; this RTM fixes the ID contract
now so every later phase references the same identifiers. Any requirement that reaches
Phase 4 without a passing test mapped here must be flagged in the Phase 4 report, not
omitted.

---

## 8. Assumptions and Constraints

### 8.1 What "blind" means per stage, and the analyst-override fallback

| Stage | Blind method | Known limitation | Fallback |
|-------|--------------|-------------------|----------|
| Sample rate (FR-05) | Header/SigMF; else spectral occupancy + cyclostationary estimate | Estimate degrades if signal does not occupy a clean, contiguous band within Nyquist | Manual Fs entry |
| Symbol rate (FR-09) | Cyclic autocorrelation / squared-envelope FFT | Degrades for non-standard pulse shaping or very low SNR | Manual symbol rate entry |
| Modulation (FR-06) | Cumulants + CNN/transformer ensemble, calibrated confidence | Accuracy at low SNR and between closely related orders (e.g., higher QAM orders) is empirically measured in Phase 4, not assumed good | Manual modulation selection |
| Interleaving (FR-07) | GF(2) rank-deficiency for block; structural analysis for convolutional/diagonal; library match for pseudo-random | Pseudo-random interleavers using a generator outside the known library are, in general, NOT identifiable blind - this is a mathematically underdetermined problem from ciphertext-like output alone, not an engineering gap | Analyst supplies permutation table or seed |
| FEC (FR-08) | Dual-code/parity recovery (convolutional), GF(2^m) structure (RS), library match (LDPC) | Blind reconstruction of an arbitrary, undocumented LDPC H matrix from bitstream alone is not attempted in automatic mode; concatenated-code detection is staged and can fail if the inner code is not first correctly identified | Analyst supplies code parameters or H matrix directly |

This table itself is a deliverable of FR-07/FR-08's acceptance criteria and is carried
into the User Manual (Phase 6) verbatim so analysts are never misled about automatic
mode's limits.

### 8.2 Constraints

- C-01: Runtime must be usable air-gapped (SR-03); all model weights, dataset
  artifacts, and dependency wheels must be obtainable and installable without a
  network call at run time (build/train time network access is allowed and is a
  separate, documented offline step).
- C-02: Hackathon delivery timeline requires phased scope: a single thin end-to-end
  path (QPSK, block interleaver, rate-1/2 K=7 convolutional code) is built and fully
  tested first, then breadth is widened to the remaining FR-06/07/08 combinations. This
  is a delivery-order constraint, not a requirement drop; all FR IDs remain in scope for
  final delivery.
- C-03: Public dataset use (RadioML 2018.01A) is subject to its published license; the
  license is checked before any redistribution of raw dataset files in the installer.
  The tool itself does not require internet access to run once trained artifacts exist.
- C-04: No dedicated RF capture hardware is assumed to be available to the team;
  validation primarily uses the synthetic generator (FR-16) and converted public
  datasets. If live captures become available they are treated as additional test
  vectors, not a redesign trigger.
- C-05: GNU Radio, Python, and C++ toolchain versions are pinned in Phase 2's tech
  stack table and are not upgraded during a release cycle without a new lockfile.

---

## 9. STRIDE Threat Model

Assets considered: input signal files (.wav/.IQ/SigMF), ML model weight files, analyst
override inputs, generated reports (JSON/PDF), the audit log, workspace files on disk,
the Python<->C++ boundary (pybind11), and the dependency supply chain.

| Category | Threat scenario | Affected asset | Mitigation (SR) |
|----------|------------------|------------------|-------------------|
| Spoofing | A crafted .IQ/SigMF file carries fabricated metadata (e.g., false center frequency or sensor ID) intended to mislead the analyst | Input file metadata | SigMF/header fields are treated as untrusted hints only, always shown with their source and never auto-trusted for a security-relevant decision (SR-01) |
| Tampering | ML model weight file on disk is modified to bias classification output | Model weights | SHA-256 checksum verified against a recorded manifest before load; mismatch fails closed (SR-02) |
| Tampering | A crafted .wav/.IQ header (bad chunk sizes, oversized declared length) is designed to trigger a buffer overrun in a C++ parser | Ingestion parser | Bounds-checked parsing, size validation before allocation, ASan/UBSan CI, fuzzing (SR-01, SR-05, SR-08) |
| Tampering | A crafted LDPC H-matrix or permutation-table file supplied by an "analyst" contains inconsistent dimensions intended to corrupt memory in the C++ decoder | User-supplied parameter files | Same parser-hardening controls apply to all user-supplied structured files, not just captured signals (SR-01, SR-05, SR-08) |
| Repudiation | An analyst denies having produced a given analysis result later used in a decision | Audit trail | Local, append-only, hash-chained audit log records user, file hash, timestamp, and parameters for every run; chain integrity is independently verifiable (SR-10) |
| Information Disclosure | Raw signal samples or decoded bitstream content leak into application logs or crash dumps | Logs | Logs are restricted to metadata/hashes only; raw signal/bitstream content is never logged (SR-09) |
| Information Disclosure | Workspace files (captured signals, reports) are readable by other local accounts on a shared machine | Workspace on disk | Optional AES-256-GCM encryption at rest with an analyst-supplied key; least-privilege file permissions (SR-11, SR-04) |
| Denial of Service | An oversized or maliciously crafted file causes unbounded memory allocation or a hang during parsing | Ingestion | Configurable max file size enforced before allocation; chunked, bounded-memory processing (SR-01, NFR-01) |
| Denial of Service | A pathological LDPC H matrix or excessive interleaver depth causes runaway/non-terminating decode | MOD-FEC, MOD-DEINT | Bounded iteration counts with explicit non-convergence reporting; GUI-level cancellation (FR-12 AC4) |
| Elevation of Privilege | Installer or update path requires or silently attempts to gain admin/root rights | Install/runtime environment | Runs and installs as a normal user on both target OSes; no auto-update mechanism exists at all (SR-04, SR-03) |
| Elevation of Privilege (supply chain) | A compromised or typo-squatted dependency is pulled in at build time and introduces malicious code | Build/dependency chain | Pinned lockfile with hashes, pip-audit on every build, CycloneDX SBOM published per release, signed release artifacts (SR-06, SR-12) |

---

## 10. Risk Register

| ID | Type | Risk | Likelihood | Impact | Mitigation |
|----|------|------|------------|--------|------------|
| R-01 | Technical | Blind identification of a pseudo-random interleaver with an unlisted generator is mathematically underdetermined, not merely hard | High | Medium (feature still usable via override) | Analyst-supplied permutation table/seed path (FR-07 AC2); document limitation explicitly in User Manual |
| R-02 | Technical | Blind reconstruction of an arbitrary, undocumented LDPC parity-check matrix from the bitstream alone is an open, generally intractable problem at this scope | High | Medium | Library-matching against standard H matrices plus analyst-supplied H (FR-08 AC3); documented as assisted, not fully blind |
| R-03 | Technical | Modulation/parameter estimation accuracy at low SNR or between closely related orders is unknown until measured | Medium | Medium | NFR-02 benchmark across -5 to 25 dB before any accuracy claim is published; confidence scores surfaced to analyst at all times |
| R-04 | Technical | pybind11 Python<->C++ boundary overhead could dominate runtime on large files if not designed for zero-copy buffers from the start | Medium | Medium | Validate with zero-copy numpy buffer passing during the thin end-to-end path build (Phase 3), before widening scope |
| R-05 | Data/Legal | RadioML 2018.01A license terms may restrict redistribution, and it ships as .npy/HDF5, not natively as .wav/.IQ pairs | Medium | Low-Medium | Verify license before packaging; convert format on demand at build/train time, do not redistribute raw dataset inside the installer |
| R-06 | Security | C++ Viterbi/LDPC/correlation kernels process attacker-influenced buffer sizes (from file content); a memory-safety bug here is high impact | Medium | High | ASan/UBSan CI gate, libFuzzer harnesses, bounds-checked containers (SR-05, SR-08) |
| R-07 | Security | ML model weight or config file tampering to bias results | Low | High | SHA-256 checksum verification before load, fail closed (SR-02) |
| R-08 | Security | Malformed .wav/.IQ/H-matrix/permutation-table files crash the parser or exhaust memory | Medium | High | Header validation, max-size enforcement, fuzz testing (SR-01, SR-08) |
| R-09 | Schedule | Full FR-06 x FR-07 x FR-08 combinatorial breadth plus the SNR sweep in NFR-02 is large scope for a hackathon timeline | High | Medium | Deliver the thin end-to-end path fully tested first (C-02), then widen breadth iteratively; demo depth-then-breadth rather than partial-everything |
| R-10 | Security | Audit log hash-chain implementation itself has a bug that breaks tamper-evidence | Low | Medium | Independent chain-verification test that flags any broken link (TC-SEC-091) |

---

## 11. Approval Gate

This concludes Phase 1. Per the SDLC process, I am stopping here for approval before
starting Phase 2 (System Design: architecture diagram, module design, data design, ML
design, GUI wireframes, security design, repo/tech-stack).

Open items for your review before I proceed:
1. Confirm the module ID legend and test ID numbering scheme (Section 2, Section 7) -
   later phases build directly on these IDs.
2. Confirm the thin end-to-end path choice (QPSK + block interleaver + rate-1/2 K=7
   convolutional code) as the Phase 3 build order starting point (C-02).
3. Confirm the two documented "not fully blind" limitations (pseudo-random interleaver
   without a known generator, arbitrary undocumented LDPC H matrix) are acceptable as
   analyst-assisted rather than automatic - these are stated as mathematical/practical
   limits, not gaps we intend to close later.
4. Any additional modulation/FEC/interleaver variants, target OS versions, or
   compliance requirements (e.g., specific to NTRO's environment) not already captured.
