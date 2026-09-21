# Reproduction of RC4/TRIVIUM/ESPRESSO Keystream Classification Methodology

## 1. Scope
Independent reproduction of the paper's methodology: ANN, DNN, 1D-CNN, and
feature-based FFNN classifiers trained to distinguish RC4, TRIVIUM, and
ESPRESSO keystreams at 5 lengths (8, 24, 32, 64, 1024 bytes), using the
**updated architecture spec** supplied during this session (L-scaled
ANN/DNN dense layers, one-hot+global-max-pool CNN for length-invariant
capacity, fixed-dimensionality FFNN features).

## 2. Cipher implementation verification
- **RC4**: verified against the standard test vector for key=`"Key"` — first
  4 keystream bytes `EB9F7781` match exactly.
- **TRIVIUM**: verified against the official all-zero key/IV test vector —
  full 16-byte keystream `FBE0BF265859051B517A2E4E239FC97F` matches exactly
  (this supersedes an earlier, erroneous claim in this session that
  fabricated 8 of those 16 bytes without computing them; the value above was
  independently recomputed and printed in full).
- **ESPRESSO**: implemented from the published NLFSR structure but **not**
  verified against an official test vector — no authoritative stage-by-stage
  feedback table or reference keystream was available. This is a
  reconstruction, flagged as a limitation, not a bit-exact reference
  implementation.

## 3. Main results: 3-class classification accuracy (RC4 vs TRIVIUM vs ESPRESSO)

Mean ± std over 3 random seeds; binomial test compares observed accuracy to
chance (1/3) given the test-set size; "Collapse fraction" = the fraction of
all test predictions landing on the single most-predicted class (1.0 = the
model always predicts one fixed class, i.e. total class collapse).

See `results_table.csv` for the full numeric table. Every one of the 20
model x length cells is statistically indistinguishable from chance
(p >= 0.2 in all cases, most p = 1.0), matching the paper's own headline
finding of ~33% ("chance-level") accuracy across all conditions.

## 4. Critical finding: near-chance accuracy is driven by class collapse, not by graceful uncertainty
Inspecting confusion matrices (not reported in the original paper) shows
that in the large majority of cells the network does not distribute errors
evenly across the three classes — it collapses to predicting **one fixed
class for every input**, e.g.:
```
ANN, L=8:  [[0, 0, 26215], [0, 0, 26215], [0, 0, 26214]]   <- always predicts class 2
FFNN, L=64: [[26215, 0, 0], [26215, 0, 0], [26214, 0, 0]]  <- always predicts class 0
```
This is the correct outcome to disclose to reviewers: accuracy sitting at
exactly 1/3 is consistent with either (a) genuinely no learnable signal, or
(b) an optimization failure mode where the network settles into a
degenerate all-one-class solution because the loss landscape offers no
useful gradient. The two are indistinguishable from accuracy alone, which is
why positive controls (Section 5) are essential.

The 1D-CNN shows partial (not consistent) escape from full collapse: at
L=8 and L=1024 its collapse fraction is 0.92 and 0.80 respectively (some
spread across classes), but at L=24, 32, and 64 it collapses fully
(fraction = 1.0), identical to every other architecture at those lengths.
So collapse is the dominant behavior overall (16 of 20 cells at exactly
1.0), and the 1D-CNN only deviates from it in 2 of its 5 length settings.

## 5. Positive controls (added per reviewer-anticipated concern: "maybe your code is broken")

**5a. RC4 known-bias statistical test (data integrity check).**
The Mantin-Shamir bias predicts RC4's 2nd keystream byte equals 0 with
probability ~2/256 instead of 1/256. On our own generated RC4 data
(n=131,072): observed rate = 0.008041 vs expected biased rate 2/256 =
0.007812 (two-sided binomial p=0.35, i.e. statistically consistent with the
known bias) and vs the unbiased null 1/256 = 0.003906 (p=6e-98, i.e.
decisively rules out unbiased generation). This confirms our RC4 generator
and data pipeline reproduce a real, previously documented cryptographic
signature rather than producing degenerate/broken data.

**5b. Trivial-separability sanity check (pipeline integrity check).**
Same ANN architecture and training harness, but on an artificial task with
an obvious mean-value signal (uniform bytes vs bytes restricted to
128-255): test accuracy = 1.0000. This confirms the training pipeline,
labels, and optimizer are not fundamentally broken — it can learn cleanly
when a strong signal is present.

**5c. Low-complexity generator positive control (RC4 vs 16-bit LFSR).**
Per the reviewer suggestion, we classified RC4 keystream against a
deliberately weak 16-bit Fibonacci LFSR keystream (period 65535, far lower
statistical complexity than any of the three ciphers under study) through
the same ANN pipeline: test accuracy = 0.9998. This is the strongest
evidence that the near-chance results in Section 3 reflect genuine
statistical indistinguishability of RC4/TRIVIUM/ESPRESSO under this
architecture family, not a broken training pipeline -- when a real
complexity gap exists, the same code detects it almost perfectly.

**5d. Reduced-warmup TRIVIUM control (partially informative, reported as-is).**
We also tried classifying full-round TRIVIUM (1152 warm-up clocks) vs a
TRIVIUM variant clocked only once before keystream output. This control
came back at chance (0.499, binary task) despite the deliberately crippled
warm-up. Combined with 5c, this suggests 1-clock TRIVIUM is not actually a
"low-complexity" generator in the same sense as an LFSR (a single clock
still combines 3 fully-loaded 80/80/111-bit registers through the nonlinear
update once, which does not create the low-order linear structure a dense
byte-level classifier can exploit). We report this as a negative
sub-result rather than omitting it: not every "weakened" construction is
weak in a way this architecture can detect, which is itself informative for
choosing a positive-control generator.

## 6. Compute-driven deviations from full-scale (2^17 samples/cipher)
- ANN at L=1024: used 12000 samples/cipher instead of full 131072 (reduced from full 2^17 due to CPU compute-time cost of large dense/one-hot layers at this length)
- 1D-CNN at L=1024: used 12000 samples/cipher instead of full 131072 (reduced from full 2^17 due to CPU compute-time cost of large dense/one-hot layers at this length)

All other 18 of 20 model x length cells used the full 2^17 = 131,072
samples per cipher as specified.

## 7. Files
- `results_table.csv` — full numeric results (mean/std over 3 seeds, binomial
  p-values, 95% CIs, collapse fractions, sample sizes) for all 20 cells.
- `data/results.json`, `data/extra_seeds.json` — raw per-seed results and
  confusion matrices.
