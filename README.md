# CipherClassification

Reproduction of a keystream cipher-classification experiment: training ANN, DNN,
1D-CNN, and feature-based FFNN classifiers to distinguish RC4, TRIVIUM, and an
ESPRESSO-like stream cipher's keystream output, at five output lengths
(8, 24, 32, 64, 1024 bytes), across 3 random seeds.

## Layout

- `src/generate_keystreams.py` — vectorized NumPy implementations of the RC4,
  TRIVIUM, and ESPRESSO-like keystream generators used to build the labeled
  dataset. (The full ~400MB generated keystream checkpoint is not committed
  here — regenerate it by calling the batch generator functions in this file.)
- `src/train.py` — model architectures (ANN, DNN, 1D-CNN, feature-based FFNN)
  and the training/evaluation loop across all (architecture x keystream length)
  combinations.
- `src/compile_results_table.py` — aggregates per-run results into the final
  results table (mean/std accuracy, confusion matrices, collapse fraction).
- `src/compile_summary_multiseed.py` — aggregates results across the 3 random
  seeds into `results/summary_multiseed.json`.
- `src/binomial_test.py` — binomial significance test of each cell's accuracy
  against the 1/3 chance baseline.
- `results/` — raw and aggregated results: `results.json` (seed 0),
  `extra_seeds.json` (seeds 1-2), `summary_multiseed.json` (aggregated mean/std
  across all 3 seeds), `binom_results.json` (binomial tests vs. chance),
  `results_table.csv` (the final results table).
- `report.md` — full write-up of the reproduction: methodology, results,
  and the class-collapse finding (most architecture x length cells collapse
  to predicting a single class rather than distributing predictions, which
  is why raw accuracy alone is a misleading signal here).

## Requirements

See `requirements.txt`. Training was run with TensorFlow/Keras on CPU.

## Notes

This reproduces the methodology described in the accompanying paper draft,
with the model architecture spec (ANN/DNN layer widths, CNN one-hot + global
max pooling) updated per the author's later revision. See `report.md` for
the full methodology and findings, including caveats about class collapse.
