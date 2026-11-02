# Section 6. Input pipeline (working notes)

English version of `数据说明/第6节_数据流水线.md`.

Code: `scripts/export_model_arrays.py` (exports the arrays), `scripts/data_pipeline.py` (tf.data pipeline and loss with empty labels masked), `scripts/section6_pipeline_checks.py` (the four checks in this section).
Environment: TensorFlow 2.16.2 + tensorflow-metal (Apple M4 Pro, Metal GPU, 24 GB memory), conda environment `tf`. On Kaggle: TensorFlow 2.20.0, Tesla T4. The local machine is used for development and smoke tests; the M4 final training runs on Kaggle Notebooks (see 6.5). Check results are saved in `数据说明/第6节/section6_stats.json`.

## 6.1 Pipeline stages (in order)

| Stage | What it does | Where it is done | Same for training / validation / test? |
|---|---|---|---|
| 1. Fill | Within the same contract, missing minutes between traded bars 2–30 minutes apart are filled as flat bars, used only to complete windows | `build_features.py` | Same |
| 2. Features | Compute 33 features, using only data at or before minute t | `build_features.py` | Same |
| 3. Long-tail handling | Log-transform 4 features, winsorize 8 features at 1%/99%; thresholds fitted on training-set samples only | `transform_tail_features.py` | Same (same thresholds) |
| 4. Standardization | z-score for 22 continuous features, mean and standard deviation fitted on training-set samples only; dummies and tod_sin, tod_cos unchanged | `standardize_features.py` | Same (same means and standard deviations) |
| 5. Sample selection | Minutes with trades, not within 3 days before a roll, all 60 window rows share the same `seg_id`, at least 1 head has a label; empty labels are set to −1 | `export_model_arrays.py` | Same rules |
| 6. Load into memory | The whole feature table `X_all` (352,187 × 33, float32, 46.5 MB) stays in memory; each sample stores only the row index t of the last window row and the 6 labels | `data_pipeline.py` | Same |
| 7. Shuffle | `shuffle`, buffer = whole training set (231,516 row indices), reshuffled every epoch, seed 42 | `data_pipeline.py` | **Training set only** |
| 8. Batch | `batch(256)`, last incomplete batch kept | `data_pipeline.py` | Same |
| 9. Slice windows | `map`: gather `X_all[t − 59 : t + 1]` by row index in batch, giving [B, 60, 33] float32 | `data_pipeline.py` | Same |
| 10. Prefetch | `prefetch(AUTOTUNE)`, prepares the next batch while the model computes the current one | `data_pipeline.py` | Same |

All three sets call the same function `make_dataset(arr, split)`; the only branch is `if split == "train": shuffle`. All fitted parameters (winsorizing thresholds, means, standard deviations, OFI typical values by time of day) are computed on the training set only and applied unchanged to the validation and test sets.

Each batch is `(x, y, t)`: x is [B, 60, 33] float32; y is [B, 6] int32, with values 0 = take-profit, 1 = stop-loss, 2 = timeout, −1 = empty label; t is the row index of the last window row, used only for checks and plots, not fed to the model.

**Sample counts:** training set 231,516 (905 batches), validation set 48,936 (192), test set 52,170 (204).

## 6.2 Reasons for caching, shuffling and batch size

- **Caching: do not cache sliced windows; keep only the raw feature table in memory.** Pre-slicing all windows would take 1.83 GB for the training set and 2.63 GB for all three sets, because the 60-row windows of neighboring samples overlap by 59 rows, so each row is copied 60 times. The feature table itself is only 46.5 MB, and slicing windows by row index on the fly is cheap: running only the pipeline, without training a model, one full pass over the training set takes 0.42 seconds (0.96 seconds on Kaggle), far less than the model compute time, so it is not a bottleneck. `.cache()` is therefore unnecessary; with the feature table in memory, it is effectively cached already.
- **Shuffling: buffer equals the whole training set.** Neighboring samples are highly correlated (Section 4, Figure 3(b): for samples 1 minute apart, the long-30-minute label is the same 85% of the time). With a small buffer, say 10,000, a batch would come only from a few neighboring days and the gradient would be biased toward one market period. Only row indices are shuffled (about 2 MB), so a full shuffle costs almost no time. Reshuffling every epoch keeps the model from seeing data in the same order each time. The validation and test sets are not shuffled; they stay in time order, which makes it easy to analyze results by month and by time of day.
- **Batch size 256:**
  - 905 steps per epoch on the training set, enough steps and enough checkpoints for learning-rate scheduling and early stopping;
  - one batch covers 256 random moments, roughly 200+ different trading hours, so class shares are stable (stop-loss about 43%), and after masking empty labels each head still has about 250 valid labels;
  - compute time: on Kaggle T4, with batch 256, LSTM 64 takes about 6 seconds per epoch (about 15 seconds locally), which is fast enough; larger batches would reduce the number of updates per epoch (I did not time different batch sizes).
  - Batch size can be tuned again as a hyperparameter in M4 (e.g. 128 / 512).

## 6.3 Plotting one batch from the end of the pipeline

Figure 8 (`数据说明/第6节/中文/图8_流水线批次示例.png`, English version `English/Fig8_pipeline_batch_example.png`): the first batch taken from the end of the training pipeline (after shuffle, batch, window slicing and prefetch), showing the first 4 samples. Features are first de-standardized (x × standard deviation + mean), and log-transformed features are exponentiated back to their original units. Labels are read directly from the batch's y and written in each subplot's title.

Check results:
- the windows in the batch are element-wise identical to the 60 rows of `X_all` ending at t; checked for all three sets;
- converting the batch's row indices back to timestamps and looking up the labels in `labels/labels_default.parquet`, all 6 labels of all 256 samples match;
- all these samples belong to the training set and none is a filled minute.

Note: winsorizing is not reversible, so winsorized values stay at the threshold after de-standardization. For example, spread_atr in column 3 of Figure 8 has a flat stretch at 0.063, the training-set 1st percentile.

## 6.4 Smoke test

Model: input [60, 33] → Flatten → Dense(18) → Reshape(6, 3), 35,658 parameters in total. The loss is cross-entropy with empty labels masked: compute the cross-entropy of each head, multiply by the mask (0 for heads with label −1), then divide by the number of heads with labels. Adam optimizer, learning rate 1e-3, 300 training steps.

| Check | Result |
|---|---|
| Batch shape | x [256, 60, 33], y [256, 6] (same for training, validation and test; the last batch may be incomplete) |
| Data type | x float32, y int32 |
| Label values | {−1, 0, 1, 2} (−1 = empty label, masked in the loss) |
| Training loss | mean of first 10 steps 1.355 → running mean at step 300 1.096; single-step loss in the last few steps about 1.03–1.08 |
| Validation loss (50 batches) | before training 1.504 → after training 1.067 |
| Reference | uniform guess ln 3 = 1.099; guessing training-set class shares = 1.066 |

Figure 9 (`数据说明/第6节/中文/图9_冒烟测试_loss.png`, English version `English/Fig9_smoke_test_loss.png`) shows the loss decreasing steadily. After 300 steps the validation loss falls to the "guess class shares" level: in so few steps a linear model mainly learns the class shares and has not yet learned any predictive signal. The smoke test only checks that data flow correctly into the model and that the loss decreases normally; it does not evaluate predictive ability.

## 6.5 Compute and memory

**Time for one full pass over the training set** (one training epoch + one validation pass, batch 256, mixed precision). Measured on the platform used for actual training, Kaggle Notebooks (Tesla T4, single GPU; Kaggle now offers only T4 × 2, and the script uses one of them). Local Metal GPU results are in the last column for comparison:

| Model | Parameters | Kaggle T4, L = 60 | Kaggle T4, L = 120 | Local Metal, L = 60 |
|---|---|---|---|---|
| Pipeline only (no training) | — | 0.96 s | 1.45 s | 0.42 s |
| Flatten + Dense (smoke test) | 35,658 | 3.5 s | 4.4 s | 5.6 s |
| Conv1D 64 × 2 layers | 32,338 | 5.6 s | 6.5 s | 10.9 s |
| LSTM 64 | 26,258 | 6.4 s | 8.9 s | 17.2 s |
| CNN-BiLSTM 64 (Conv1D 64 × 2 layers → bidirectional LSTM 64) | 99,538 | 11.3 s | 18.9 s | 192 s |
| CNN-BiLSTM 128 (Conv1D 64 × 2 layers → bidirectional LSTM 128) | 233,426 | 15.7 s | 28.5 s | 194 s |
| M2 full architecture, BiLSTM 64 | 107,154 | — | 22.6 s | — |
| M2 full architecture, BiLSTM 128 | 246,930 | — | 30.7 s | — |

- "M2 full architecture" is the model in the M2 proposal: batch normalization and dropout after each convolutional layer, a shared dense layer after the BiLSTM (width 64; M2 says 32 or 64, the wider one is used), then 6 output heads. Measured only in the slowest setting (L = 120). In the same run, CNN-BiLSTM 128 took 26.5 seconds per epoch and 30.7 seconds with these layers added, about 16% slower, so the upper bound below is based on the M2 full architecture. The same model differs by about 2 seconds across Kaggle sessions (CNN-BiLSTM 128: 26.5 seconds this time, 28.5 seconds last time).
- At L = 120 the smoke-test model has 71,298 parameters, because the input after Flatten is longer; the parameter counts of the other models do not depend on L.
- Mixed precision gives no speed-up on these small models: compared with float32, every entry differs by at most 1.5 seconds. It stays on because the course requires it and it reduces GPU memory use.
- The local bidirectional LSTM is more than 10 times slower than on Kaggle, possibly because Metal has no dedicated optimization for bidirectional LSTM, while Kaggle's T4 has cuDNN. So final training runs on Kaggle, and the local machine is used only for development and smoke tests.
- Scripts: `scripts/kaggle_timing.py` (Kaggle), `scripts/time_cnn_bilstm.py` (local CNN-BiLSTM); results saved in `数据说明/第6节/kaggle_timing_T4.csv`.

**Total compute estimate:**

Total time = time per epoch × planned epochs × number of runs

- Planned epochs: at most 30, with early stopping, usually fewer in practice;
- Number of runs: 3 models × L ∈ {30, 60, 120} × 3 random seeds = 27 runs, plus about 13 hyperparameter trials, about 40 in total; plus the three additional training items required by Section 7, done only with the best selected model: label settings 4 × 3 random seeds = 12 runs (of the 5 settings that passed the checks, the default one is already included in the 40 runs above; see "M4 candidate settings" in Section 2), the ablation with / without order-flow features 3 runs, batch size 2 runs, 17 in total. About 57 runs overall.
- Time per epoch is taken for the slowest case: M2 full architecture, BiLSTM 128, L = 120, 30.7 seconds. L = 30 and L = 60 are both faster.

30.7 seconds × 30 × 57 ≈ 52,500 seconds ≈ **14.6 hours**, which is the upper bound. If LSTM 64 with L = 60 is the main model, 6.4 seconds × 30 × 57 ≈ 3.0 hours.

**Comparison with weekly GPU time:** Kaggle's GPU quota is 30 hours per week.
- The upper bound of 14.6 hours is less than half the weekly quota, so everything can run within one week with room left for reruns;
- Lower bound: the smoke test takes 3.5 seconds per epoch, the fastest any training pass can be; × 30 × 57 gives about 1.7 hours;
- So the plan is feasible and does not need to be scaled down.

During training, `BackupAndRestore` allows resuming after an interruption, `ModelCheckpoint` saves the best model to Kaggle's output dataset, and mixed precision is enabled.

**Does the data fit in memory:** Yes.
- Current approach: feature table 46.5 MB plus row indices and labels about 3 MB, less than 100 MB in total.
- Even with all windows pre-sliced: 1.83 GB for the training set, 2.63 GB for all three sets, still far below the local 24 GB memory and the memory of common GPUs.
- Windows are not pre-sliced because it is unnecessary, not because they would not fit in memory.
