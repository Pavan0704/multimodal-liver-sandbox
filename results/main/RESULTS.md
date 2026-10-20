# Results

These are measured outputs from the deterministic synthetic cohort generator. They are not estimates of liver-disease accuracy.

The experiment used 360 synthetic patient IDs, 8 patches per ID, 18 synthetic profile features, and a 70/15/15 patient-level train/validation/test split. The test set contains 54 generated patients. The model was selected by validation BCE, with no test-driven tuning.

Values are means across three initialization seeds; SD describes variation across those seeds on one fixed synthetic test cohort. It is not a confidence interval.

| Model | ROC-AUC | Avg precision | Balanced accuracy, validation threshold | Balanced accuracy, 0.5 | Brier score | 10-bin ECE |
|---|---:|---:|---:|---:|---:|---:|
| image_only | 0.966 ± 0.014 | 0.970 ± 0.011 | 0.883 ± 0.028 | 0.537 ± 0.064 | 0.228 ± 0.023 | 0.195 ± 0.174 |
| molecular_only | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 |
| multimodal | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.000 |

ROC-AUC, average precision, Brier score, and ECE use continuous probabilities. The primary balanced accuracy uses a threshold selected to maximize balanced accuracy on the validation split for that mode and seed. The 0.5 threshold result is also included to show the effect of threshold choice. No threshold is selected from test labels.

See `metrics_by_seed.csv`, `training_history.csv`, `validation_predictions.csv`, `test_predictions.csv`, `split_manifest.json`, and `config.json` for the recorded details. `attention_example.png` visualizes weights from the image-only MIL model for one synthetic bag; attention is not a validated explanation method.
