# Learning Guide

Use this guide to understand the project well enough to reproduce and explain it. The repository was built with AI assistance. Be clear about that if you discuss it in an application or interview.

## 1. Follow one patient through the pipeline

1. `make_synthetic_cohort` generates a binary latent state, eight toy image patches, and 18 profile values for each synthetic patient.
2. `train_test_split` partitions patient IDs before any model training. Every patch and profile belonging to a patient stays in one split.
3. `GatedAttentionMIL` encodes each patch with the same CNN, scores the patch embeddings, normalizes the scores with softmax, and computes a weighted bag representation.
4. The molecular branch maps the profile vector to a 24-value representation. The multimodal model concatenates it with the 32-value image-bag representation.
5. A binary classifier produces one logit per patient. `BCEWithLogitsLoss` trains the model.
6. Validation BCE selects the checkpoint, and validation balanced accuracy selects a classification threshold. The held-out test labels are used only after these decisions to compute metrics.

## 2. Read the code in this order

- `make_synthetic_cohort`: identify exactly what signal the simulator makes available in each modality.
- `GatedAttentionMIL`: trace tensor shapes. Input is batch × patches × channels × height × width. Output is one image representation and one weight per patch.
- `MultimodalClassifier`: compare the three feature routes.
- `_fit_one`: follow optimization, early stopping, checkpoint selection, and test scoring.
- `run_experiment`: inspect the patient split and saved artifacts.

## 3. Reproduce it yourself

Create a clean Python environment, install `requirements.txt`, and run:

```powershell
python multimodal_liver.py --output results/reproduced
```

Compare your metrics with `results/main/metrics_by_seed.csv`. Exact outputs can vary with Python, PyTorch, CPU kernels, and dependency versions. Explain any differences instead of forcing them to match.

## 4. Know what the metrics say

- ROC-AUC summarizes ranking across thresholds.
- Average precision emphasizes ranking of the positive class.
- Balanced accuracy averages sensitivity and specificity. The primary result uses a threshold chosen on validation data; a fixed 0.5 result is also retained for comparison.
- Brier score is mean squared probability error.
- The 10-bin expected calibration error compares average confidence and accuracy within confidence bins. It depends on binning and is not a guarantee of clinical calibration.
- The reported SD is across three model seeds on one generated test cohort. It is not a confidence interval.

## 5. Be precise about the limitations

This is not a digital pathology model trained on tissue. It does not use MASH, ASH, HCC, patient-derived slides, transcriptomics, spatial omics, radiology, or an external cohort. The simulator explicitly creates the patterns it asks the models to learn. Its scores are a software demonstration, not scientific evidence about liver disease. Attention values show where this particular model placed weight in a toy bag; they do not explain biological mechanisms.

Good interview wording after you have studied and reproduced the code:

> I worked through an AI-assisted synthetic multimodal learning project that compares attention-based pooling over image-patch bags with a small molecular-feature branch. I reproduced the experiment and examined the patient-level split, model-selection procedure, metrics, and limits. The data are synthetic, so I use it to discuss pipeline mechanics rather than clinical performance.

