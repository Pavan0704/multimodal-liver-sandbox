"""Small, reproducible synthetic benchmark for multimodal pathology workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_SEEDS = (7, 17, 27)
MODES = ("image_only", "molecular_only", "multimodal")
FEATURE_COUNT = 18


def _make_tile(rng: np.random.Generator, state: int, informative: bool, size: int) -> np.ndarray:
    """Draw a toy stain-coloured patch; this is not a histology image."""
    yy, xx = np.mgrid[0:size, 0:size]
    stain = rng.normal(0.0, 5.0, size=(size, size, 3))
    base = np.array([224.0, 165.0, 190.0], dtype=np.float32)
    base[0] += rng.normal(0, 5)
    base[1] += rng.normal(0, 6)
    base[2] += rng.normal(0, 6)
    tile = np.clip(base[None, None, :] + stain, 0, 255).astype(np.float32)

    if informative:
        # The positive latent state has more and slightly larger nuclei.
        n_cells = int(rng.poisson(8 if state == 0 else 22))
        radius_mean = 1.25 if state == 0 else 1.65
    else:
        # Most patches carry only shared background structure.
        n_cells = int(rng.poisson(13))
        radius_mean = 1.4

    for _ in range(n_cells):
        cx = int(rng.integers(1, size - 1))
        cy = int(rng.integers(1, size - 1))
        radius = max(1.0, float(rng.normal(radius_mean, 0.28)))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        nuclear = np.array([65.0, 48.0, 112.0]) + rng.normal(0, 8, size=3)
        tile[mask] = np.clip(nuclear, 0, 255)

    # A mild, random illumination field avoids identical-looking patches.
    field = (2.5 * np.sin(xx / rng.uniform(4.5, 7.0) + rng.uniform(0, 6)))[..., None]
    return np.clip((tile + field) / 255.0, 0.0, 1.0).astype(np.float32)


def make_synthetic_cohort(
    n_patients: int = 360,
    n_patches: int = 8,
    image_size: int = 28,
    seed: int = 20261020,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Create paired bags and profiles with a shared synthetic latent label."""
    rng = np.random.default_rng(seed)
    labels = np.tile(np.array([0, 1], dtype=np.int64), (n_patients + 1) // 2)[:n_patients]
    rng.shuffle(labels)
    images = np.empty((n_patients, n_patches, 3, image_size, image_size), dtype=np.float32)
    molecular = rng.normal(0.0, 0.8, size=(n_patients, FEATURE_COUNT)).astype(np.float32)
    batch = rng.integers(0, 4, size=n_patients)

    # Six pathway-like measurements encode a noisy, paired signal.
    directions = np.array([1, -1, 1, 1, -1, -1], dtype=np.float32)
    molecular[:, :6] += ((labels[:, None] * 2) - 1) * directions[None, :] * 0.9
    # Four unrelated features carry a nuisance batch offset.
    batch_offsets = np.array([-0.7, -0.2, 0.3, 0.8], dtype=np.float32)
    molecular[:, 6:10] += batch_offsets[batch, None]

    patient_ids = [f"SYN{i:04d}" for i in range(n_patients)]
    for i, label in enumerate(labels):
        informative_indices = set(rng.choice(n_patches, size=max(1, n_patches // 4), replace=False))
        for j in range(n_patches):
            tile = _make_tile(rng, int(label), j in informative_indices, image_size)
            images[i, j] = tile.transpose(2, 0, 1)
    return images, molecular, labels, patient_ids


class GatedAttentionMIL(nn.Module):
    """Encode each patch and pool a bag with learned gated attention weights."""

    def __init__(self, width: int = 32, attention_width: int = 24):
        super().__init__()
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.attention_v = nn.Linear(width, attention_width)
        self.attention_u = nn.Linear(width, attention_width)
        self.attention_w = nn.Linear(attention_width, 1, bias=False)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, patch_count = bag.shape[:2]
        patch_features = self.patch_encoder(bag.flatten(0, 1)).view(batch_size, patch_count, -1)
        gated = torch.tanh(self.attention_v(patch_features)) * torch.sigmoid(self.attention_u(patch_features))
        scores = self.attention_w(gated).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(patch_features * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class MultimodalClassifier(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        if mode != "molecular_only":
            self.image_encoder = GatedAttentionMIL()
        if mode != "image_only":
            self.molecular_encoder = nn.Sequential(
                nn.Linear(FEATURE_COUNT, 32), nn.ReLU(), nn.Dropout(0.10), nn.Linear(32, 24), nn.ReLU()
            )
        width = 32 if mode == "image_only" else 24 if mode == "molecular_only" else 56
        self.classifier = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1))

    def forward(
        self, images: torch.Tensor, molecular: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention = None
        if self.mode == "image_only":
            features, attention = self.image_encoder(images)
        elif self.mode == "molecular_only":
            features = self.molecular_encoder(molecular)
        else:
            image_features, attention = self.image_encoder(images)
            molecular_features = self.molecular_encoder(molecular)
            features = torch.cat([image_features, molecular_features], dim=1)
        return self.classifier(features).squeeze(1), attention


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _tensor_dataset(images: np.ndarray, molecular: np.ndarray, labels: np.ndarray, ids: np.ndarray):
    selected_images = np.ascontiguousarray(images[ids])
    selected_molecular = np.ascontiguousarray(molecular[ids])
    selected_labels = np.ascontiguousarray(labels[ids].astype(np.float32))
    return TensorDataset(
        torch.frombuffer(memoryview(selected_images), dtype=torch.float32).reshape(selected_images.shape).clone(),
        torch.frombuffer(memoryview(selected_molecular), dtype=torch.float32).reshape(selected_molecular.shape).clone(),
        torch.frombuffer(memoryview(selected_labels), dtype=torch.float32).clone(),
    )


def _predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    logits_all, labels_all, attention_all = [], [], []
    with torch.no_grad():
        for images, molecular, labels in loader:
            logits, attention = model(images, molecular)
            logits_all.extend(logits.tolist())
            labels_all.extend(labels.tolist())
            if attention is not None:
                attention_all.extend(attention.tolist())
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits_all, dtype=np.float64)))
    return probs, np.asarray(labels_all, dtype=int), np.asarray(attention_all, dtype=np.float32) if attention_all else None


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predicted = (probabilities >= 0.5).astype(int)
    correct = predicted == labels
    error = 0.0
    for i in range(bins):
        mask = (confidence >= edges[i]) & (confidence < edges[i + 1] if i < bins - 1 else confidence <= edges[i + 1])
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a balanced-accuracy threshold using validation predictions only."""
    candidates = np.unique(np.concatenate([np.linspace(0.0, 1.0, 201), probabilities]))
    scores = [balanced_accuracy_score(labels, (probabilities >= value).astype(int)) for value in candidates]
    best = max(scores)
    tied = candidates[np.asarray(scores) == best]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def _score(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    predicted_at_half = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(labels, predicted_at_half)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece_10_bins": expected_calibration_error(labels, probabilities),
    }


def _fit_one(
    mode: str,
    seed: int,
    images: np.ndarray,
    molecular: np.ndarray,
    labels: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    patient_ids: list[str],
    out_dir: Path,
    max_epochs: int,
    batch_size: int,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[nn.Module, np.ndarray, np.ndarray]]:
    set_seed(seed)
    model = MultimodalClassifier(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(
        _tensor_dataset(images, molecular, labels, train_ids), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        _tensor_dataset(images, molecular, labels, val_ids), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        _tensor_dataset(images, molecular, labels, test_ids), batch_size=batch_size, shuffle=False
    )
    best_val = float("inf")
    best_state = None
    patience, stale_epochs = 4, 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch_images, batch_molecular, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_images, batch_molecular)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        val_probs, val_labels, _ = _predict(model, val_loader)
        val_loss = float(
            -np.mean(
                val_labels * np.log(np.clip(val_probs, 1e-7, 1.0))
                + (1 - val_labels) * np.log(np.clip(1.0 - val_probs, 1e-7, 1.0))
            )
        )
        history_rows.append(
            {"mode": mode, "seed": seed, "epoch": epoch, "train_bce": float(np.mean(losses)), "validation_bce": val_loss}
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint selected on validation loss")
    model.load_state_dict(best_state)
    val_probs, val_labels, _ = _predict(model, val_loader)
    threshold = select_threshold(val_labels, val_probs)
    validation_rows = pd.DataFrame(
        {
            "patient_id": [patient_ids[i] for i in val_ids],
            "target": val_labels,
            "probability_state_1": val_probs,
            "selected_threshold": threshold,
            "mode": mode,
            "seed": seed,
        }
    )
    test_probs, test_labels, test_attention = _predict(model, test_loader)
    metrics = _score(test_labels, test_probs, threshold)
    prediction_rows = pd.DataFrame(
        {
            "patient_id": [patient_ids[i] for i in test_ids],
            "target": test_labels,
            "probability_state_1": test_probs,
            "predicted_state": (test_probs >= threshold).astype(int),
            "selected_threshold": threshold,
            "mode": mode,
            "seed": seed,
        }
    )
    if test_attention is not None:
        prediction_rows["attention_entropy"] = -np.sum(
            test_attention * np.log(np.clip(test_attention, 1e-9, 1.0)), axis=1
        )
    row: dict[str, object] = {
        "mode": mode,
        "seed": seed,
        "best_validation_bce": best_val,
        "epochs_run": len(history_rows),
        "selected_threshold": threshold,
        **metrics,
    }
    return row, pd.DataFrame(history_rows), validation_rows, prediction_rows, (model, test_attention, test_ids)


def _save_example(images: np.ndarray, patient_ids: list[str], model: nn.Module, patient_index: int, path: Path) -> None:
    one_array = np.ascontiguousarray(images[patient_index : patient_index + 1])
    one_images = torch.frombuffer(memoryview(one_array), dtype=torch.float32).reshape(one_array.shape).clone()
    with torch.no_grad():
        _, attention = model(one_images, torch.zeros((1, FEATURE_COUNT), dtype=torch.float32))
    weights = np.asarray(attention.squeeze(0).tolist(), dtype=np.float32)
    patch_count = images.shape[1]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[patient_index, i].transpose(1, 2, 0))
        ax.set_title(f"Patch {i + 1} | attention {weights[i]:.2f}", pad=7)
        ax.axis("off")
    fig.suptitle(f"Synthetic image-only MIL bag {patient_ids[patient_index]} | {patch_count} patches", fontsize=13)
    fig.text(0.5, 0.015, "Generated toy pixels only. Attention weights are not clinical explanations.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92), h_pad=2.4, w_pad=1.2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_metric_plot(summary: pd.DataFrame, path: Path) -> None:
    names = ["image_only", "molecular_only", "multimodal"]
    labels = ["Image bag\n(CNN + attention MIL)", "Molecular profile\n(MLP)", "Multimodal\n(late fusion)"]
    metrics = [("roc_auc", "ROC-AUC (higher is better)"), ("balanced_accuracy", "Balanced accuracy\n(validation-selected threshold)"), ("brier_score", "Brier score (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    for ax, (metric, title) in zip(axes, metrics):
        rows = summary.set_index("mode").loc[names]
        ax.bar(range(3), rows[f"{metric}_mean"], yerr=rows[f"{metric}_sd"], capsize=4, color=["#6887a8", "#88a982", "#b884a4"])
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Synthetic held-out patients | mean and SD across three model seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _split_digest(patient_ids: list[str], ids: np.ndarray) -> str:
    payload = "\n".join(sorted(patient_ids[i] for i in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_experiment(
    out_dir: Path,
    n_patients: int = 360,
    max_epochs: int = 14,
    batch_size: int = 32,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> None:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    images, molecular, labels, patient_ids = make_synthetic_cohort(n_patients=n_patients)
    all_ids = np.arange(n_patients)
    train_ids, heldout_ids = train_test_split(all_ids, test_size=0.30, random_state=2026, stratify=labels)
    val_ids, test_ids = train_test_split(heldout_ids, test_size=0.50, random_state=2027, stratify=labels[heldout_ids])
    sets = [set(train_ids.tolist()), set(val_ids.tolist()), set(test_ids.tolist())]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Patient split overlap detected")

    split_record = {}
    for name, ids in zip(("train", "validation", "test"), (train_ids, val_ids, test_ids)):
        split_record[name] = {
            "patients": int(len(ids)),
            "state_0": int((labels[ids] == 0).sum()),
            "state_1": int((labels[ids] == 1).sum()),
            "patient_id_sha256": _split_digest(patient_ids, ids),
        }
    split_record["overlap_check"] = "passed: patient IDs are disjoint across all three splits"
    (out_dir / "split_manifest.json").write_text(json.dumps(split_record, indent=2), encoding="utf-8")

    config = {
        "project": "multimodal-liver-sandbox",
        "data": "procedurally generated synthetic toy cohort; no real tissue, patients, genes, or diagnoses",
        "n_patients": n_patients,
        "patches_per_patient": int(images.shape[1]),
        "patch_shape_chw": list(images.shape[2:]),
        "molecular_features": FEATURE_COUNT,
        "label": "synthetic binary latent state",
        "patient_level_split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "model_seeds": list(seeds),
        "max_epochs": max_epochs,
        "early_stopping_patience": 4,
        "batch_size": batch_size,
        "optimizer": "AdamW(lr=0.002, weight_decay=0.0001)",
        "model_selection": "lowest validation BCE; balanced-accuracy threshold selected on validation; test split used only for final scoring",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metric_rows, history_frames, validation_frames, prediction_frames, model_outputs = [], [], [], [], {}
    for seed in seeds:
        for mode in MODES:
            row, history, validation_predictions, predictions, output = _fit_one(
                mode, seed, images, molecular, labels, train_ids, val_ids, test_ids,
                patient_ids, out_dir, max_epochs, batch_size,
            )
            metric_rows.append(row)
            history_frames.append(history)
            validation_frames.append(validation_predictions)
            prediction_frames.append(predictions)
            model_outputs[(seed, mode)] = output
            print(f"seed={seed:>2} mode={mode:<16} val_bce={row['best_validation_bce']:.3f} test_auc={row['roc_auc']:.3f}", flush=True)

    by_seed = pd.DataFrame(metric_rows)
    by_seed.to_csv(out_dir / "metrics_by_seed.csv", index=False)
    summary_metrics = ("roc_auc", "average_precision", "balanced_accuracy", "balanced_accuracy_at_0_5", "brier_score", "ece_10_bins")
    summary = by_seed.groupby("mode", sort=False).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in summary_metrics},
        **{f"{metric}_sd": (metric, "std") for metric in summary_metrics},
        selected_threshold_mean=("selected_threshold", "mean"),
        selected_threshold_sd=("selected_threshold", "std"),
    ).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(out_dir / "training_history.csv", index=False)
    pd.concat(validation_frames, ignore_index=True).to_csv(out_dir / "validation_predictions.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(out_dir / "test_predictions.csv", index=False)

    first_image_model, _, _ = model_outputs[(seeds[0], "image_only")]
    positive_test = next((int(i) for i in test_ids if labels[i] == 1), int(test_ids[0]))
    _save_example(images, patient_ids, first_image_model, positive_test, out_dir / "attention_example.png")
    _save_metric_plot(summary, out_dir / "metrics.png")
    elapsed = time.perf_counter() - started
    (out_dir / "runtime.json").write_text(json.dumps({"elapsed_seconds": round(elapsed, 2)}, indent=2), encoding="utf-8")
    report = [
        "# Results",
        "",
        "These are measured outputs from the deterministic synthetic cohort generator. They are not estimates of liver-disease accuracy.",
        "",
        f"The experiment used {n_patients} synthetic patient IDs, {images.shape[1]} patches per ID, {FEATURE_COUNT} synthetic profile features, and a 70/15/15 patient-level train/validation/test split. The test set contains {len(test_ids)} generated patients. The model was selected by validation BCE, with no test-driven tuning.",
        "",
        "Values are means across three initialization seeds; SD describes variation across those seeds on one fixed synthetic test cohort. It is not a confidence interval.",
        "",
        "| Model | ROC-AUC | Avg precision | Balanced accuracy, validation threshold | Balanced accuracy, 0.5 | Brier score | 10-bin ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        report.append(
            f"| {r['mode']} | {r['roc_auc_mean']:.3f} ± {r['roc_auc_sd']:.3f} | {r['average_precision_mean']:.3f} ± {r['average_precision_sd']:.3f} | {r['balanced_accuracy_mean']:.3f} ± {r['balanced_accuracy_sd']:.3f} | {r['balanced_accuracy_at_0_5_mean']:.3f} ± {r['balanced_accuracy_at_0_5_sd']:.3f} | {r['brier_score_mean']:.3f} ± {r['brier_score_sd']:.3f} | {r['ece_10_bins_mean']:.3f} ± {r['ece_10_bins_sd']:.3f} |"
        )
    report.extend(
        [
            "",
            "ROC-AUC, average precision, Brier score, and ECE use continuous probabilities. The primary balanced accuracy uses a threshold selected to maximize balanced accuracy on the validation split for that mode and seed. The 0.5 threshold result is also included to show the effect of threshold choice. No threshold is selected from test labels.",
            "",
            "See `metrics_by_seed.csv`, `training_history.csv`, `validation_predictions.csv`, `test_predictions.csv`, `split_manifest.json`, and `config.json` for the recorded details. `attention_example.png` visualizes weights from the image-only MIL model for one synthetic bag; attention is not a validated explanation method.",
        ]
    )
    (out_dir / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nSaved outputs under {out_dir.resolve()} ({elapsed:.1f}s)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/main"))
    parser.add_argument("--patients", type=int, default=360)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.output, n_patients=args.patients, max_epochs=args.epochs, batch_size=args.batch_size)
