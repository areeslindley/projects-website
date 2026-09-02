"""Shared data, metrics, and model helpers for the cancer-imaging project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

IMAGE_SIZE = 128
NPZ_NAME = "breastmnist_128.npz"
MALIGNANT_LABEL_IN_FILE = 0  # MedMNIST: 0 = malignant, 1 = normal/benign
CLASS_NAMES = {0: "benign / normal", 1: "malignant"}
METRIC_ORDER = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "brier",
    "mcc",
]

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:  # pragma: no cover - optional for sklearn-only notebooks
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False


def project_dir() -> Path:
    """Resolve the project directory from a notebook or a script."""
    here = Path(__file__).resolve().parent
    if (here / "cancer_cv_utils.py").exists():
        return here
    alt = Path("projects/cancer-imaging").resolve()
    return alt if (alt / "cancer_cv_utils.py").exists() else here


def data_dir() -> Path:
    return project_dir() / "data"


def artifacts_dir() -> Path:
    path = project_dir() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    path = project_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def npz_path() -> Path:
    return data_dir() / NPZ_NAME


def _to_float_images(arr: np.ndarray) -> np.ndarray:
    images = np.asarray(arr)
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    images = images.astype(np.float32)
    if images.max() > 1.5:
        images = images / 255.0
    return images


def _to_malignant_positive(labels: np.ndarray) -> np.ndarray:
    """Map MedMNIST labels so 1 = malignant (cancer-positive)."""
    raw = np.asarray(labels).reshape(-1).astype(int)
    return (raw == MALIGNANT_LABEL_IN_FILE).astype(np.int64)


def load_splits() -> dict[str, dict[str, np.ndarray]]:
    """Load official BreastMNIST 128×128 splits with malignant as class 1."""
    path = npz_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python projects/cancer-imaging/_prepare_data.py"
        )
    blob = np.load(path)
    splits: dict[str, dict[str, np.ndarray]] = {}
    for name in ("train", "val", "test"):
        images = _to_float_images(blob[f"{name}_images"])
        labels = _to_malignant_positive(blob[f"{name}_labels"])
        splits[name] = {"images": images, "labels": labels}
    return splits


def flatten_images(images: np.ndarray) -> np.ndarray:
    n = images.shape[0]
    return images.reshape(n, -1)


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Binary metrics with malignant as the positive class."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
    }


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """Pick a decision threshold on a validation split (default: max F1)."""
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.15, 0.85, 15):
        score = classification_metrics(y_true, y_prob, threshold=float(t))[metric]
        if score > best_s:
            best_t, best_s = float(t), score
    return best_t


def metrics_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, mets in results.items():
        row = {"model": name}
        for key in METRIC_ORDER:
            row[key] = mets.get(key)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_metrics() -> dict[str, dict[str, Any]]:
    path = artifacts_dir() / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run _train_models.py first.")
    return json.loads(path.read_text())


def load_predictions() -> dict[str, np.ndarray]:
    path = artifacts_dir() / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run _train_models.py first.")
    blob = np.load(path)
    return {key: blob[key] for key in blob.files}


# ---------------------------------------------------------------------------
# Classical computer-vision features
# ---------------------------------------------------------------------------

def _intensity_features(image: np.ndarray) -> tuple[np.ndarray, list[str]]:
    flat = image.ravel()
    h, w = image.shape
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    centre = image[cy0:cy1, cx0:cx1]
    edge_mask = np.ones_like(image, dtype=bool)
    edge_mask[cy0:cy1, cx0:cx1] = False
    hist, _ = np.histogram(flat, bins=16, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    entropy = float(-(hist * np.log(hist)).sum())
    values = np.array(
        [
            float(flat.mean()),
            float(flat.std()),
            float(flat.min()),
            float(flat.max()),
            float(np.percentile(flat, 25)),
            float(np.percentile(flat, 50)),
            float(np.percentile(flat, 75)),
            float(centre.mean()),
            float(image[edge_mask].mean()),
            float((flat < 0.30).mean()),
            entropy,
        ],
        dtype=np.float32,
    )
    names = [
        "int_mean",
        "int_std",
        "int_min",
        "int_max",
        "int_p25",
        "int_p50",
        "int_p75",
        "int_centre_mean",
        "int_edge_mean",
        "int_dark_frac",
        "int_entropy",
    ]
    return values, names


def extract_cv_features(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """HOG + uniform LBP histogram + intensity summaries."""
    from skimage.feature import hog, local_binary_pattern

    feat_rows = []
    names: list[str] | None = None
    for image in images:
        hog_vec = hog(
            image,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True,
        ).astype(np.float32)
        lbp = local_binary_pattern(
            np.clip(image * 255.0, 0, 255).astype(np.uint8),
            P=8,
            R=1,
            method="uniform",
        )
        n_bins = 8 + 2
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )
        intensity, intensity_names = _intensity_features(image)
        row = np.concatenate(
            [hog_vec, lbp_hist.astype(np.float32), intensity]
        )
        if names is None:
            names = (
                [f"hog_{i}" for i in range(len(hog_vec))]
                + [f"lbp_{i}" for i in range(len(lbp_hist))]
                + intensity_names
            )
        feat_rows.append(row)
    return np.vstack(feat_rows), names or []


def feature_group(name: str) -> str:
    if name.startswith("hog_"):
        return "HOG (oriented edges)"
    if name.startswith("lbp_"):
        return "LBP (local texture)"
    return "Intensity / darkness"


def grouped_importance(importances: np.ndarray, names: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"feature": names, "importance": importances})
    frame["group"] = frame["feature"].map(feature_group)
    return (
        frame.groupby("group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )


# ---------------------------------------------------------------------------
# PyTorch models
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class SmallCNN(nn.Module):
        """A compact conv net for 128×128 grayscale ultrasound patches."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(128, 1),
            )

        def forward(self, x: Any) -> Any:
            x = self.features(x)
            x = self.gap(x)
            return self.classifier(x).squeeze(1)

        @property
        def gradcam_layer(self) -> Any:
            return self.features[-3]  # last conv before ReLU (Conv2d)

    def last_conv(model: SmallCNN) -> Any:
        # features: Conv, BN, ReLU, Pool, ... last Conv is index -3 if ending ReLU
        for module in reversed(list(model.features.children())):
            if isinstance(module, nn.Conv2d):
                return module
        raise RuntimeError("No conv layer found")


    def build_resnet18(head_state: dict | None = None, freeze_backbone: bool = True):
        from torchvision.models import ResNet18_Weights, resnet18

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 1)
        if freeze_backbone:
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("fc.")
        if head_state is not None:
            model.fc.load_state_dict(head_state)
        return model


    def imagenet_batch(gray: Any) -> Any:
        """Resize 1×128×128 grayscale tensors to ImageNet-normalised 3×224×224."""
        if gray.ndim == 3:
            gray = gray.unsqueeze(1)
        rgb = gray.repeat(1, 3, 1, 1)
        rgb = F.interpolate(rgb, size=224, mode="bilinear", align_corners=False)
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return (rgb - mean) / std


    @torch.no_grad()
    def predict_torch(
        model: Any,
        images: np.ndarray,
        batch_size: int = 32,
        resnet: bool = False,
        device: str | None = None,
    ) -> np.ndarray:
        model.eval()
        device = device or next(model.parameters()).device
        probs = []
        tensor = torch.from_numpy(np.asarray(images, dtype=np.float32))
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            if resnet:
                batch = imagenet_batch(batch)
            logits = model(batch)
            if logits.ndim > 1:
                logits = logits.reshape(logits.shape[0], -1)[:, 0]
            probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs)


    def gradcam_map(
        model: Any,
        image: np.ndarray,
        target_layer: Any,
        resnet: bool = False,
        device: str | None = None,
    ) -> np.ndarray:
        """Class-activation map for the malignant logit (positive class)."""
        model.eval()
        device = device or next(model.parameters()).device
        activations: list[Any] = []
        gradients: list[Any] = []

        def forward_hook(_module, _inp, output):
            activations.append(output)
            output.register_hook(lambda grad: gradients.append(grad))

        handle = target_layer.register_forward_hook(forward_hook)
        try:
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            batch = tensor.unsqueeze(0).to(device)
            if resnet:
                batch = imagenet_batch(batch)
            elif batch.ndim == 3:
                batch = batch.unsqueeze(1)
            batch = batch.detach().requires_grad_(True)
            logits = model(batch)
            score = logits.reshape(-1)[0]
            model.zero_grad(set_to_none=True)
            score.backward()
            act = activations[0][0].detach()
            grad = gradients[0][0].detach()
            weights = grad.mean(dim=(1, 2))
            cam = torch.relu((weights[:, None, None] * act).sum(0))
            cam = cam / (cam.max() + 1e-8)
            cam = (
                F.interpolate(
                    cam[None, None],
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        finally:
            handle.remove()
        return cam

else:  # pragma: no cover

    class SmallCNN:  # type: ignore[no-redef]
        def __init__(self) -> None:
            raise ImportError("PyTorch is required for SmallCNN")


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a grayscale image with a colour heatmap (matplotlib plasma)."""
    import matplotlib.cm as cm

    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0.0, 1.0)
    heat = cm.plasma(cam)[..., :3]
    rgb = np.repeat(image[..., None], 3, axis=2)
    return np.clip((1 - alpha) * rgb + alpha * heat, 0.0, 1.0)


def display_plotly(fig) -> None:
    """Embed Plotly with CDN JS — fig.show() is blank in Jupyter Book HTML."""
    from IPython.display import HTML, display

    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))
