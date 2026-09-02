"""Train the model ladder and write metrics, weights, and preview figures."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from cancer_cv_utils import (  # noqa: E402
    HAS_TORCH,
    artifacts_dir,
    classification_metrics,
    extract_cv_features,
    figures_dir,
    grouped_importance,
    load_splits,
    overlay_heatmap,
    save_json,
    tune_threshold,
)

plt.style.use("seaborn-v0_8-whitegrid")
RNG = 42
MALIGNANT_COLOR = "#c0392b"
BENIGN_COLOR = "#1a6b7a"


def _save_fig(fig, name: str) -> None:
    path = figures_dir() / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {path.name}")


def plot_montage(splits: dict) -> None:
    images = splits["train"]["images"]
    labels = splits["train"]["labels"]
    fig, axes = plt.subplots(2, 6, figsize=(11, 4.2))
    for row, (cls, title, color) in enumerate(
        [
            (0, "Benign / normal", BENIGN_COLOR),
            (1, "Malignant", MALIGNANT_COLOR),
        ]
    ):
        idx = np.where(labels == cls)[0][:6]
        for col, i in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
            ax.set_axis_off()
            if col == 0:
                ax.set_title(title, loc="left", color=color, fontsize=10)
    fig.suptitle("BreastMNIST 128×128 — training examples", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, "01_class_montage.png")


def plot_mean_images(splits: dict) -> None:
    images = splits["train"]["images"]
    labels = splits["train"]["labels"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    mean_b = images[labels == 0].mean(axis=0)
    mean_m = images[labels == 1].mean(axis=0)
    axes[0].imshow(mean_b, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Mean benign / normal")
    axes[1].imshow(mean_m, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mean malignant")
    diff = mean_m - mean_b
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    axes[2].set_title("Malignant − benign")
    for ax in axes:
        ax.set_axis_off()
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("Where the two classes differ on average")
    fig.tight_layout()
    _save_fig(fig, "01_mean_images.png")


def train_dummy(splits: dict) -> tuple[DummyClassifier, np.ndarray, dict]:
    dummy = DummyClassifier(strategy="most_frequent", random_state=RNG)
    x_train = splits["train"]["images"].reshape(len(splits["train"]["labels"]), -1)
    dummy.fit(x_train, splits["train"]["labels"])
    y = splits["test"]["labels"]
    x_test = splits["test"]["images"].reshape(len(y), -1)
    prob = dummy.predict_proba(x_test)[:, 1]
    return dummy, prob, classification_metrics(y, prob)


def train_logreg(splits: dict) -> tuple[Pipeline, np.ndarray, dict, np.ndarray]:
    x_train = splits["train"]["images"].reshape(len(splits["train"]["labels"]), -1)
    x_val = splits["val"]["images"].reshape(len(splits["val"]["labels"]), -1)
    x_test = splits["test"]["images"].reshape(len(splits["test"]["labels"]), -1)
    y_train = splits["train"]["labels"]
    y_val = splits["val"]["labels"]

    best_auc = -1.0
    best_k = 40
    for k in (20, 40, 60, 80):
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=k, random_state=RNG)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RNG,
                    ),
                ),
            ]
        )
        pipe.fit(x_train, y_train)
        val_prob = pipe.predict_proba(x_val)[:, 1]
        auc = classification_metrics(y_val, val_prob)["roc_auc"]
        if auc > best_auc:
            best_auc, best_k = auc, k

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=best_k, random_state=RNG)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RNG,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    val_prob = pipe.predict_proba(x_val)[:, 1]
    test_prob = pipe.predict_proba(x_test)[:, 1]
    threshold = tune_threshold(splits["val"]["labels"], val_prob, metric="f1")
    mets = classification_metrics(splits["test"]["labels"], test_prob, threshold=threshold)
    mets["n_components"] = int(best_k)
    mets["val_auc"] = float(best_auc)

    pca: PCA = pipe.named_steps["pca"]
    scaler: StandardScaler = pipe.named_steps["scaler"]
    coef = pipe.named_steps["clf"].coef_.ravel()
    # Map logistic weights from PCA space back to pixels (undo PCA + scaling).
    spatial = ((coef @ pca.components_) / scaler.scale_).reshape(128, 128)
    return pipe, test_prob, mets, spatial


def train_rf(splits: dict) -> tuple[RandomForestClassifier, np.ndarray, dict, pd.DataFrame, list[str]]:
    x_train, names = extract_cv_features(splits["train"]["images"])
    x_test, _ = extract_cv_features(splits["test"]["images"])
    y_train = splits["train"]["labels"]
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RNG,
        n_jobs=-1,
    )
    rf.fit(x_train, y_train)
    x_val, _ = extract_cv_features(splits["val"]["images"])
    val_prob = rf.predict_proba(x_val)[:, 1]
    prob = rf.predict_proba(x_test)[:, 1]
    threshold = tune_threshold(splits["val"]["labels"], val_prob, metric="f1")
    mets = classification_metrics(splits["test"]["labels"], prob, threshold=threshold)
    grouped = grouped_importance(rf.feature_importances_, names)
    grouped["permutation_auc"] = _group_permutation_auc(
        rf, x_test, splits["test"]["labels"], names
    )
    return rf, prob, mets, grouped, names


def _group_permutation_auc(model, x, y, names: list[str], repeats: int = 10) -> np.ndarray:
    """Shuffle each feature family and record the drop in ROC-AUC."""
    from cancer_cv_utils import feature_group
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(RNG)
    base = roc_auc_score(y, model.predict_proba(x)[:, 1])
    groups = [feature_group(n) for n in names]
    unique = list(dict.fromkeys(groups))
    drops = []
    for group in unique:
        cols = np.array([g == group for g in groups])
        scores = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[:, cols] = rng.permutation(shuffled[:, cols])
            scores.append(roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        drops.append(base - float(np.mean(scores)))
    order = {g: d for g, d in zip(unique, drops)}
    # Align with grouped_importance row order (sorted by Gini).
    grouped = grouped_importance(model.feature_importances_, names)
    return np.array([order[g] for g in grouped["group"]])


def _torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _augment_batch(batch):
    import torch

    # Horizontal flip
    flip = torch.rand(len(batch), device=batch.device) > 0.5
    if flip.any():
        batch = batch.clone()
        batch[flip] = torch.flip(batch[flip], dims=[-1])
    return batch


def train_small_cnn(splits: dict):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from cancer_cv_utils import SmallCNN, last_conv, predict_torch

    device = _torch_device()
    model = SmallCNN().to(device)
    y_train = splits["train"]["labels"]
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x_tr = torch.from_numpy(splits["train"]["images"]).unsqueeze(1)
    y_tr = torch.from_numpy(splits["train"]["labels"].astype(np.float32))
    x_va = torch.from_numpy(splits["val"]["images"]).unsqueeze(1)
    y_va = splits["val"]["labels"]
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)

    best_state = None
    best_auc = -1.0
    patience = 0
    history = []
    epochs = 40
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = _augment_batch(xb.to(device))
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(x_va.to(device))).cpu().numpy()
        val_auc = classification_metrics(y_va, val_prob)["roc_auc"]
        history.append({"epoch": epoch, "loss": running / len(x_tr), "val_auc": val_auc})
        print(f"    CNN epoch {epoch:02d}  loss={running / len(x_tr):.3f}  val AUC={val_auc:.3f}")
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)
    model.to(device)
    val_prob = predict_torch(model, splits["val"]["images"], resnet=False, device=str(device))
    prob = predict_torch(model, splits["test"]["images"], resnet=False, device=str(device))
    threshold = tune_threshold(splits["val"]["labels"], val_prob, metric="f1")
    mets = classification_metrics(splits["test"]["labels"], prob, threshold=threshold)
    mets["val_auc"] = float(best_auc)
    mets["epochs_trained"] = int(history[-1]["epoch"])
    torch.save({"state_dict": best_state, "history": history}, artifacts_dir() / "small_cnn.pt")
    return model, prob, mets, last_conv(model)


def train_resnet(splits: dict):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from cancer_cv_utils import build_resnet18, imagenet_batch, predict_torch

    device = _torch_device()
    model = build_resnet18(freeze_backbone=True).to(device)
    y_train = splits["train"]["labels"]
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    x_tr = torch.from_numpy(splits["train"]["images"]).unsqueeze(1)
    y_tr = torch.from_numpy(splits["train"]["labels"].astype(np.float32))
    x_va = torch.from_numpy(splits["val"]["images"]).unsqueeze(1)
    y_va = splits["val"]["labels"]
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=16, shuffle=True)

    best_head = None
    best_auc = -1.0
    patience = 0
    history = []
    for epoch in range(1, 25):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = imagenet_batch(_augment_batch(xb.to(device)))
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            val_logits = model(imagenet_batch(x_va.to(device))).squeeze(1)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = classification_metrics(y_va, val_prob)["roc_auc"]
        history.append({"epoch": epoch, "loss": running / len(x_tr), "val_auc": val_auc})
        print(f"    ResNet epoch {epoch:02d}  loss={running / len(x_tr):.3f}  val AUC={val_auc:.3f}")
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_head = {k: v.detach().cpu().clone() for k, v in model.fc.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break
    model.fc.load_state_dict(best_head)
    model.to(device)
    val_prob = predict_torch(
        model, splits["val"]["images"], batch_size=16, resnet=True, device=str(device)
    )
    prob = predict_torch(
        model, splits["test"]["images"], batch_size=16, resnet=True, device=str(device)
    )
    threshold = tune_threshold(splits["val"]["labels"], val_prob, metric="f1")
    mets = classification_metrics(splits["test"]["labels"], prob, threshold=threshold)
    mets["val_auc"] = float(best_auc)
    mets["epochs_trained"] = int(history[-1]["epoch"])
    torch.save({"fc_state_dict": best_head, "history": history}, artifacts_dir() / "resnet18_head.pt")
    return model, prob, mets, model.layer4[-1]


def plot_confusion(y_true, y_prob, title: str, filename: str, threshold: float = 0.5) -> None:
    y_pred = (np.asarray(y_prob).reshape(-1) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    _save_fig(fig, filename)


def plot_metrics_bars(metrics: dict) -> None:
    keys = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc"]
    models = list(metrics)
    x = np.arange(len(keys))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 4.6))
    palette = ["#7f8c8d", "#2980b9", "#16a085", "#8e44ad", "#c0392b"]
    for i, name in enumerate(models):
        vals = [metrics[name][k] for k in keys]
        ax.bar(x + (i - 2) * width, vals, width, label=name, color=palette[i % len(palette)])
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "Balanced acc.", "ROC-AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test metrics — simple models to transfer learning")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    _save_fig(fig, "05_metrics_bars.png")


def plot_roc_pr(y_true, probs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for name, prob in probs.items():
        RocCurveDisplay.from_predictions(y_true, prob, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_true, prob, name=name, ax=axes[1])
    axes[0].set_title("ROC")
    axes[1].set_title("Precision–recall")
    axes[0].plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8, label="Chance")
    fig.suptitle("Test-set ranking quality")
    fig.tight_layout()
    _save_fig(fig, "05_roc_pr.png")


def plot_spatial_coef(spatial: np.ndarray) -> None:
    lim = np.percentile(np.abs(spatial), 99)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    im = ax.imshow(spatial, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_axis_off()
    ax.set_title("Logistic (PCA) — inverse-mapped coefficients")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    _save_fig(fig, "04_logistic_coef.png")


def plot_rf_groups(grouped: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(grouped["group"], grouped["importance"], color=BENIGN_COLOR)
    ax.invert_yaxis()
    ax.set_xlabel("Total Gini importance")
    ax.set_title("Random forest — which feature family matters?")
    fig.tight_layout()
    _save_fig(fig, "04_rf_groups.png")


def _case_indices(y_true, y_prob, kind: str, n: int = 4, threshold: float = 0.5) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    if kind == "tp":
        mask = (y_true == 1) & (y_pred == 1)
        order = np.argsort(-y_prob)
    elif kind == "fp":
        mask = (y_true == 0) & (y_pred == 1)
        order = np.argsort(-y_prob)
    elif kind == "fn":
        mask = (y_true == 1) & (y_pred == 0)
        order = np.argsort(y_prob)
    else:
        mask = (y_true == 0) & (y_pred == 0)
        order = np.argsort(y_prob)
    picked = [i for i in order if mask[i]][:n]
    return np.array(picked, dtype=int)


def plot_gradcam_grid(
    model,
    layer,
    images,
    y_true,
    y_prob,
    resnet: bool,
    filename: str,
    title: str,
    threshold: float = 0.5,
) -> None:
    from cancer_cv_utils import gradcam_map

    device = next(model.parameters()).device
    kinds = [("tp", "True positive"), ("fp", "False positive"), ("fn", "False negative")]
    fig, axes = plt.subplots(3, 4, figsize=(10.5, 8.2))
    for r, (kind, row_title) in enumerate(kinds):
        idxs = _case_indices(y_true, y_prob, kind, n=4, threshold=threshold)
        for c in range(4):
            ax = axes[r, c]
            if c >= len(idxs):
                ax.set_axis_off()
                continue
            i = int(idxs[c])
            cam = gradcam_map(model, images[i], layer, resnet=resnet, device=str(device))
            ax.imshow(overlay_heatmap(images[i], cam))
            ax.set_axis_off()
            if c == 0:
                ax.set_ylabel(row_title)
                ax.set_title(row_title if c == 0 else "", loc="left", fontsize=10)
            ax.set_title(f"p̂={y_prob[i]:.2f}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    _save_fig(fig, filename)


def main() -> None:
    import random

    random.seed(RNG)
    np.random.seed(RNG)
    if HAS_TORCH:
        import torch

        torch.manual_seed(RNG)
        torch.use_deterministic_algorithms(False)

    print("Loading BreastMNIST splits…")
    splits = load_splits()
    for name, split in splits.items():
        y = split["labels"]
        print(f"  {name:5s}  n={len(y):3d}  malignant={y.mean():.1%}")

    figures_dir()
    artifacts_dir()
    plot_montage(splits)
    plot_mean_images(splits)

    print("Training dummy classifier…")
    dummy, dummy_prob, dummy_mets = train_dummy(splits)
    print("  ", dummy_mets)

    print("Training PCA + logistic regression…")
    logreg, log_prob, log_mets, spatial = train_logreg(splits)
    print("  ", {k: log_mets[k] for k in ("accuracy", "precision", "recall", "roc_auc")})

    print("Extracting HOG/LBP features and training random forest…")
    rf, rf_prob, rf_mets, grouped, feat_names = train_rf(splits)
    print("  ", {k: rf_mets[k] for k in ("accuracy", "precision", "recall", "roc_auc")})

    joblib.dump(
        {
            "dummy": dummy,
            "logreg": logreg,
            "rf": rf,
            "feature_names": feat_names,
        },
        artifacts_dir() / "sklearn_models.joblib",
    )
    np.save(artifacts_dir() / "logistic_spatial_coef.npy", spatial)
    grouped.to_csv(artifacts_dir() / "rf_grouped_importance.csv", index=False)

    if not HAS_TORCH:
        sys.exit("PyTorch is required to train the CNN models.")

    print("Training small CNN…")
    cnn, cnn_prob, cnn_mets, cnn_layer = train_small_cnn(splits)
    print("  ", {k: cnn_mets[k] for k in ("accuracy", "precision", "recall", "roc_auc")})

    print("Training frozen ResNet-18 head…")
    resnet, res_prob, res_mets, res_layer = train_resnet(splits)
    print("  ", {k: res_mets[k] for k in ("accuracy", "precision", "recall", "roc_auc")})

    metrics = {
        "Dummy (majority)": dummy_mets,
        "Logistic + PCA": log_mets,
        "HOG/LBP + RF": rf_mets,
        "Small CNN": cnn_mets,
        "ResNet-18 transfer": res_mets,
    }
    save_json(artifacts_dir() / "metrics.json", metrics)

    y_test = splits["test"]["labels"]
    np.savez_compressed(
        artifacts_dir() / "predictions.npz",
        y_true=y_test,
        dummy=dummy_prob,
        logreg=log_prob,
        rf=rf_prob,
        cnn=cnn_prob,
        resnet=res_prob,
    )

    plot_spatial_coef(spatial)
    plot_rf_groups(grouped)
    plot_metrics_bars(metrics)
    plot_roc_pr(
        y_test,
        {
            "Dummy": dummy_prob,
            "Logistic + PCA": log_prob,
            "HOG/LBP + RF": rf_prob,
            "Small CNN": cnn_prob,
            "ResNet-18": res_prob,
        },
    )
    for name, prob, fname, mets in [
        ("Dummy (majority)", dummy_prob, "02_cm_dummy.png", dummy_mets),
        ("Logistic + PCA", log_prob, "02_cm_logreg.png", log_mets),
        ("HOG/LBP + RF", rf_prob, "02_cm_rf.png", rf_mets),
        ("Small CNN", cnn_prob, "03_cm_cnn.png", cnn_mets),
        ("ResNet-18 transfer", res_prob, "03_cm_resnet.png", res_mets),
    ]:
        plot_confusion(y_test, prob, name, fname, threshold=mets["threshold"])

    print("Computing Grad-CAM case grids…")
    images = splits["test"]["images"]
    plot_gradcam_grid(
        cnn,
        cnn_layer,
        images,
        y_test,
        cnn_prob,
        resnet=False,
        filename="04_gradcam_cnn.png",
        title="Small CNN — Grad-CAM (what the conv net attends to)",
        threshold=cnn_mets["threshold"],
    )
    plot_gradcam_grid(
        resnet,
        res_layer,
        images,
        y_test,
        res_prob,
        resnet=True,
        filename="04_gradcam_resnet.png",
        title="ResNet-18 — Grad-CAM on malignant predictions",
        threshold=res_mets["threshold"],
    )

    print("Done. Artifacts in", artifacts_dir())


if __name__ == "__main__":
    main()
