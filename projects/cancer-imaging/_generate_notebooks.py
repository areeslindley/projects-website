"""Generate cancer-imaging notebooks. Run from the project directory or repo root."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

PROJ = Path(__file__).parent
KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}

SETUP = r'''
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

PROJ_DIR = Path(".").resolve()
if not (PROJ_DIR / "cancer_cv_utils.py").exists():
    PROJ_DIR = Path("projects/cancer-imaging").resolve()
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

from cancer_cv_utils import (
    CLASS_NAMES,
    METRIC_ORDER,
    artifacts_dir,
    classification_metrics,
    display_plotly,
    extract_cv_features,
    figures_dir,
    grouped_importance,
    load_metrics,
    load_predictions,
    load_splits,
    metrics_frame,
    overlay_heatmap,
    tune_threshold,
)

SPLITS = load_splits()
FIG = figures_dir()
ART = artifacts_dir()
print("Splits:", {k: v["labels"].shape[0] for k, v in SPLITS.items()})
print("Malignant rates:", {k: f"{v['labels'].mean():.1%}" for k, v in SPLITS.items()})
'''


def md(text: str):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def nav(title, prev_l, prev_t, next_l, next_t, desc):
    prev_p = f"[← Previous: {prev_t}]({prev_l})" if prev_l else ""
    next_p = f"[Next: {next_t} →]({next_l})" if next_l else ""
    sep = " | " if prev_p and next_p else ""
    return md(f"# {title}\n\n**Navigation**: {prev_p}{sep}{next_p}\n\n{desc}\n")


def footer(prev_l, prev_t, next_l, next_t):
    prev_p = f"[← Previous: {prev_t}]({prev_l})" if prev_l else ""
    next_p = f"[Next: {next_t} →]({next_l})" if next_l else ""
    sep = " | " if prev_p and next_p else ""
    return md(f"---\n\n**Navigation**: {prev_p}{sep}{next_p}\n")


def save(name, cells):
    nb = {"cells": cells, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}
    (PROJ / name).write_text(json.dumps(nb, indent=1))
    print(f"Wrote {name}")


def nb01():
    cells = [
        nav(
            "Images & Labels",
            "index.md", "Project Overview",
            "02_classical_models.ipynb", "Classical Models",
            "BreastMNIST 128×128 ultrasound: official splits, class balance, and what malignant vs benign looks like at this resolution.",
        ),
        md(
            "> **Disclaimer.** Educational benchmark only — not a diagnostic tool. "
            "Images are centre-cropped/resized BUSI scans at 128×128, far coarser than a clinical workstation."
        ),
        md(
            "## The task\n\n"
            "Each image is a breast ultrasound patch labelled **malignant** or **benign/normal**. "
            "MedMNIST stores malignant as class 0; we remap so that **1 = malignant** (the positive class "
            "for precision and recall). Prevalence is about **27%** in every official split, so a classifier "
            "that never predicts cancer still records ~73% accuracy.\n\n"
            "That is why later chapters lead with recall, precision, balanced accuracy, ROC-AUC and PR-AUC "
            "rather than accuracy alone."
        ),
        code(SETUP),
        md("## Split sizes and class balance"),
        code(
            "rows = []\n"
            "for name, split in SPLITS.items():\n"
            "    y = split['labels']\n"
            "    rows.append({\n"
            "        'split': name,\n"
            "        'n': int(len(y)),\n"
            "        'malignant': int(y.sum()),\n"
            "        'benign / normal': int((y == 0).sum()),\n"
            "        'prevalence': float(y.mean()),\n"
            "    })\n"
            "pd.DataFrame(rows).set_index('split')"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(6.2, 3.6))\n"
            "names = list(SPLITS)\n"
            "mal = [SPLITS[k]['labels'].mean() for k in names]\n"
            "ben = [1 - m for m in mal]\n"
            "ax.bar(names, ben, label='benign / normal', color='#1a6b7a')\n"
            "ax.bar(names, mal, bottom=ben, label='malignant', color='#c0392b')\n"
            "ax.set_ylabel('Share of images')\n"
            "ax.set_ylim(0, 1)\n"
            "ax.set_title('Class balance is the same in train, val, and test')\n"
            "ax.legend(frameon=False)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## What the two classes look like\n\n"
            "Malignant lesions on ultrasound often appear as darker (hypoechoic) regions with irregular "
            "margins and posterior shadowing. Benign cysts and fibroadenomas tend to be more oval and "
            "sharply bounded. At 128×128 those cues are still visible, but measurement crosshairs from "
            "the original scans remain in some frames — a shortcut a model could cheat with."
        ),
        code(
            "from IPython.display import Image, display\n"
            "display(Image(str(FIG / '01_class_montage.png')))"
        ),
        md("## Mean images\n\nAveraging every training example of each class highlights where they differ on average."),
        code("display(Image(str(FIG / '01_mean_images.png')))"),
        md(
            "## Intensity distributions\n\n"
            "If malignant patches are systematically darker, a one-pixel statistic already carries signal — "
            "which is why the later random forest includes intensity summaries alongside HOG and LBP."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7.2, 3.8))\n"
            "for cls, colour, label in [(0, '#1a6b7a', 'benign / normal'), (1, '#c0392b', 'malignant')]:\n"
            "    pix = SPLITS['train']['images'][SPLITS['train']['labels'] == cls].ravel()\n"
            "    ax.hist(pix, bins=40, density=True, alpha=0.55, color=colour, label=label)\n"
            "ax.set_xlabel('Pixel intensity (0–1)')\n"
            "ax.set_ylabel('Density')\n"
            "ax.set_title('Training-set intensity histograms')\n"
            "ax.legend(frameon=False)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Attribution\n\n"
            "Yang et al., *MedMNIST v2*, Scientific Data (2023). Source images: Al-Dhabyani et al., "
            "*Dataset of breast ultrasound images*, Data in Brief (2020). See `data/README.md`."
        ),
        footer("index.md", "Project Overview", "02_classical_models.ipynb", "Classical Models"),
    ]
    save("01_images.ipynb", cells)


def nb02():
    cells = [
        nav(
            "Classical Models",
            "01_images.ipynb", "Images & Labels",
            "03_deep_learning.ipynb", "Deep Learning",
            "A majority dummy, logistic regression on PCA-reduced pixels, and a random forest on HOG / LBP / intensity features.",
        ),
        md(
            "## Why start simple?\n\n"
            "With **546** training images, a model that sees 16,384 raw pixels is over-parameterised. "
            "The first two models force a strong inductive bias: (1) a linear decision in a 20–80 "
            "dimensional PCA subspace, and (2) engineered edge/texture descriptors that computer vision "
            "used for a decade before conv nets. Both train in seconds and give a yardstick the CNNs "
            "have to beat.\n\n"
            "All reported numbers in this chapter are **test-set** scores. PCA rank and the forest's "
            "decision threshold are chosen on the **validation** split only."
        ),
        code(SETUP),
        md("## Dummy classifier\n\nAlways predict the majority class (benign / normal). Accuracy looks respectable; cancer detection is zero."),
        code(
            "from sklearn.dummy import DummyClassifier\n"
            "from IPython.display import Image, display\n"
            "\n"
            "y_tr, y_te = SPLITS['train']['labels'], SPLITS['test']['labels']\n"
            "x_tr = SPLITS['train']['images'].reshape(len(y_tr), -1)\n"
            "x_te = SPLITS['test']['images'].reshape(len(y_te), -1)\n"
            "dummy = DummyClassifier(strategy='most_frequent', random_state=42).fit(x_tr, y_tr)\n"
            "dummy_prob = dummy.predict_proba(x_te)[:, 1]\n"
            "dummy_mets = classification_metrics(y_te, dummy_prob)\n"
            "pd.Series(dummy_mets)[METRIC_ORDER].to_frame('Dummy (majority)')"
        ),
        md("## Logistic regression on PCA-reduced pixels"),
        md(
            "Each image is flattened, standardised, and projected to $k$ principal components. "
            "$k$ is picked by validation ROC-AUC from $\\{20,40,60,80\\}$. Class-weighted logistic "
            "regression then predicts malignancy. The decision threshold is tuned on validation F1 "
            "so we are not locked to 0.5 on a 27% positive class."
        ),
        code(
            "from sklearn.decomposition import PCA\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "\n"
            "x_va = SPLITS['val']['images'].reshape(len(SPLITS['val']['labels']), -1)\n"
            "y_va = SPLITS['val']['labels']\n"
            "best_auc, best_k = -1, 40\n"
            "for k in (20, 40, 60, 80):\n"
            "    pipe = Pipeline([\n"
            "        ('scaler', StandardScaler()),\n"
            "        ('pca', PCA(n_components=k, random_state=42)),\n"
            "        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),\n"
            "    ]).fit(x_tr, y_tr)\n"
            "    auc = classification_metrics(y_va, pipe.predict_proba(x_va)[:, 1])['roc_auc']\n"
            "    if auc > best_auc:\n"
            "        best_auc, best_k = auc, k\n"
            "\n"
            "logreg = Pipeline([\n"
            "    ('scaler', StandardScaler()),\n"
            "    ('pca', PCA(n_components=best_k, random_state=42)),\n"
            "    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),\n"
            "]).fit(x_tr, y_tr)\n"
            "log_val = logreg.predict_proba(x_va)[:, 1]\n"
            "log_prob = logreg.predict_proba(x_te)[:, 1]\n"
            "log_t = tune_threshold(y_va, log_val, metric='f1')\n"
            "log_mets = classification_metrics(y_te, log_prob, threshold=log_t)\n"
            "print(f'PCA components = {best_k}, val AUC = {best_auc:.3f}, threshold = {log_t:.2f}')\n"
            "pd.Series(log_mets)[METRIC_ORDER].to_frame('Logistic + PCA')"
        ),
        md("## HOG + LBP + intensity random forest"),
        md(
            "**HOG** (histogram of oriented gradients) summarises edge direction in 16×16 cells — useful "
            "for mass margins and shadowing. **Uniform LBP** is a local texture histogram. "
            "A handful of intensity statistics (mean, centre vs edge, dark-pixel fraction) capture "
            "hypoechogenicity. A class-weighted random forest of 200 trees is trained on the concatenated vector."
        ),
        code(
            "from sklearn.ensemble import RandomForestClassifier\n"
            "\n"
            "x_tr_f, names = extract_cv_features(SPLITS['train']['images'])\n"
            "x_va_f, _ = extract_cv_features(SPLITS['val']['images'])\n"
            "x_te_f, _ = extract_cv_features(SPLITS['test']['images'])\n"
            "rf = RandomForestClassifier(\n"
            "    n_estimators=200, max_depth=12, min_samples_leaf=3,\n"
            "    class_weight='balanced', random_state=42, n_jobs=-1,\n"
            ").fit(x_tr_f, y_tr)\n"
            "rf_val = rf.predict_proba(x_va_f)[:, 1]\n"
            "rf_prob = rf.predict_proba(x_te_f)[:, 1]\n"
            "rf_t = tune_threshold(y_va, rf_val, metric='f1')\n"
            "rf_mets = classification_metrics(y_te, rf_prob, threshold=rf_t)\n"
            "print(f'{len(names)} features, threshold = {rf_t:.2f}')\n"
            "pd.Series(rf_mets)[METRIC_ORDER].to_frame('HOG/LBP + RF')"
        ),
        md("## Side-by-side (this notebook's live fit)"),
        code(
            "live = {\n"
            "    'Dummy (majority)': dummy_mets,\n"
            "    'Logistic + PCA': log_mets,\n"
            "    'HOG/LBP + RF': rf_mets,\n"
            "}\n"
            "metrics_frame(live).round(3)"
        ),
        md(
            "Confusion matrices from the committed training run (same seeds, validation-tuned thresholds):"
        ),
        code(
            "for fname in ['02_cm_dummy.png', '02_cm_logreg.png', '02_cm_rf.png']:\n"
            "    display(Image(str(FIG / fname)))"
        ),
        md(
            "## Takeaways\n\n"
            "- The dummy's accuracy is a trap: **recall is 0**.\n"
            "- PCA-logistic already ranks reasonably (ROC-AUC around 0.80) but is conservative once "
            "the F1-tuned threshold is applied — precision over recall.\n"
            "- Hand-crafted edges lift both ranking quality and F1. That is the baseline a CNN has to beat."
        ),
        footer("01_images.ipynb", "Images & Labels", "03_deep_learning.ipynb", "Deep Learning"),
    ]
    save("02_classical_models.ipynb", cells)


def nb03():
    cells = [
        nav(
            "Deep Learning",
            "02_classical_models.ipynb", "Classical Models",
            "04_interpretation.ipynb", "Interpretation",
            "A compact CNN trained from scratch on 546 images, then a frozen ImageNet ResNet-18 with a new head.",
        ),
        md(
            "## Two conv nets, two bets\n\n"
            "**Small CNN.** Four conv-BN-ReLU blocks with max-pooling, global average pooling, dropout, "
            "and a single logit. Trained with binary cross-entropy and a positive-class weight of "
            "`n_benign / n_malignant`, horizontal flips, and early stopping on validation ROC-AUC.\n\n"
            "**ResNet-18 transfer.** ImageNet-pretrained backbone, grayscale repeated to three channels "
            "and resized to 224×224, **all convolutional weights frozen**. Only the final linear head "
            "is trained. With a few hundred labels this is usually a better bet than learning filters "
            "from scratch.\n\n"
            "Weights in `artifacts/` were produced by `_train_models.py` so the Jupyter Book build does "
            "not retrain on CPU in CI. This notebook loads those runs and reports held-out metrics."
        ),
        code(SETUP),
        code(
            "from IPython.display import Image, display\n"
            "import torch\n"
            "from cancer_cv_utils import SmallCNN, last_conv\n"
            "\n"
            "def _load_ckpt(path):\n"
            "    try:\n"
            "        return torch.load(path, map_location='cpu', weights_only=False)\n"
            "    except TypeError:\n"
            "        return torch.load(path, map_location='cpu')\n"
            "\n"
            "cnn_ckpt = _load_ckpt(ART / 'small_cnn.pt')\n"
            "res_ckpt = _load_ckpt(ART / 'resnet18_head.pt')\n"
            "cnn = SmallCNN()\n"
            "cnn.load_state_dict(cnn_ckpt['state_dict'])\n"
            "n_params = sum(p.numel() for p in cnn.parameters())\n"
            "print(f'Small CNN parameters: {n_params:,}')\n"
            "print(f\"Grad-CAM target: {type(last_conv(cnn)).__name__}\")\n"
            "print('CNN epochs recorded:', cnn_ckpt['history'][-1]['epoch'])\n"
            "print('ResNet epochs recorded:', res_ckpt['history'][-1]['epoch'])"
        ),
        md("## Training curves (validation ROC-AUC)"),
        code(
            "fig, ax = plt.subplots(figsize=(7.5, 3.8))\n"
            "for hist, label, colour in [\n"
            "    (cnn_ckpt['history'], 'Small CNN', '#8e44ad'),\n"
            "    (res_ckpt['history'], 'ResNet-18 head', '#c0392b'),\n"
            "]:\n"
            "    ax.plot([h['epoch'] for h in hist], [h['val_auc'] for h in hist],\n"
            "            marker='o', label=label, color=colour)\n"
            "ax.set_xlabel('Epoch')\n"
            "ax.set_ylabel('Validation ROC-AUC')\n"
            "ax.set_title('Early stopping on val AUC — not on test')\n"
            "ax.legend(frameon=False)\n"
            "ax.set_ylim(0.45, 1.0)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("## Held-out metrics"),
        code(
            "mets = load_metrics()\n"
            "metrics_frame({\n"
            "    'Small CNN': mets['Small CNN'],\n"
            "    'ResNet-18 transfer': mets['ResNet-18 transfer'],\n"
            "}).round(3)"
        ),
        md(
            "Thresholds were chosen by **validation F1**, then frozen for the test set. "
            "The small CNN lands on a similar accuracy to the random forest; the ResNet head "
            "trades precision for recall (it misses fewer cancers, flags more benign scans). "
            "ROC-AUC and PR-AUC, which do not depend on a single threshold, both favour transfer learning."
        ),
        code(
            "for fname in ['03_cm_cnn.png', '03_cm_resnet.png']:\n"
            "    display(Image(str(FIG / fname)))"
        ),
        md(
            "## Why the from-scratch CNN is not an automatic win\n\n"
            "546 labelled 128×128 scans are a small conv-net dataset. Data augmentation here is only "
            "a flip; there is no multi-site sample. ImageNet filters, even from natural photos, still "
            "transfer enough edge and blob detectors to improve **ranking quality** (AUC) and **recall**. "
            "The next chapter asks whether those nets are looking at the lesion."
        ),
        footer("02_classical_models.ipynb", "Classical Models", "04_interpretation.ipynb", "Interpretation"),
    ]
    save("03_deep_learning.ipynb", cells)


def nb04():
    cells = [
        nav(
            "Interpretation",
            "03_deep_learning.ipynb", "Deep Learning",
            "05_evaluation.ipynb", "Evaluation",
            "What each model uses: inverse-mapped logistic weights, grouped forest importances, and Grad-CAM overlays.",
        ),
        md(
            "## Three kinds of explanation\n\n"
            "1. **Linear map.** Logistic coefficients in PCA space, mapped back to 128×128 pixels "
            "(undoing PCA and standardisation). Red regions raise the malignant logit.\n"
            "2. **Feature families.** Random-forest Gini importance plus grouped permutation "
            "(shuffle all HOG bins together, then LBP, then intensity) measured as drop in ROC-AUC.\n"
            "3. **Grad-CAM.** For each conv net, the last convolutional feature map is weighted by "
            "the gradient of the malignant logit. Overlays are shown for true positives, false positives, "
            "and false negatives on the test set."
        ),
        code(SETUP),
        code("from IPython.display import Image, display"),
        md("## Logistic regression — which pixels?"),
        code("display(Image(str(FIG / '04_logistic_coef.png')))"),
        md(
            "The map is a *global* explanation: one weight image for the whole test set, not a per-case heatmap. "
            "Because PCA mixes pixels, the pattern is spatially smooth rather than a sharp lesion outline. "
            "Treat it as 'where a linear model can put weight', not as a segmentation."
        ),
        md("## Random forest — which feature family?"),
        code(
            "imp = pd.read_csv(ART / 'rf_grouped_importance.csv')\n"
            "display(imp.round(3))\n"
            "display(Image(str(FIG / '04_rf_groups.png')))"
        ),
        md(
            "HOG (oriented edges) dominates both Gini share and the permutation drop in AUC. "
            "That matches the radiology story: margins and shadowing are edge structure. "
            "LBP texture and raw intensity add little once HOG is present — the forest is mostly "
            "an edge detector with a non-linear head."
        ),
        md("## Small CNN — Grad-CAM"),
        code("display(Image(str(FIG / '04_gradcam_cnn.png')))"),
        md("## ResNet-18 — Grad-CAM"),
        code("display(Image(str(FIG / '04_gradcam_resnet.png')))"),
        md(
            "## How to read the grids\n\n"
            "- **True positives** with high $\\hat p$ should light up the mass or its posterior shadow, "
            "not the empty periphery.\n"
            "- **False positives** show the shortcuts: a benign cyst, a bright artefact, or a measurement "
            "crosshair that happens to correlate with labels.\n"
            "- **False negatives** are the dangerous misses. If Grad-CAM ignores a visible irregular mass, "
            "the representation failed; if the mass is subtle even to the eye at 128×128, the limit is the data.\n\n"
            "Grad-CAM is a localisation hint, not a proof of causal reasoning. It cannot tell us the model "
            "*understands* malignancy — only where in the pixel grid the malignant logit is sensitive."
        ),
        footer("03_deep_learning.ipynb", "Deep Learning", "05_evaluation.ipynb", "Evaluation"),
    ]
    save("04_interpretation.ipynb", cells)


def nb05():
    cells = [
        nav(
            "Evaluation",
            "04_interpretation.ipynb", "Interpretation",
            "index.md", "Project Overview",
            "Held-out comparison: accuracy, precision, recall, F1, specificity, ROC/PR, Brier score, and MCC.",
        ),
        md(
            "## What to trust on an imbalanced medical task\n\n"
            "| Metric | What it answers |\n"
            "| --- | --- |\n"
            "| **Accuracy** | Overall hit rate. Inflated by the 73% benign majority. |\n"
            "| **Precision (PPV)** | Of scans flagged malignant, how many are? |\n"
            "| **Recall (sensitivity)** | Of true cancers, how many did we catch? |\n"
            "| **Specificity** | Of benign scans, how many did we leave unflagged? |\n"
            "| **F1** | Harmonic mean of precision and recall; used to pick thresholds on val. |\n"
            "| **Balanced accuracy** | Mean of recall and specificity; chance is 0.5. |\n"
            "| **ROC-AUC** | Ranking quality across all thresholds. |\n"
            "| **PR-AUC** | Ranking quality with emphasis on the positive class. |\n"
            "| **Brier** | Mean squared error of predicted probabilities (lower is better). |\n"
            "| **MCC** | Correlation between predictions and labels; 0 is chance. |\n\n"
            "There is no single winner. A screening-style reader may prefer recall; a second-reader "
            "tool may prefer precision. The dummy model makes that tension obvious."
        ),
        code(SETUP),
        code(
            "mets = load_metrics()\n"
            "preds = load_predictions()\n"
            "table = metrics_frame(mets).round(3)\n"
            "table"
        ),
        md("## Interactive metric table"),
        code(
            "import plotly.graph_objects as go\n"
            "show = table.reset_index().rename(columns={'model': 'Model'})\n"
            "fig = go.Figure(data=[go.Table(\n"
            "    header=dict(values=list(show.columns), fill_color='#0b3d4a', font=dict(color='white')),\n"
            "    cells=dict(values=[show[c] for c in show.columns], align='left'),\n"
            ")])\n"
            "fig.update_layout(margin=dict(l=0, r=0, t=8, b=8), height=280)\n"
            "display_plotly(fig)"
        ),
        md("## Bars and ranking curves"),
        code(
            "from IPython.display import Image, display\n"
            "display(Image(str(FIG / '05_metrics_bars.png')))\n"
            "display(Image(str(FIG / '05_roc_pr.png')))"
        ),
        md("## ROC and precision–recall from stored test probabilities"),
        code(
            "from sklearn.metrics import precision_recall_curve, roc_curve, auc as sk_auc\n"
            "\n"
            "y = preds['y_true']\n"
            "series = [\n"
            "    ('Dummy', preds['dummy']),\n"
            "    ('Logistic + PCA', preds['logreg']),\n"
            "    ('HOG/LBP + RF', preds['rf']),\n"
            "    ('Small CNN', preds['cnn']),\n"
            "    ('ResNet-18', preds['resnet']),\n"
            "]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))\n"
            "for name, prob in series:\n"
            "    fpr, tpr, _ = roc_curve(y, prob)\n"
            "    prec, rec, _ = precision_recall_curve(y, prob)\n"
            "    axes[0].plot(fpr, tpr, label=f'{name} ({sk_auc(fpr, tpr):.2f})')\n"
            "    axes[1].plot(rec, prec, label=f'{name} ({sk_auc(rec, prec):.2f})')\n"
            "axes[0].plot([0, 1], [0, 1], ls='--', c='grey', lw=0.8)\n"
            "axes[0].set_xlabel('False positive rate'); axes[0].set_ylabel('Recall')\n"
            "axes[0].set_title('ROC')\n"
            "axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')\n"
            "axes[1].set_title('Precision–recall')\n"
            "axes[1].axhline(y.mean(), ls=':', c='grey', label='Prevalence')\n"
            "for ax in axes:\n"
            "    ax.legend(frameon=False, fontsize=8)\n"
            "fig.tight_layout()\n"
            "plt.show()"
        ),
        md("## Calibration (Brier and reliability)"),
        code(
            "from sklearn.calibration import calibration_curve\n"
            "\n"
            "brier_of = {\n"
            "    'Logistic': mets['Logistic + PCA']['brier'],\n"
            "    'RF': mets['HOG/LBP + RF']['brier'],\n"
            "    'CNN': mets['Small CNN']['brier'],\n"
            "    'ResNet': mets['ResNet-18 transfer']['brier'],\n"
            "}\n"
            "fig, ax = plt.subplots(figsize=(5.6, 4.6))\n"
            "ax.plot([0, 1], [0, 1], ls='--', c='grey', lw=0.8, label='Perfect')\n"
            "for name, key in [('Logistic', 'logreg'), ('RF', 'rf'), ('CNN', 'cnn'), ('ResNet', 'resnet')]:\n"
            "    frac, meanp = calibration_curve(y, preds[key], n_bins=6, strategy='quantile')\n"
            "    ax.plot(meanp, frac, marker='o', label=f'{name} (Brier={brier_of[name]:.3f})')\n"
            "ax.set_xlabel('Predicted P(malignant)')\n"
            "ax.set_ylabel('Observed malignant rate')\n"
            "ax.set_title('Reliability diagram (test set)')\n"
            "ax.legend(frameon=False, fontsize=8)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "## Error counts at the chosen thresholds\n\n"
            "False negatives are missed cancers; false positives are extra work for a human reader."
        ),
        code(
            "err = pd.DataFrame({\n"
            "    name: {'TP': m['tp'], 'FP': m['fp'], 'TN': m['tn'], 'FN': m['fn'], 'threshold': m['threshold']}\n"
            "    for name, m in mets.items()\n"
            "}).T\n"
            "err"
        ),
        md(
            "## What improved?\n\n"
            "- **Dummy → logistic:** ranking appears (ROC-AUC ~0.80). Accuracy barely moves; recall is no longer zero.\n"
            "- **Logistic → HOG/LBP forest:** best precision of the ladder, strong accuracy and AUC. "
            "Classical CV still earns its keep on a small dataset.\n"
            "- **Forest → small CNN:** similar accuracy, slightly higher F1 and PR-AUC, worse Brier "
            "(probabilities are less calibrated).\n"
            "- **CNN → ResNet-18 transfer:** highest ROC-AUC and PR-AUC and the fewest missed cancers "
            "(FN), at the cost of more false positives and lower accuracy than the forest.\n\n"
            "On 156 test images these gaps are noisy. The honest summary is: **engineered edges beat "
            "raw pixels; transfer learning is the best ranker and the most sensitive detector; "
            "no model is ready for clinic.** Domain shift, 128px resolution, and a single-source "
            "ultrasound set are hard limits."
        ),
        md(
            "## Rebuild\n\n"
            "```bash\n"
            "python projects/cancer-imaging/_prepare_data.py   # if the NPZ is missing\n"
            "python projects/cancer-imaging/_train_models.py   # regenerate weights and figures\n"
            "python projects/cancer-imaging/_generate_notebooks.py\n"
            "```"
        ),
        footer("04_interpretation.ipynb", "Interpretation", "index.md", "Project Overview"),
    ]
    save("05_evaluation.ipynb", cells)


def main():
    nb01()
    nb02()
    nb03()
    nb04()
    nb05()
    print("All notebooks generated.")


if __name__ == "__main__":
    main()
