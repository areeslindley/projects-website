# BreastMNIST ultrasound data

Official **128×128** BreastMNIST split used by the notebooks.

## Bundled file

- `breastmnist_128.npz` — `train` / `val` / `test` images and labels from [MedMNIST](https://medmnist.com/).

Notebooks read **only** this NPZ so GitHub Actions does not fetch Zenodo at build time.

MedMNIST stores **0 = malignant**, **1 = normal/benign**. The project remaps labels so **1 = malignant** (the positive class for precision and recall).

| Split | Images | Notes |
| --- | --- | --- |
| train | 546 | Model fitting only |
| val | 78 | Early stopping / PCA components |
| test | 156 | Held-out metrics in every chapter |

## Rebuild from MedMNIST

From the repository root (requires network):

```bash
python projects/cancer-imaging/_prepare_data.py
```

## Attribution

- MedMNIST collection: Yang et al., *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data, 2023. Licensed **CC BY 4.0**.
- Source images: Al-Dhabyani, Gomaa, Khaled, Fahmy, *Dataset of breast ultrasound images*, Data in Brief, 2020 (BUSI).

This project is an independent educational use of the standardised 128×128 split. It is **not** a medical device and is **not** for diagnosis.
