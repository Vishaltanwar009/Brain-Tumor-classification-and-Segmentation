# Joint Brain Tumor Segmentation & Classification on MRI (FinalNet / MPF-Net)

**Project Team & Authors:**
- **Vishal Tanwar** — Roll No: `2024UCA1809`
- **Kumar Sahil** — Roll No: `2024UCA1817`

A multi-task deep-learning pipeline that **segments** a brain tumor and **classifies** its type (meningioma, glioma, pituitary) from a single contrast-enhanced T1 MRI slice, using one shared network. The repository contains three PyTorch notebooks that evaluate the same model family under different architectures and data-splitting protocols.

| Notebook | Experiment | Model | Input | Split protocol |
|---|---|---|---|---|
| `brain-tt.ipynb` | **Exp. 2** – baseline | FinalNet | 256×256 | Patient-level (honest) |
| `brain-tt-2.ipynb` | **Exp. 2b** – architecture study | FinalNet + ASPP + FPN (`FinalNet-ASPP-FPN-512`) | 512×512 | Patient-level (same test set as Exp. 2) |
| `brain-t-4.ipynb` | **Exp. 3** – paper comparison | `MPF-Net-ASPP-FPN-512` (same model as 2b) | 512×512 | Slice-level (replicates the reference paper) |

> **Naming note:** the notebooks use `FinalNet` and `MPF-Net` for the same code base; Exp. 3 simply uses the newer name.

---

## 1. Problem being solved

Brain tumor diagnosis from MRI usually needs two things: *where* the tumor is (segmentation) and *what kind* it is (classification). Doing this manually is slow and subject to inter-reader variability, and training two separate models ignores that the tasks help each other.

This project solves both with one network, and puts particular emphasis on **evaluation honesty**. The Figshare dataset contains several slices per patient, so a random image-level split leaks near-identical neighbouring slices into the test set and inflates scores. Experiments 2 / 2b therefore split **by patient**, while Experiment 3 deliberately uses the image-level split so results can be compared fairly against papers that used it.

## 2. Key features

- **Joint segmentation + classification** from one forward pass.
- **Pretrained EfficientNet-B0 encoder** (ImageNet) with a U-Net decoder and scSE attention.
- **Deep supervision:** 3 auxiliary segmentation heads + 1 main head.
- **Soft-mask feature fusion:** the predicted tumor probability map weights the encoder features before classification (no threshold, no detach), so the classifier focuses on the tumor region.
- **Learned uncertainty weighting** (Kendall et al.) to balance segmentation and classification losses automatically.
- **Training recipe:** AdamW, warm-up + cosine schedule, EMA weights (0.999), gradient clipping, mixed precision (AMP), early stopping.
- **Patient-level leakage control:** `StratifiedGroupKFold` grouped by patient ID, with explicit assertions that no patient crosses train / validation / test.
- **5-fold ensemble:** probabilities and masks are averaged over the five best-EMA fold checkpoints, then thresholded at 0.5. The held-out test set is evaluated **once**.
- **Thorough metrics:** accuracy, macro precision/recall/F1, one-vs-rest AUC, confusion matrix, Dice, IoU, pixel precision/recall/specificity, HD95, per-class breakdowns, with documented conventions for empty masks.
- **Reproducibility:** seeds, split patient IDs, run config and all per-image predictions are saved to disk.
- **Smoke-test modes** (`SMOKE`, `SMOKE_TEST_ONLY`) and automatic micro-batch probing with gradient accumulation for 512×512 training on small GPUs (2b / 3).

## 3. Model architecture

```
MRI slice (1 ch → replicated to 3 ch)
        │
EfficientNet-B0 encoder (ImageNet weights, strides 2/4/8/16/32)
        │
        ├── [Exp. 2b / 3 only] ASPP on the stride-32 bottleneck
        │       1×1 | 3×3 dilated (rates 2, 4, 6) | global pooling → 1×1 projection, residual (320 ch)
        ├── [Exp. 2b / 3 only] FPN-style top-down fusion over the 5 encoder stages
        │       64-ch laterals + 3×3 smoothing, 1×1 adapters (zero-initialised) added back to each skip
        │
U-Net decoder with scSE attention (256-128-64-32-16 channels)
        │
        ├── Main segmentation head  (full resolution)
        ├── 3 auxiliary heads       (deep supervision)
        │
        └── Soft-mask classifier:
              concat[ global-avg(stride-32), mask-weighted-pool(stride-32), mask-weighted-pool(stride-16) ]
              → Linear(…, 256) → ReLU → Dropout(0.3) → Linear(256, 3)
```

**Losses:** Tversky (α = 0.7) + BCE for segmentation (averaged over main + auxiliary heads); cross-entropy with label smoothing 0.1 for classification; the two are combined with learned uncertainty weights.

**Size:** the 256×256 baseline has about 6.08 M trainable parameters (measured by the architecture check in `brain-tt.ipynb`).

**What changes between experiments**

| | Exp. 2 | Exp. 2b | Exp. 3 |
|---|---|---|---|
| ASPP | – | ✔ | ✔ |
| FPN fusion | – | ✔ | ✔ |
| Input size | 256 | 512 | 512 |
| Batch | 16 | 16 effective (micro-batch 8 × accumulation 2) | same as 2b |
| Test split | ~20 % of patients | identical to Exp. 2 (629 images / 46 patients) | ~10 % of slices (~307 images) |

## 4. Dataset and source

**Figshare Brain Tumor Dataset** (Cheng et al.), used via its Kaggle mirror: [`ashkhagan/figshare-brain-tumor-dataset`](https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset).

- 3,064 T1-weighted contrast-enhanced MRI slices from **233 patients**
- Stored as MATLAB `.mat` files with fields `cjdata.PID`, `image`, `label`, `tumorMask`, `tumorBorder`
- Both MATLAB v5/v7 (read with SciPy) and v7.3/HDF5 (read with h5py) formats are handled
- Labels (from the dataset README): `1 = meningioma`, `2 = glioma`, `3 = pituitary tumor`

| Class | Slices | Patients |
|---|---|---|
| Meningioma | 708 | 82 |
| Glioma | 1,426 | 89 |
| Pituitary | 930 | 62 |

The notebooks verify these counts on load (`STRICT_COUNTS = True`). The dataset's own `cvind.mat` split is **not** used.

**Pre-processing:** per-slice min-max scaling to uint8, resize to the target size (`INTER_AREA` when shrinking, masks with nearest-neighbour), normalisation `(x/255 − 0.449) / 0.226`. **Augmentation** (training only, applied jointly to image and mask): horizontal flip, rotation ±15°, brightness/contrast.

## 5. Installation

The notebooks are written for **Kaggle** (GPU, e.g. T4) and find the data under `/kaggle/input`.

### Option A – Kaggle (recommended)

1. Upload the notebook(s) to Kaggle.
2. *Add Input* → search for `ashkhagan/figshare-brain-tumor-dataset`.
3. Set the accelerator to **GPU**.
4. Turn **Internet ON** the first time (installs `segmentation-models-pytorch` and downloads ImageNet weights for EfficientNet-B0). If Internet must stay off, attach offline wheels and weights as a Kaggle Dataset; the 2b / 3 notebooks print instructions and never silently fall back to a randomly initialised encoder.

### Option B – Local machine

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick the build for your CUDA version
pip install segmentation-models-pytorch timm albumentations opencv-python \
            numpy pandas scipy scikit-learn matplotlib h5py tqdm jupyter
```

Download the dataset from Kaggle and edit two things in the notebooks:

- In `find_root()` (the dataset-location cell) replace `/kaggle/input` with the folder that contains the dataset.
- Outputs go to `./<results folder>` automatically when `/kaggle/working` does not exist.

## 6. Required dependencies

| Package | Version used in the recorded run | Purpose |
|---|---|---|
| Python | 3.10+ | |
| `torch` | 2.11.0 (CUDA 13.0 build) | model, training |
| `segmentation-models-pytorch` | 0.6.0 | U-Net + scSE decoder |
| `timm` | any recent | EfficientNet-B0 (`tu-efficientnet_b0`) |
| `albumentations` | any recent | augmentation |
| `opencv-python` | any recent | resizing |
| `numpy`, `pandas` | any recent | data handling |
| `scipy` | 1.16.3 | `.mat` v5/v7 loading, distance transforms (HD95) |
| `h5py` | 3.14.0 | `.mat` v7.3 (HDF5) loading |
| `scikit-learn` | 1.6.1 | `StratifiedGroupKFold`, metrics |
| `matplotlib`, `tqdm` | any recent | plots, progress bars |

A CUDA GPU is strongly recommended. Memory use measured on the 256×256 baseline: about 1.7–2.0 GiB peak per training step at batch 16. The 512×512 notebooks probe the largest micro-batch that fits automatically (`AUTO_MICRO_BATCH`).

## 7. How to execute the code

1. Open the notebook and run the cells **in order** (top to bottom). Do not skip the config, data-loading and split cells.
2. **Dry run first (optional but recommended):** set `SMOKE = True` in the config cell (about 30 patients, 2 epochs, 2 folds). For 2b / 3 you can also set `SMOKE_TEST_ONLY = True` to stop right after the architecture check.
3. Set `SMOKE = False` and run the full notebook.
4. **Run order across experiments:** run `brain-tt.ipynb` (Exp. 2) first. `brain-tt-2.ipynb` (Exp. 2b) reuses its `Figshare_Experiment_Results/split_info.json` (attach it as a Kaggle input) so the test set is exactly the same 629 images / 46 patients; if absent it regenerates the identical deterministic split and asserts those counts. `brain-t-4.ipynb` (Exp. 3) is independent.

Outputs are written to:

| Notebook | Output folder |
|---|---|
| `brain-tt.ipynb` | `Figshare_Experiment_Results/` |
| `brain-tt-2.ipynb` | `Figshare_Experiment_ASPP_FPN_512/` |
| `brain-t-4.ipynb` | `Figshare_Experiment3_SliceLevel_512/` |

Each folder contains `checkpoints/` (per-fold `foldK_best_ema.pt` and `final_ensemble.pt`), `figures/`, metric tables (`classification_results.csv`, `segmentation_results.csv`, `per_class_metrics.csv`), JSON metric files, `training_history.csv`, split files, `run_config.json` and `test_per_image_results.csv`.

## 8. How to train and test the model

Training and testing are part of the same notebook flow:

1. **Load & verify data.** Reads all `.mat` files, resizes, and checks the counts against the published dataset statistics.
2. **Split.**
   - *Exp. 2 / 2b:* 300 seeded `StratifiedGroupKFold` candidates are screened for class balance (using only labels and patient IDs). The best gives a ~20 % patient hold-out test set and 5 patient-level CV folds on the rest. Leakage assertions then run.
   - *Exp. 3:* stratified slice-level split (~10 % test, 5-fold CV on the remainder). Patient overlap is measured and reported instead of forbidden.
3. **Train 5 folds.** Each fold trains up to 35 epochs (batch 16, AdamW lr 2e-4, weight decay 1e-4, cosine schedule with one-epoch warm-up, EMA 0.999) with early stopping (patience 8). The validation fold is used **only** for early stopping and checkpoint selection (score = macro-F1 + per-image Dice).
4. **Test.** The five best-EMA fold checkpoints are ensembled by averaging softmax and sigmoid probabilities, then thresholding masks at 0.5. The ensemble is evaluated **once** on the untouched test set.
5. **Report.** Table A (classification), Table B (segmentation), per-class metrics, and the figures (confusion matrix, ROC curves, per-class Dice, training curves, representative segmentations).

**Inference with a saved ensemble:** `final_ensemble.pt` stores all five state dicts plus metadata (classes, image size, threshold). Rebuild `FinalNet(NC, pretrained=False)`, load each state dict, average the outputs, and threshold the mask probability at the stored value.

**Tips:** if you hit CUDA out-of-memory, lower `HP["batch"]` (256 px) or `HP["micro_batch"]` (512 px). GPU runs are not bit-wise deterministic even with fixed seeds.

## 9. Sample output / results

Results below are from the recorded run of **`brain-tt.ipynb` (Exp. 2: FinalNet, 256×256, patient-level)**, evaluated on the hold-out test set of 629 images from 46 patients.

> **Important:** `brain-tt-2.ipynb` (Exp. 2b) and `brain-t-4.ipynb` (Exp. 3) were provided **without executed outputs**, so no results for them are reported here. Run them and add the numbers to this README. Nothing has been estimated.

**Confusion matrix** (rows = true class, columns = predicted)

| | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| **Meningioma** | 128 | 14 | 0 |
| **Glioma** | 0 | 296 | 0 |
| **Pituitary** | 0 | 0 | 191 |

**Saved figures** (in `figures/`): `data_alignment_check.png`, `confusion_matrix_test.png`, `roc_curves_test.png`, `per_class_dice_test.png`, `training_history.png`, `representative_segmentations.png` (worst / median / best case per class, with ground truth in green and prediction in red).

## 10. Performance metrics

### Exp. 2 (patient-level, 256×256): 5-fold ensemble on the hold-out test set

**Table A – Classification**

| Metric | Value |
|---|---|
| Accuracy | 0.9777 |
| Macro precision | 0.9849 |
| Macro recall | 0.9671 |
| Macro F1 | 0.9750 |
| Macro ROC-AUC (one-vs-rest) | 0.9996 |

| Class | Precision | Recall | F1 | AUC | Support |
|---|---|---|---|---|---|
| Meningioma | 1.0000 | 0.9014 | 0.9481 | 0.9989 | 142 |
| Glioma | 0.9548 | 1.0000 | 0.9769 | 0.9997 | 296 |
| Pituitary | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 191 |

**Table B – Segmentation** (threshold 0.5, 256×256, all 629 images included)

| Metric | Mean per-image | Global (pooled pixels) |
|---|---|---|
| **Dice** (primary: per-image) | **0.7751** | 0.8131 |
| IoU | 0.6840 | 0.6851 |
| Pixel precision | 0.7892 | 0.8411 |
| Pixel recall / sensitivity | 0.8153 | 0.7870 |
| Specificity | 0.9974 | 0.9974 |

HD95 (pixels): mean 9.95 and median 5.00 over the 609 images where it is defined; 21.15 mean when the 20 undefined cases (empty prediction) are penalised with the image diagonal (362 px).

| Class | Images | Dice (mean ± SD) | Dice (pooled) | HD95 mean (px) |
|---|---|---|---|---|
| Meningioma | 142 | 0.9160 ± 0.1413 | 0.9405 | 5.03 |
| Glioma | 296 | 0.6727 ± 0.2860 | 0.7395 | 15.92 |
| Pituitary | 191 | 0.8291 ± 0.1651 | 0.8554 | 4.92 |

**Cross-validation (validation folds, mean ± std over 5 folds):** accuracy 0.9582 ± 0.0170, macro-F1 0.9524 ± 0.0210, Dice 0.7813 ± 0.0086. These were used for checkpoint selection and are therefore optimistic. The test-set numbers above are the unbiased estimate.

**Reading the results:**
- Classification is strong. All errors are meningiomas predicted as glioma (14 of 142).
- Segmentation is the weaker task, driven mostly by gliomas: 19 of the 20 empty predictions and most of the Dice = 0 cases (27 in total) are gliomas, which pulls the per-image Dice down.

### Exp. 2b and Exp. 3

Both notebooks report the same metric set (Table A / Table B), and Exp. 3 adds per-fold test metrics, pixel accuracy and a comparison with the reference paper (Sahoo et al., *Biocybern Biomed Eng* 43 (2023) 616–633). Exp. 3 uses a slice-level split, which can place slices of the same patient in train and test, so its numbers are expected to be optimistic. Use Exp. 2 / 2b as the reference for generalisation and Exp. 3 only for like-for-like comparison with image-level papers.

---

## Metric conventions

- **Dice (primary):** mean of per-image Dice over *all* test images; both-empty = 1; missed tumor or false alarm = 0. The global pooled variant is reported alongside.
- **HD95:** 95th-percentile Hausdorff distance in pixels (MedPy-style); undefined when exactly one mask is empty (counted and reported separately, plus a penalised mean).
- **AUC:** one-vs-rest per class; macro-AUC is the mean over defined classes and is cross-checked against scikit-learn.
- **Threshold:** a single 0.5 threshold, applied after probability averaging.

## Limitations

- Single dataset, single imaging contrast (T1-CE), 2-D slices only; no external validation.
- The test set has just 46 patients, so metrics carry meaningful uncertainty; confidence intervals are not computed.
- Not a medical device. For research use only; it must not be used for clinical decisions.

## Citation / credits

- Dataset: Cheng et al., Figshare Brain Tumor Dataset (Kaggle mirror: `ashkhagan/figshare-brain-tumor-dataset`).
- Reference paper for Exp. 3: Sahoo AK, Parida P, Muralibabu K, Dash S. *Efficient simultaneous segmentation and classification of brain tumors from MRI scans using deep learning.* Biocybern Biomed Eng 43 (2023) 616–633.
- Built with PyTorch, `segmentation-models-pytorch`, `timm`, Albumentations and scikit-learn.
