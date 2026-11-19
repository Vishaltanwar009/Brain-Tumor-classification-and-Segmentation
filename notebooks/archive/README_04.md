# MPF-Net: Mask-Pooled Fusion Network for Joint Brain Tumor Segmentation and Classification from MRI

MPF-Net is a single end-to-end network that segments the tumor and classifies the MRI slice
into the classes of each dataset (4 classes on D1 and D3, 5 classes on D2). The experiments follow the datasets and evaluation
setup of *"A Computationally Efficient and Improved Brain Tumor Recognition System by
MRI–Segmentation Integrated Classification Network"* (BRAIN-CNet, J. Imaging Inform. Med., 2026).

- **Authors:** Kumar Sahil, Vishal Tanwar
- **Notebook:** `mpfnet_brain_tumor_kaggle.ipynb` (runs on Kaggle, GPU + Internet on)

---

## 1. Datasets

| ID | Kaggle dataset | Images | Classes | Masks |
|----|----------------|--------|---------|-------|
| D1 | `atikaakter11/brain-tumor-segmentation-dataset` | 4237 | normal 1595, glioma 649, meningioma 999, pituitary 994 | yes |
| D2 | `shubhamgajjar/brain-tumor-classification-2d` | 3465 (paper: 3670) | astrocytoma 529, glioblastoma 732, gliomas 1237, meningioma 640, neurocytoma 327 (5 classes, no normal) | 3349 images (116 neurocytoma images have no mask) |
| D3 | `wajeehaldeenaljunaid/mixed-brain-tumor-dataset` | 6380 | 1595 per class (normal, glioma, meningioma, pituitary) | yes |

Datasets run: **D1, D2 and D3**

Preprocessing: grayscale slice resized to 512×512 (D1, D3) or 224×224 (D2) with bilinear interpolation, scaled to [0, 1]
and normalized with the ImageNet grayscale mean/std (0.449 / 0.226), then replicated to 3 channels at the model input. Masks are binarized and resized with nearest-neighbor.
Healthy slices get an empty mask. No blur and no cropping.

## 2. Model

| Part | Setting |
|------|---------|
| Encoder | EfficientNet-B0, ImageNet pretrained (`tu-efficientnet_b0` via `segmentation_models_pytorch`) |
| Decoder | U-Net, widths (256, 128, 64, 32, 16), scSE attention |
| Deep supervision | 3 auxiliary 1×1 heads on decoder blocks 1–3 (128, 64 and 32 channels); the final head is a 3×3 conv on the last decoder block |
| Fusion | Soft mask (sigmoid of the segmentation output), no threshold, no detach |
| Classifier input | Global average pool (stride 32) + mask-weighted pool (stride 32) + mask-weighted pool (stride 16) |
| Head | Linear → 256 → ReLU → Dropout 0.3 → Linear (number of classes) |

In MPF-Net the predicted mask is pooled with the encoder features (mask-weighted average pooling), so the classifier
focuses on the tumor region and gradients from classification flow back into the segmentation branch.

## 3. Training

| Item | Setting |
|------|---------|
| Segmentation loss | Tversky (α = 0.7) + BCE, on the final output and the 3 auxiliary outputs (averaged) |
| Classification loss | Cross-entropy, label smoothing 0.1 (inverse-frequency class weights on D2 only) |
| Loss balance | Learned uncertainty weights (Kendall et al.) |
| Optimizer | AdamW, lr 2e-4, weight decay 1e-4 (the 2 uncertainty-weight parameters use lr 1e-3, no weight decay) |
| Schedule | 1-epoch linear warm-up, then cosine decay |
| EMA | Decay 0.999 (the EMA weights are the ones validated and saved) |
| Epochs | 35 max, early stopping patience 8 on val F1 + Dice |
| Batch size | 16 |
| Precision | Mixed precision (AMP), gradient clipping 1.0 |
| Augmentation | Horizontal flip, rotation ±15°, brightness/contrast, applied jointly to image and mask |

Environment: Kaggle GPU notebook, PyTorch 2.11, `segmentation_models_pytorch` 0.6.0.

Not used: ASPP, test-time augmentation, Grad-CAM.

## 4. Evaluation protocol

1. A stratified 20% hold-out **test set** is split off once (seed 42).
2. The remaining 80% is split with stratified 5-fold cross-validation. A fold's validation part is used
   only for early stopping and checkpoint selection.
3. Each fold model is evaluated on the same hold-out test set. We report the **per-fold mean ± std**
   (comparable to the paper) and the **5-fold ensemble** (softmax and mask probabilities averaged).

Metrics: accuracy and macro-averaged precision, recall and F1. Dice is the mean per-image Dice
at threshold 0.5 over test images with a non-empty ground-truth mask.

## 5. Results

### Summary of all datasets (per-fold mean ± std on the hold-out test set)

Same layout as Table 13 of the paper.

| Dataset | Accuracy | Precision | Recall | F1-score | Dice |
|---|---|---|---|---|---|
| Dataset 1 (ours) | 0.9962 ± 0.0009 | 0.9962 ± 0.0007 | 0.9961 ± 0.0009 | 0.9962 ± 0.0008 | 0.9271 ± 0.0018 |
| Dataset 2 (ours, 5 classes)* | 0.9934 ± 0.0007 | 0.9902 ± 0.0016 | 0.9870 ± 0.0012 | 0.9885 ± 0.0012 | 0.8045 ± 0.0028 |
| Dataset 3 (ours) | 0.9975 ± 0.0008 | 0.9975 ± 0.0008 | 0.9975 ± 0.0008 | 0.9975 ± 0.0008 | 0.8927 ± 0.0050 |
| Dataset 1 (paper) | 0.969 ± 0.004 | 0.960 ± 0.006 | 0.960 ± 0.005 | 0.966 ± 0.005 | 0.929 ± 0.004 |
| Dataset 2 (paper, 4 classes) | 0.994 ± 0.002 | 0.985 ± 0.003 | 0.983 ± 0.003 | 0.984 ± 0.003 | 0.923 ± 0.001 |
| Dataset 3 (paper) | 0.982 ± 0.003 | 0.983 ± 0.003 | 0.982 ± 0.003 | 0.982 ± 0.003 | 0.875 ± 0.004 |

\* Our D2 run has 5 classes and no normal class, so it is **not comparable** with the paper's D2 row (see section 7).
The paper's D2 Dice is also reported as 0.893 in its Table 5, and its D3 Dice as 0.913 in its Table 6.
Our classification scores are higher than the paper's on D1 and D3, most likely helped by the slice-level split
(see Limitations), so they should not be read as a like-for-like improvement. Detailed per-dataset results follow.

### D1 (hold-out test set, 848 images)

| | Accuracy | Precision | Recall | F1 | Dice |
|---|---|---|---|---|---|
| Per-fold mean ± std | 0.9962 ± 0.0009 | 0.9962 ± 0.0007 | 0.9961 ± 0.0009 | 0.9962 ± 0.0008 | 0.9271 ± 0.0018 |
| 5-fold ensemble | 0.9965 | 0.9963 | 0.9967 | 0.9965 | 0.9324 |
| Paper (BRAIN-CNet, Table 13) | 0.969 | 0.960 | 0.960 | 0.966 | 0.929 |

Per-fold results are in `D1_results.json`.

Ensemble confusion matrix on the test set (rows = true, columns = predicted):

| | normal | glioma | meningioma | pituitary |
|---|---|---|---|---|
| normal | 318 | 0 | 1 | 0 |
| glioma | 0 | 130 | 0 | 0 |
| meningioma | 0 | 0 | 200 | 0 |
| pituitary | 0 | 0 | 2 | 197 |

The ensemble makes 3 errors out of 848 images. Our classification scores are above the paper's, most likely
helped by the slice-level split (see Limitations), so they should not be read as a like-for-like improvement.
Dice (0.927 per-fold mean, 0.932 ensemble) is in line with the paper's 0.929.

### D3 (hold-out test set, 1276 images, 5 folds)

| | Accuracy | Precision | Recall | F1 | Dice |
|---|---|---|---|---|---|
| Per-fold mean ± std | 0.9975 ± 0.0008 | 0.9975 ± 0.0008 | 0.9975 ± 0.0008 | 0.9975 ± 0.0008 | 0.8927 ± 0.0050 |
| 5-fold ensemble | 0.9992 | 0.9992 | 0.9992 | 0.9992 | 0.8996 |
| Paper (BRAIN-CNet, Table 13) | 0.982 | 0.983 | 0.982 | 0.982 | 0.875 |

The paper reports D3 Dice as 0.913 in Table 6 and 0.875 in Table 13. Our Dice (0.893 per-fold mean, 0.900 ensemble)
lies between the two.

Ensemble confusion matrix on the test set (rows = true, columns = predicted):

| | normal | glioma | meningioma | pituitary |
|---|---|---|---|---|
| normal | 319 | 0 | 0 | 0 |
| glioma | 0 | 319 | 0 | 0 |
| meningioma | 0 | 0 | 319 | 0 |
| pituitary | 0 | 0 | 1 | 318 |

The ensemble makes 1 error out of 1276 images. Scores this close to perfect most likely reflect the slice-level split
and possible near-duplicate images between train and test (see Limitations), not a true clinical-level accuracy.
Per-fold results are in `D3_results.json`.

### D2 (hold-out test set, 693 images, 5 folds, 5 classes)

| | Accuracy | Precision | Recall | F1 | Dice |
|---|---|---|---|---|---|
| Per-fold mean ± std | 0.9934 ± 0.0007 | 0.9902 ± 0.0016 | 0.9870 ± 0.0012 | 0.9885 ± 0.0012 | 0.8045 ± 0.0028 |
| 5-fold ensemble | 0.9928 | 0.9894 | 0.9858 | 0.9875 | 0.8142 |

**Not comparable with the paper's D2 row** (accuracy 0.994, F1 0.984, Dice 0.923 in Table 13). The Kaggle copy we used
has no normal class, so we trained on the five folders it contains (astrocytoma, glioblastoma, gliomas, meningioma,
neurocytoma). The paper's D2 is a 4-class task that includes 1500 normal images. See section 7.

Ensemble confusion matrix on the test set (rows = true, columns = predicted):

| | astrocytoma | glioblastoma | gliomas | meningioma | neurocytoma |
|---|---|---|---|---|---|
| astrocytoma | 105 | 0 | 0 | 0 | 1 |
| glioblastoma | 0 | 146 | 0 | 0 | 0 |
| gliomas | 0 | 0 | 248 | 0 | 0 |
| meningioma | 0 | 0 | 0 | 128 | 0 |
| neurocytoma | 4 | 0 | 0 | 0 | 61 |

The ensemble makes 5 errors out of 693 images, all between astrocytoma and neurocytoma. Dice is computed only on
images that have a mask (116 neurocytoma images do not), and is lower than on D1 and D3. Per-fold results are in
`D2_results.json`.

**Possible reasons for the lower Dice on D2.** We did not run an ablation to separate these, so they are
plausible contributors, not measured causes:
- **Smaller images.** D2 is trained at 224×224, so small tumors are only a few pixels wide, and Dice drops quickly
  when a small mask is off by a pixel or two. D1 and D3 are trained at 512×512.
- **Mixed image types.** D2 mixes T1, T1CE, T2 and FLAIR slices, so the segmentation branch has more variety to learn.
- **Fewer masks.** 116 neurocytoma images have no mask. They are left out of the Dice score, and the segmentation
  branch learns from fewer images.
- **Inconsistent paper numbers.** The paper itself lists 0.893 (Table 5) and 0.923 (Table 13) for D2 Dice, so a gap
  between our value and a single paper value is not surprising.

### Computational cost

Paper values are from Table 14 of the paper (single forward pass, input 3×224×224, batch size 1, NVIDIA A2000).
Our values were measured with `measure_model_cost.ipynb` (random weights, same architecture).

| Model | Parameters | FLOPs (3×224×224) | Inference time |
|---|---|---|---|
| CNN (paper) | ≈ 2.5 M | ≈ 1.1 G | 2–5 ms |
| VGG16 (paper) | ≈ 138 M | ≈ 15.5 G | 15–20 ms |
| ResNet50 (paper) | ≈ 25.6 M | ≈ 3.9 G | 8–12 ms |
| Post-Hoc Seg-Class (paper) | ≈ 15.23 M | ≈ 5.95 G | 20 ms |
| U-Net (paper) | ≈ 31.04 M | ≈ 386.36 G | 1–3 ms |
| BRAIN-CNet (paper) | ≈ 1.68 M | ≈ 1.8 G | 3–5 ms |
| **MPF-Net (ours)** | **6.08 M** (6,080,362 with 4 classes; 6,080,619 with 5) | **4.42 G** (= 2.21 GMACs) | **12.10 ms** (fp32, GPU: NVIDIA T4 (Kaggle)) |

MPF-Net parameter breakdown: encoder 3.60 M, decoder 2.29 M, segmentation and auxiliary heads 0.0004 M, classifier head 0.19 M.
At the training resolution of D1 and D3 (3×512×512): **23.10 G FLOPs** (11.55 GMACs) and **20.54 ms** per image (fp32, batch 1).
FLOPs are counted as 2 × multiply-accumulates (PyTorch's flop counter); the paper does not state its convention, so both are shown.
With fp16 autocast the times were 18.19 ms (224×224) and 18.14 ms (512×512). At batch size 1 this model is limited by
kernel-launch overhead rather than arithmetic, so fp16 gives no speed-up, and the 224×224 and 512×512 times are close.

MPF-Net is larger and slower than the paper's reported BRAIN-CNet (6.08 M vs 1.68 M parameters, 4.42 G vs 1.8 G FLOPs,
12.10 ms vs 3–5 ms). It has fewer parameters than ResNet50 (25.6 M) but about the same FLOPs. The paper's 1.68 M parameter count is smaller than an
EfficientNet-B0 encoder alone (about 3.6 M parameters without its ImageNet classifier), so the paper's number may use a
different definition; we report ours exactly as measured. Inference times depend on the GPU and are only roughly comparable.

## 6. How to run

1. On Kaggle: **Create → New Notebook → File → Import Notebook** and upload the `.ipynb`.
2. **Add Input** and attach the dataset(s) listed above.
3. Set the session to **GPU** and **Internet: On** (needed for pip and the pretrained weights).
4. In the config cell set `DATASET = "D1"` (or `"D2"` / `"D3"`) and `SMOKE = False`.
   Use `SMOKE = True` first for a quick 2-epoch check.
5. **Save Version → Save & Run All (Commit)**. Full D1 training takes roughly 5–7 hours on a Kaggle GPU.
6. Outputs (in `/kaggle/working`): `<DATASET>_results.json` and `<DATASET>_fold0.pt … <DATASET>_fold4.pt`
   (for example `D1_results.json`, `D3_results.json`).

If class folders are not recognized, edit `CLASS_OVERRIDE` in the config cell.
For D1 it maps folders `0/1/2/3` to normal / glioma / meningioma / pituitary.

## 7. Differences from the paper

| Aspect | Paper | This work |
|--------|-------|-----------|
| Mask fusion | Binary mask concatenated to the image as an extra channel | Soft mask weights encoder features (stride 32 and 16) |
| Seg loss | Dice + Tversky | Tversky + BCE |
| Loss weighting | Fixed λ = 0.2 | Learned uncertainty weights |
| LR schedule | ReduceLROnPlateau | Warm-up + cosine, with EMA |
| Preprocessing | Gaussian blur + tight crop | None |
| Split | Repeated random 80/20 | Stratified hold-out 20% + 5-fold CV, plus ensemble |

### D2 differences

| Aspect | Paper | This work |
|--------|-------|-----------|
| Classes | neurocytoma, glioma (astrocytoma + glioblastoma), meningioma, normal (4 classes) | astrocytoma, glioblastoma, gliomas, meningioma, neurocytoma (5 classes, as in the folders) |
| Images | 3670 (normal 1500, glioma 1261, meningioma 640, neurocytoma 269) | 3465 (glioma folders 529 + 732 + 1237, meningioma 640, neurocytoma 327) |
| Normal class | present | not present in the Kaggle copy used |
| Image size | 224×224 | 224×224 |

Astrocytoma + glioblastoma add up to the paper's 1261 glioma images and meningioma matches (640). Neurocytoma (327 vs 269),
the extra "Gliomas" folder (1237) and the missing normal class do not match the paper, and we could not explain why.

## 8. Limitations

- Slices are split at random, not by patient, because patient IDs are not available. Neighboring slices of
  the same scan can end up in both train and test, which likely inflates accuracy. The paper notes the same limitation.
- D3 is a mixed dataset, and its near-perfect scores (99.9% ensemble accuracy) suggest that near-duplicate or
  highly similar images appear in both train and test. We did not check for duplicates.
- On D2, the "Gliomas" folder may overlap with the astrocytoma and glioblastoma folders (all are glioma types),
  which could add leakage on top of the slice-level split. We did not check for overlap.
- 2D slices only, with no 3D context, and public retrospective data only.
- Dice and F1 definitions are not specified in the paper. The ones above are our choice, so small differences
  from the paper's numbers are expected.
- The paper reports slightly different values for the same dataset in different places (for example D3 Dice
  0.913 in Table 6 vs 0.875 in Table 13). We compare against Table 13.

## 9. Files

| File | Description |
|------|-------------|
| `mpfnet_brain_tumor_kaggle.ipynb` | Full pipeline: data loading, model, training, evaluation |
| `D1_results.json`, `D2_results.json`, `D3_results.json` | Per-fold metrics, mean ± std, ensemble metrics, hyperparameters |
| `D1_fold{0-4}.pt`, `D2_fold{0-4}.pt`, `D3_fold{0-4}.pt` | Best EMA weights of each fold |
| `README.md` | This file |

## 10. Reference

Musthafa N., Memon Q., Masud M. *A Computationally Efficient and Improved Brain Tumor Recognition System
by MRI–Segmentation Integrated Classification Network.* Journal of Imaging Informatics in Medicine, 2026.
https://doi.org/10.1007/s10278-026-02063-2
