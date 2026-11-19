# FinalNet: Joint Brain Tumor Segmentation and Classification from MRI

A single end-to-end network that segments the tumor and classifies the MRI slice
(normal / glioma / meningioma / pituitary). The experiments follow the datasets and evaluation
setup of *"A Computationally Efficient and Improved Brain Tumor Recognition System by
MRI–Segmentation Integrated Classification Network"* (BRAIN-CNet, J. Imaging Inform. Med., 2026).

> Fill in the fields marked **[TO FILL]** before submitting.

- **Author(s):** [TO FILL]
- **Notebook:** `finalnet_brain_tumor_kaggle.ipynb` (runs on Kaggle, GPU + Internet on)

---

## 1. Datasets

| ID | Kaggle dataset | Images | Classes | Masks |
|----|----------------|--------|---------|-------|
| D1 | `atikaakter11/brain-tumor-segmentation-dataset` | 4237 | normal 1595, glioma 649, meningioma 999, pituitary 994 | yes |
| D2 | `shubhamgajjar/brain-tumor-classification-2d` | 3670 | normal 1500, glioma 1261 (astrocytoma + glioblastoma merged), meningioma 640, neurocytoma 269 | **[TO FILL: found / not found]** |
| D3 | `wajeehaldeenaljunaid/mixed-brain-tumor-dataset` | 6380 | 1595 per class (normal, glioma, meningioma, pituitary) | yes |

Datasets run: **[TO FILL: D1 / D2 / D3]**

Preprocessing: grayscale slice resized to 512×512 (D1, D3) or 224×224 (D2), replicated to 3 channels
and normalized with ImageNet statistics. Masks are binarized and resized with nearest-neighbor.
Healthy slices get an empty mask. No blur and no cropping.

## 2. Model

| Part | Setting |
|------|---------|
| Encoder | EfficientNet-B0, ImageNet pretrained (`tu-efficientnet_b0` via `segmentation_models_pytorch`) |
| Decoder | U-Net, widths (256, 128, 64, 32, 16), scSE attention |
| Deep supervision | 3 auxiliary 1×1 heads on decoder blocks 1–3 |
| Fusion | Soft mask (sigmoid of the segmentation output), no threshold, no detach |
| Classifier input | Global average pool (stride 32) + mask-weighted pool (stride 32) + mask-weighted pool (stride 16) |
| Head | Linear → 256 → ReLU → Dropout 0.3 → Linear (number of classes) |

The predicted mask weights the encoder features before classification, so the classifier
focuses on the tumor region and gradients from classification flow back into the segmentation branch.

## 3. Training

| Item | Setting |
|------|---------|
| Segmentation loss | Tversky (α = 0.7) + BCE, on the final output and the 3 auxiliary outputs (averaged) |
| Classification loss | Cross-entropy, label smoothing 0.1 (class weights on D2) |
| Loss balance | Learned uncertainty weights (Kendall et al.) |
| Optimizer | AdamW, lr 2e-4, weight decay 1e-4 |
| Schedule | 1-epoch linear warm-up, then cosine decay |
| EMA | Decay 0.999 (the EMA weights are the ones validated and saved) |
| Epochs | 35 max, early stopping patience 8 on val F1 + Dice |
| Batch size | 16 (**[TO FILL: 8 if you reduced it]**) |
| Precision | Mixed precision (AMP), gradient clipping 1.0 |
| Augmentation | Horizontal flip, rotation ±15°, brightness/contrast, applied jointly to image and mask |

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

Hold-out test set, D1 (848 images). **[TO FILL from the final cell output]**

| | Accuracy | Precision | Recall | F1 | Dice |
|---|---|---|---|---|---|
| Per-fold mean ± std | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| 5-fold ensemble | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] | [TO FILL] |
| Paper (BRAIN-CNet, Table 13) | 0.969 | 0.960 | 0.960 | 0.966 | 0.929 |

Per-fold results are in `D1_results.json`. Fold 0 (observed during training): accuracy 0.9953,
F1 0.9955, Dice 0.9264.

Add the same table for D2 / D3 if you ran them.

## 6. How to run

1. On Kaggle: **Create → New Notebook → File → Import Notebook** and upload the `.ipynb`.
2. **Add Input** and attach the dataset(s) listed above.
3. Set the session to **GPU** and **Internet: On** (needed for pip and the pretrained weights).
4. In the config cell set `DATASET = "D1"` (or `"D2"` / `"D3"`) and `SMOKE = False`.
   Use `SMOKE = True` first for a quick 2-epoch check.
5. **Save Version → Save & Run All (Commit)**. Full D1 training takes roughly 5–7 hours on a Kaggle GPU.
6. Outputs (in `/kaggle/working`): `D1_results.json` and `D1_fold0.pt … D1_fold4.pt`.

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

## 8. Limitations

- Slices are split at random, not by patient, because patient IDs are not available. Neighboring slices of
  the same scan can end up in both train and test, which likely inflates accuracy. The paper notes the same limitation.
- 2D slices only, with no 3D context, and public retrospective data only.
- Dice and F1 definitions are not specified in the paper. The ones above are our choice, so small differences
  from the paper's numbers are expected.
- The paper reports slightly different values for the same dataset in different places (for example D3 Dice
  0.913 in Table 6 vs 0.875 in Table 13). We compare against Table 13.

## 9. Files

| File | Description |
|------|-------------|
| `finalnet_brain_tumor_kaggle.ipynb` | Full pipeline: data loading, model, training, evaluation |
| `D1_results.json` | Per-fold metrics, mean ± std, ensemble metrics, hyperparameters |
| `D1_fold{0-4}.pt` | Best EMA weights of each fold |
| `README.md` | This file |

## 10. Reference

Musthafa N., Memon Q., Masud M. *A Computationally Efficient and Improved Brain Tumor Recognition System
by MRI–Segmentation Integrated Classification Network.* Journal of Imaging Informatics in Medicine, 2026.
https://doi.org/10.1007/s10278-026-02063-2
