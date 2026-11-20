# Research Gaps and Missing Information
## MPF-Net: Brain Tumor Classification and Segmentation
### For review before journal/conference submission

---

## 1. Author and Institutional Information (REQUIRED BEFORE SUBMISSION)

| Field | Status |
|---|---|
| Full institutional name (university, department) | ❌ MISSING — placeholder used |
| City and country of institution | ❌ MISSING |
| Author email addresses | ❌ MISSING — placeholder used |
| ORCID iDs for all authors | ❌ MISSING |
| Funding / grant acknowledgements | ❌ MISSING — placeholder used |

**Action required:** Fill in `\IEEEauthorblockA` and `\IEEEauthorblockN` fields in `main.tex` before submission.

---

## 2. Dataset Discrepancies

### 2.1 D2 (Kaggle — Shubham Gajjar): 5 Classes, Not 4
- **Issue:** The Kaggle `shubhamgajjar/brain-tumor-classification-2d` archive contains **5 classes** (Astrocytoma, Ependymoma, Glioma, Meningioma, Neurocytoma) with **no normal class**.
- **BRAIN-CNet base paper** reports D2 results for a **4-class** setup. Direct numeric comparison is **invalid**.
- **Paper handling:** Table IV includes a footnote explicitly disclosing this. The D2 MPF-Net result column is left without a BRAIN-CNet comparison value.
- **Action required:** Confirm whether you obtained the 4-class version separately. If yes, re-run experiments and add comparable results.

### 2.2 D2: Missing Segmentation Masks for Neurocytoma
- **Issue:** 116 neurocytoma images in D2 lack segmentation masks.
- **Paper handling:** Footnote in Table I. Segmentation Dice for D2 is computed only on the masked subset.
- **Action required:** Confirm exact number of masked images used in segmentation evaluation (116 excluded → 577 test images for segmentation).

### 2.3 D3 (Mixed Dataset): Near-Duplicate Risk
- **Issue:** README warns that D3 may contain near-duplicate slices from multiple sources. No formal deduplication audit was performed.
- **D3 ensemble results (Acc=0.9992, F1=0.9992)** are suspiciously high and may be inflated by duplicates.
- **Paper handling:** Flagged in Discussion and Limitations sections.
- **Action required:** Run perceptual hashing (e.g., `imagehash`) or structural similarity audit on D3 before claiming these results in a peer-reviewed journal.

---

## 3. Patient-Level vs Slice-Level Split — Critical Leakage Issue

### 3.1 Figshare Exp 3 (Slice-Level) vs Exp 2/2b (Patient-Level)
- **Exp 2 / Exp 2b:** Proper patient-level splits using 46 patients. No data leakage. These are the **scientifically valid** results.
- **Exp 3:** Slice-level random split with 154 patients overlapping between train and test sets. This constitutes **data leakage**.
- **Quantified inflation:** Accuracy +1.58%, Dice +6.39% compared to honest patient-level evaluation.
- **Paper handling:** Exp 3 results are included for reproducibility but clearly labelled as slice-level (inflated). Patient-level results (Exp 2/2b) are primary.
- **Action required:** Do not use Exp 3 metrics as primary results in comparisons against Sahoo et al. (who may also use slice-level splits — confirm from their paper).

### 3.2 Sahoo et al. Dice (0.9011) vs Our Patient-Level Dice (0.7751)
- **Issue:** Sahoo et al. report Dice = 0.9011 on Figshare, which exceeds our patient-level Dice of 0.7751.
- **Unknown:** Whether Sahoo et al. used patient-level or slice-level splits.
- **Paper handling:** Discussed in Section VIII. Comparison is made with explicit split-type disclosure.
- **Action required:** Retrieve Sahoo et al. experimental protocol and confirm split type. If they used slice-level, our Exp 3 (Dice=0.8390) is the appropriate comparison.

---

## 4. Missing Experimental Data

### 4.1 Notebook Execution Outputs
- **Exp 1 (`figshare_exp1_baseline_256.ipynb`):** Has executed cell outputs (1.82 MB). ✅
- **Exp 2b (`figshare_exp2b_512_aspp_fpn.ipynb`):** No executed outputs. ⚠️
- **Exp 3 (`figshare_exp3_slice_level.ipynb`):** No executed outputs. ⚠️
- **Impact:** Cannot verify training curves, epoch-by-epoch metrics, or intermediate losses for Exp 2b/3.

### 4.2 ROC Curve Data
- **Issue:** No ROC curve CSV/JSON files found in experiment archives. AUC values exist for Exp 2 (per-class: 0.9989, 0.9997, 1.0000), but raw TPR/FPR arrays are unavailable.
- **Figure status:** `fig6_roc_curves.pdf` cannot be generated from actual data. The figure generation script uses the AUC scalar values only.
- **Action required:** Re-run Exp 2 with ROC curve export enabled, or provide the raw score files from your Kaggle/Colab session.

### 4.3 Training History for Exp 2b and Exp 3
- **Issue:** `training_history.csv` exists in the Exp 1 zip archive. No equivalent file confirmed for Exp 2b or Exp 3.
- **Impact:** Training curves figure (`fig4`) uses Exp 1 data only.
- **Action required:** Export `training_history.csv` from your Exp 2b and Exp 3 Kaggle sessions if available.

### 4.4 Qualitative Segmentation Examples
- **Issue:** No actual MRI image files or predicted mask files are stored in the repository. `fig8_segmentation_examples` cannot be generated.
- **Action required:** Save 3–5 test image+mask+prediction triplets (one per tumour class) from your experiments and add to `IEEE_Research_Paper/figures/` as PNG files.

### 4.5 Hardware Environment for Figshare Experiments
- **Known:** RTX 4050 Laptop GPU for local benchmarks. CUDA 12.6, PyTorch 2.12.1.
- **Unknown:** GPU type used in Kaggle notebooks for Exp 1/2/2b/3 (T4 or P100?).
- **Action required:** Add Kaggle session GPU type to paper's Implementation Details section.

---

## 5. Architectural Notes and Ambiguities

### 5.1 "FinalNet" vs "MPF-Net" Class Name
- **Issue:** Figshare Exp 1 and Exp 2 use a class named `FinalNet`; Exp 3 and Kaggle experiments use `MPFNet`. Both are architecturally identical (same code, different class name).
- **Paper handling:** The paper consistently uses "MPF-Net" as the model name.
- **Action required:** Rename `FinalNet` → `MPFNet` in source code and re-push to GitHub for consistency.

### 5.2 ASPP and FPN Availability
- **Baseline (Exp 1/2):** No ASPP, no FPN. Uses plain EfficientNet-B0 encoder + U-Net decoder + scSE + MPF fusion.
- **Extended (Exp 2b/3 and Kaggle):** Adds ASPP on stride-32 bottleneck and FPN lateral connections.
- **Paper handling:** Presents baseline (without ASPP/FPN) and full model (with ASPP+FPN) separately. Kaggle results use the full model.

### 5.3 No Formal Ablation Table
- **Issue:** No controlled ablation experiment was run isolating individual components (ASPP alone, FPN alone, MPF alone, Kendall weighting alone).
- **Available:** Exp 2 (baseline) vs Exp 2b (baseline + ASPP + FPN) on the same patient-level split — this is a partial ablation for ASPP+FPN combined.
- **Paper handling:** Presented as "Component Comparison" rather than a full ablation study. No ablation section is claimed.

---

## 6. Missing Comparisons

### 6.1 No Comparison Against nnU-Net or TransUNet on Figshare
- These are strong baselines for medical image segmentation. No such experiments were run.
- **Paper handling:** Discussed in Related Work and Limitations.

### 6.2 No Multi-Modal Experiments
- Only T1-weighted MRI slices used. FLAIR, T2, DWI not explored.
- **Paper handling:** Listed as future work.

---

## 7. Statistical Validity

### 7.1 No Statistical Significance Testing
- Results are reported as ensemble and 5-fold mean ± SD. No t-test, Wilcoxon, or McNemar test performed.
- **Paper handling:** Results presented descriptively; no "significantly outperforms" language used.

### 7.2 Small Figshare Test Set
- Patient-level split: only 46 patients, 629 test slices. Per-class counts: Meningioma=142, Glioma=296, Pituitary=191.
- Confidence intervals are not reported.
- **Action required:** Consider bootstrapped 95% CI for the Figshare results before journal submission.

---

## 8. References

### 8.1 References Requiring DOI Verification
All `references.bib` entries were constructed from known publication records. The following should be double-checked before submission:

| Citation key | Concern |
|---|---|
| `musthafa2026braincnet` | DOI `10.1007/s10278-026-02063-2` — verify at Springer link |
| `sahoo2023brain` | DOI `10.1016/j.bbe.2023.08.003` — verify at Elsevier |
| `wong2025vgg` | PLOS ONE `pone.0322624` — verify publication year (2025) |
| `cheng2015enhanced` | Figshare dataset paper — verify PLOS ONE volume/page |

### 8.2 Uncited References in BibTeX
`references.bib` contains entries for `ioffe2015batchnorm`, `polyak1992ema`, `micikevicius2018amp` and `menze2015brats` which may not be cited in `main.tex`. Remove any unused entries before final submission to keep the bibliography clean.

---

## 9. Submission Checklist

- [ ] Fill in author affiliations and emails
- [ ] Confirm D2 class count discrepancy with original dataset provider
- [ ] Audit D3 for near-duplicates before claiming Acc=0.9992
- [ ] Confirm Sahoo et al. split type (patient vs slice)
- [ ] Export ROC curve data from Kaggle/Colab sessions
- [ ] Export training history CSVs for Exp 2b and Exp 3
- [ ] Prepare 3–5 qualitative segmentation examples (MRI + mask + prediction)
- [ ] Rename `FinalNet` → `MPFNet` in source code, push to GitHub
- [ ] Verify all DOIs in `references.bib`
- [ ] Remove unused `references.bib` entries
- [ ] Run `pdflatex` → `bibtex` → `pdflatex` × 2 and confirm zero errors
- [ ] Check that all `\ref{}` labels match table/figure `\label{}` declarations
- [ ] Confirm GPU type used in Kaggle experiments (T4 or P100)
- [ ] (Optional) Compute bootstrapped 95% CI for Figshare patient-level metrics
