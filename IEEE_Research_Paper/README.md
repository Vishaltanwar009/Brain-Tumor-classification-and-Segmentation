# IEEE_Research_Paper — Compilation Guide

## Project: MPF-Net Brain Tumor Classification and Segmentation

**Authors:** Vishal Tanwar (2024UCA1809), Kumar Sahil (2024UCA1817)

---

## Folder Structure

```
IEEE_Research_Paper/
├── main.tex                          ← Main LaTeX paper (IEEEtran journal)
├── references.bib                    ← BibTeX bibliography (verified)
├── figures/
│   ├── generate_figures.py           ← Run this to create all PDF figures
│   ├── fig1_architecture.pdf         ← MPF-Net overall architecture
│   ├── fig2_mpf_fusion.pdf           ← Mask-Pooled Fusion detail
│   ├── fig3_dataset_distribution.pdf ← Class distribution bar charts
│   ├── fig4_training_curves.pdf      ← Loss/metric training curves
│   ├── fig5_confusion_matrix.pdf     ← Figshare Exp 2 confusion matrix
│   ├── fig6_roc_curves.pdf           ← ROC curves (AUC scalars only)
│   └── fig7_per_class_dice.pdf       ← Per-class Dice comparison
└── tables/
    ├── tab1_dataset_summary.tex
    ├── tab2_figshare_classification.tex
    ├── tab3_figshare_segmentation.tex
    ├── tab4_kaggle_results.tex
    ├── tab5_complexity.tex
    └── tab6_perclass_figshare.tex
```

> **Note:** `fig8_segmentation_examples.pdf` is NOT auto-generated.  
> You must manually export MRI+mask+prediction image triplets and place them  
> in `figures/` as PNG files, then uncomment the figure block in `main.tex`.

---

## Step 1 — Generate Figures

> Requires Python 3.8+ with `matplotlib`, `numpy`, `seaborn`.

```powershell
# Install dependencies (if needed)
pip install matplotlib numpy seaborn

# Generate all figures
cd "c:\Users\bisut\Downloads\brain tumor classification and segmentation\IEEE_Research_Paper\figures"
python generate_figures.py
```

All figures are saved as **PDF** (vector, publication-quality) in the `figures/` folder.

---

## Step 2A — Compile on Overleaf (Recommended)

1. Zip the entire `IEEE_Research_Paper/` folder.
2. Go to [https://www.overleaf.com](https://www.overleaf.com) → **New Project** → **Upload Project**.
3. Upload the ZIP file.
4. Set the **main document** to `main.tex`.
5. Set the **compiler** to `pdfLaTeX`.
6. Click **Recompile**.

Overleaf handles bibliography automatically via `natbib` / `bibtex`.

---

## Step 2B — Compile Locally (pdfLaTeX + BibTeX)

> Requires a full TeX distribution: [MiKTeX](https://miktex.org/) (Windows) or [TeX Live](https://tug.org/texlive/).

```powershell
cd "c:\Users\bisut\Downloads\brain tumor classification and segmentation\IEEE_Research_Paper"

pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Run `pdflatex` three times total so cross-references and citations resolve correctly.

### Using latexmk (easier)

```powershell
latexmk -pdf main.tex
```

---

## Step 3 — Before Submission

Fill in the placeholders in `main.tex`:

| Placeholder | Replace with |
|---|---|
| `[INSTITUTION PLACEHOLDER]` | Your university name, department |
| `[CITY, COUNTRY]` | City and country |
| `[EMAIL1@PLACEHOLDER.EDU]` | Vishal Tanwar's email |
| `[EMAIL2@PLACEHOLDER.EDU]` | Kumar Sahil's email |
| `[FUNDING PLACEHOLDER]` | Grant/funding details, or remove line |

See `research_gaps_and_missing_information.md` for the full pre-submission checklist.

---

## Common Errors

| Error | Fix |
|---|---|
| `! LaTeX Error: File 'IEEEtran.cls' not found.` | Install the `IEEEtran` package via MiKTeX Manager, or use Overleaf (has it built-in) |
| `! LaTeX Error: File 'figures/fig1_architecture.pdf' not found.` | Run `generate_figures.py` first |
| Bibliography not appearing | Run `bibtex main` between the two `pdflatex` calls |
| Overfull `\hbox` warnings | Normal for IEEE two-column format; minor and do not affect compilation |

---

## GitHub Push

After generating all files:

```powershell
cd "c:\Users\bisut\Downloads\brain tumor classification and segmentation"
git add IEEE_Research_Paper/
git commit -m "Add complete IEEE LaTeX research paper (MPF-Net)"
git push origin main
```
