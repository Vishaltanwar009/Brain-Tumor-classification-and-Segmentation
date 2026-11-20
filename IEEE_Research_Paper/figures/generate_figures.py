"""
generate_figures.py — High-Quality IEEE Publication Figures (Base Paper Style Matching)
========================================================================================
Implements exact visual design from user reference:
  - Fig 1: Elegant EfficientNet-B0 dashed container with top oval badge,
           spacious horizontal encoder, ASPP + FPN modules, U-Net decoder with
           orange dashed container and bottom badge, MPF module, dual heads, Kendall loss.
  - Fig 2: Crystal clear parallel lanes matching reference: Encoder Stage 4 (blue),
           Encoder Stage 3 (gold), Segmentation Branch (green), 3 pooling blocks,
           Feature Fusion (purple), Classifier (green), and bottom mathematical pill.
  - Fig 3: Updated with exact user D2 class counts:
           Gliomas (1237), Glioblastoma (732), Meningioma (640), Astrocytoma (529), Neurocytoma (211).
           Total matched pairs: 3349.
  - Fig 8: Generates 3 separate large, high-resolution figures:
           fig8a_exp1_segmentations.pdf, fig8b_exp2_segmentations.pdf, fig8c_exp3_segmentations.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
import csv
import zipfile as zf

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(OUT, "..", ".."))
DPI = 300

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  [OK] Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Overall Architecture (Matching User Reference Image Perfectly)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    ax.set_xlim(0, 18.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x, y, w, h, t1, t2="", t3="", color="#EBF5FB", edge="#2980B9", lw=1.2, r=0.18):
        rect = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={r}",
                              lw=lw, edgecolor=edge, facecolor=color, zorder=4)
        ax.add_patch(rect)
        cy = y + h / 2.0
        if t2 and t3:
            ax.text(x + w/2.0, cy + 0.28, t1, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1B2631", zorder=5)
            ax.text(x + w/2.0, cy, t2, ha="center", va="center", fontsize=7.2, color="#2C3E50", zorder=5)
            ax.text(x + w/2.0, cy - 0.26, t3, ha="center", va="center", fontsize=6.8, color="#566573", zorder=5)
        elif t2:
            ax.text(x + w/2.0, cy + 0.16, t1, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1B2631", zorder=5)
            ax.text(x + w/2.0, cy - 0.18, t2, ha="center", va="center", fontsize=7.2, color="#2C3E50", zorder=5)
        else:
            ax.text(x + w/2.0, cy, t1, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1B2631", zorder=5)

    def arrow(x1, y1, x2, y2, color="#1B2631", lw=1.3, rad=0.0):
        con = f"arc3,rad={rad}" if rad != 0.0 else "arc3"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                   connectionstyle=con, shrinkA=1, shrinkB=1), zorder=6)

    # 1. Left Input: MRI Input
    box(0.4, 3.2, 1.4, 1.6, "MRI Input", "3 × H × W", "224 / 512²", color="#E8F8F5", edge="#27AE60", lw=1.3)
    arrow(2.0, 4.0, 2.45, 4.0)

    # 2. Encoder dashed container
    enc_container = FancyBboxPatch((2.4, 2.7), 8.0, 2.65, boxstyle="round,pad=0.2",
                                   lw=1.3, linestyle="--", edgecolor="#3498DB", facecolor="none", zorder=2)
    ax.add_patch(enc_container)
    # Oval badge on top of encoder container
    badge = FancyBboxPatch((3.7, 5.3), 5.4, 0.48, boxstyle="round,pad=0.15",
                           lw=1.2, edgecolor="#2980B9", facecolor="#EBF5FB", zorder=5)
    ax.add_patch(badge)
    ax.text(6.4, 5.54, "EfficientNet-B0 Backbone (ImageNet Pretrained)", ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#1B4F72", style="italic", zorder=6)

    # Encoder stages E0 to E4
    enc_x = [2.6, 4.2, 5.8, 7.4, 9.0]
    stages = [
        ("E0", "16 ch", "stride 2"),
        ("E1", "24 ch", "stride 4"),
        ("E2", "40 ch", "stride 8"),
        ("E3", "112 ch", "stride 16"),
        ("E4", "320 ch", "stride 32"),
    ]
    for i, (name, ch, st) in enumerate(stages):
        box(enc_x[i], 3.2, 1.25, 1.6, name, ch, st, color="#EBF5FB", edge="#2980B9", lw=1.2)
        if i < 4:
            arrow(enc_x[i] + 1.4, 4.0, enc_x[i+1] - 0.15, 4.0)

    # 3. ASPP and FPN modules
    arrow(10.45, 4.0, 11.0, 4.0)
    box(11.0, 3.4, 2.2, 1.4, "ASPP", "Dilated Conv", "r ∈ {1, 4, 6}", color="#F4ECF7", edge="#8E44AD", lw=1.3)
    box(11.0, 1.5, 2.2, 1.4, "FPN", "Lateral Fusion", "(Adapter γ=0)", color="#F4ECF7", edge="#8E44AD", lw=1.3)
    arrow(12.1, 3.25, 12.1, 3.05, color="#8E44AD", lw=1.2)

    # 4. MPF Module (Top)
    box(10.8, 5.5, 2.6, 1.35, "MPF Module", "wpool(fb, M) + wpool(fs, M)", "Differentiable Soft Mask",
        color="#FDEDEC", edge="#C0392B", lw=1.3)
    # Red curved arrow from E4 to MPF Module
    arrow(9.8, 4.9, 10.7, 6.1, color="#C0392B", lw=1.3, rad=-0.2)

    # 5. Decoder dashed container
    dec_container = FancyBboxPatch((13.8, 1.2), 1.9, 5.6, boxstyle="round,pad=0.2",
                                   lw=1.3, linestyle="--", edgecolor="#F39C12", facecolor="none", zorder=2)
    ax.add_patch(dec_container)
    # Bottom badge for decoder
    dec_badge = FancyBboxPatch((13.6, 0.48), 2.3, 0.45, boxstyle="round,pad=0.15",
                               lw=1.1, edgecolor="#D68910", facecolor="#FEF9E7", zorder=5)
    ax.add_patch(dec_badge)
    ax.text(14.75, 0.70, "U-Net Decoder + scSE Blocks", ha="center", va="center",
            fontsize=7.2, fontweight="bold", color="#7D6608", style="italic", zorder=6)

    # Decoder stages D0 to D4
    dec_y = [5.6, 4.5, 3.4, 2.3, 1.2]
    d_stages = [("D0", "256 ch"), ("D1", "128 ch"), ("D2", "64 ch"), ("D3", "32 ch"), ("D4", "16 ch")]
    for j, (dname, dch) in enumerate(d_stages):
        box(14.0, dec_y[j], 1.5, 0.85, dname, dch, color="#FEF9E7", edge="#F39C12", lw=1.1)
        if j < 4:
            arrow(14.75, dec_y[j] - 0.08, 14.75, dec_y[j+1] + 0.95, color="#D68910", lw=1.0)

    # Arrow from ASPP to D0
    arrow(13.35, 4.1, 13.9, 5.95, color="#8E44AD", lw=1.2, rad=-0.15)
    # Arrow from FPN to D4
    arrow(13.35, 2.2, 13.9, 1.62, color="#8E44AD", lw=1.2)

    # 6. Heads & Loss (Rightmost Column)
    box(16.4, 5.6, 1.8, 1.25, "Cls Head", "Linear 752 → 256", "Linear 256 → C", color="#EBF5FB", edge="#2980B9", lw=1.3)
    arrow(13.55, 6.18, 16.3, 6.18, color="#2980B9", lw=1.3)
    arrow(15.65, 6.02, 16.3, 6.02, color="#2980B9", lw=1.1)

    box(16.4, 3.4, 1.8, 1.5, "Kendall Loss", "Lcls + Lseg + Lreg", "(uncertainty Wt.)", color="#E8F8F5", edge="#27AE60", lw=1.3)
    arrow(17.3, 5.45, 17.3, 5.05, color="#1B2631", lw=1.1)

    box(16.4, 1.2, 1.8, 1.1, "Seg Head", "σ(Conv 1×1)", color="#E8F8F5", edge="#27AE60", lw=1.3)
    arrow(15.65, 1.62, 16.3, 1.62, color="#27AE60", lw=1.2)
    arrow(17.3, 2.45, 17.3, 3.25, color="#1B2631", lw=1.1)

    # Deep supervision connection from D1/D2 to Kendall Loss
    arrow(15.65, 4.8, 16.35, 4.3, color="#27AE60", lw=1.1, rad=0.1)

    fig.tight_layout()
    savefig(fig, "fig1_architecture.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — MPF Fusion Mechanism (Matching User Reference Image Perfectly)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_mpf_fusion():
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    def box(x, y, w, h, t1, t2="", t3="", color="#EBF5FB", edge="#2980B9", lw=1.2, r=0.18):
        rect = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={r}",
                              lw=lw, edgecolor=edge, facecolor=color, zorder=4)
        ax.add_patch(rect)
        cy = y + h / 2.0
        if t2 and t3:
            ax.text(x + w/2.0, cy + 0.28, t1, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1B2631", zorder=5)
            ax.text(x + w/2.0, cy, t2, ha="center", va="center", fontsize=7.2, color="#2C3E50", zorder=5)
            ax.text(x + w/2.0, cy - 0.28, t3, ha="center", va="center", fontsize=6.8, color="#566573", zorder=5)
        elif t2:
            ax.text(x + w/2.0, cy + 0.16, t1, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1B2631", zorder=5)
            ax.text(x + w/2.0, cy - 0.18, t2, ha="center", va="center", fontsize=7.2, color="#2C3E50", zorder=5)
        else:
            ax.text(x + w/2.0, cy, t1, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#1B2631", zorder=5)

    def arrow(x1, y1, x2, y2, color="#1B2631", lw=1.3, rad=0.0):
        con = f"arc3,rad={rad}" if rad != 0.0 else "arc3"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                   connectionstyle=con, shrinkA=1, shrinkB=1), zorder=6)

    # Left: Encoder Stage 4 (Blue) and Encoder Stage 3 (Gold)
    box(0.5, 4.6, 2.7, 1.5, "Encoder Stage 4", "fb ∈ ℝ^{C × H/32 × W/32}", "Stride: 32", color="#D4E6F1", edge="#2980B9", lw=1.3)
    box(0.5, 1.8, 2.7, 1.5, "Encoder Stage 3", "fs ∈ ℝ^{C × H/16 × W/16}", "Stride: 16", color="#FCF3CF", edge="#F39C12", lw=1.3)

    # Center Top: Segmentation Branch (Green)
    box(3.8, 5.8, 2.8, 1.25, "Segmentation Branch", "Soft Mask M ∈ [0, 1]^{H × W} ⊗", "σ(Conv 1×1)", color="#D5F5E3", edge="#27AE60", lw=1.3)

    # 3 Middle Pooling Blocks (Cleanly stacked, NO overlapping arrows)
    box(7.5, 5.0, 2.5, 1.3, "Mask-Weighted Pool", "wpool(fb, M)", "Output: 320 ch", color="#FDEDEC", edge="#E74C3C", lw=1.3)
    box(7.5, 3.4, 2.5, 1.2, "Global Avg Pool", "GAP(fb)", "Output: 320 ch", color="#EBF5FB", edge="#3498DB", lw=1.3)
    box(7.5, 1.8, 2.5, 1.3, "Mask-Weighted Pool", "wpool(fs, M)", "Output: 112 ch", color="#FDEDEC", edge="#E74C3C", lw=1.3)

    # Arrows from Encoder Stage 4:
    arrow(3.35, 5.4, 7.35, 5.65, color="#2980B9", lw=1.2)
    arrow(3.35, 5.1, 7.35, 4.0, color="#2980B9", lw=1.2, rad=-0.1)

    # Arrow from Encoder Stage 3:
    arrow(3.35, 2.55, 7.35, 2.45, color="#F39C12", lw=1.2)

    # Arrows from Segmentation Branch (Green):
    arrow(6.75, 6.2, 7.35, 6.0, color="#27AE60", lw=1.2)
    arrow(6.75, 6.0, 7.35, 2.8, color="#27AE60", lw=1.2, rad=-0.2)

    # Feature Fusion (Purple)
    box(10.7, 2.0, 1.7, 4.1, "Feature\nFusion", "320 + 320 + 112\n=\n752 ch", color="#F4ECF7", edge="#8E44AD", lw=1.4)
    arrow(10.15, 5.65, 10.55, 5.0, color="#E74C3C", lw=1.2)
    arrow(10.15, 4.0, 10.55, 4.0, color="#3498DB", lw=1.2)
    arrow(10.15, 2.45, 10.55, 3.0, color="#E74C3C", lw=1.2)

    # Classifier (Green)
    box(13.0, 2.6, 2.2, 2.9, "Classifier", "Linear 752 → 256", "Linear 256 → C\n(9 or 13)", color="#E8F8F5", edge="#27AE60", lw=1.3)
    arrow(12.55, 4.05, 12.85, 4.05, color="#1B2631", lw=1.4)

    # Bottom Equation Card
    formula_box = FancyBboxPatch((3.2, 0.4), 8.5, 0.95, boxstyle="round,pad=0.15",
                                  lw=1.0, edgecolor="#BDC3C7", facecolor="#F8F9F9", zorder=3)
    ax.add_patch(formula_box)
    ax.text(7.45, 0.88, r"$\mathbf{wpool}(f, M) = \frac{\sum_{i=1}^N M_i \cdot f_i}{\sum_{i=1}^N M_i + \varepsilon}, \quad \varepsilon = 10^{-6}$",
            ha="center", va="center", fontsize=9.2, color="#1B2631", zorder=4)

    fig.tight_layout()
    savefig(fig, "fig2_mpf_fusion.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Dataset Distribution (Updated with User's Exact D2 Class Counts)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_dataset_distribution():
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.6))
    datasets = {
        "D1: Atika Akter\n(4,237 images)": {
            "Normal": 505, "Glioma": 650, "Meningioma": 1000, "Pituitary": 988
        },
        "D2: Shubham Gajjar\n(3,349 matched pairs)": {
            "Gliomas": 1237, "Glioblastoma": 732, "Meningioma": 640, "Astrocytoma": 529, "Neurocytoma": 211
        },
        "D3: Mixed\n(6,380 images)": {
            "Normal": 1595, "Glioma": 1595, "Meningioma": 1595, "Pituitary": 1595
        },
        "Figshare: Cheng et al.\n(3,064 images)": {
            "Meningioma": 708, "Glioma": 1426, "Pituitary": 930
        }
    }
    palette = {
        "Normal": "#2ECC71", "Glioma": "#3498DB", "Gliomas": "#3498DB",
        "Glioblastoma": "#E74C3C", "Meningioma": "#E67E22", "Pituitary": "#9B59B6",
        "Astrocytoma": "#F39C12", "Neurocytoma": "#1ABC9C"
    }

    for ax, (title, cls_dict) in zip(axes, datasets.items()):
        labels = list(cls_dict.keys())
        counts = list(cls_dict.values())
        colors = [palette.get(l, "#95A5A6") for l in labels]
        bars = ax.bar(range(len(labels)), counts, color=colors, edgecolor="#2C3E50", lw=0.6, width=0.65)
        ax.set_xticks(range(len(labels)))
        short_labels = [l if len(l) <= 7 else l[:5] + ".." for l in labels]
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8.0, fontweight="bold")
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=6)
        ax.set_ylabel("Number of Slices", fontsize=8.0)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4, lw=0.5)
        ax.set_axisbelow(True)
        max_c = max(counts)
        ax.set_ylim(0, max_c * 1.18)
        for b, v in zip(bars, counts):
            ax.text(b.get_x() + b.get_width()/2.0, v + max_c*0.02, str(v),
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig3_dataset_distribution.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Training Curves
# ══════════════════════════════════════════════════════════════════════════════
def fig4_training_curves():
    zip_info = {
        "Exp 1: Baseline 256² (Patient-Level)": (
            os.path.join(BASE, "figshare exp-1 results.zip"),
            "Figshare_Experiment_Results/training_history.csv",
            "#2980B9", "-"),
        "Exp 2: +ASPP+FPN 512² (Patient-Level)": (
            os.path.join(BASE, "figshare exp-2 results.zip"),
            "Figshare_Experiment_ASPP_FPN_512/training_history.csv",
            "#27AE60", "-"),
        "Exp 3: +ASPP+FPN 512² (Slice-Level*)": (
            os.path.join(BASE, "figshare exp-3 results.zip"),
            "Figshare_Experiment3_SliceLevel_512/training_history.csv",
            "#E74C3C", "--"),
    }

    def load_avg(zpath, inner):
        data = {}
        with zf.ZipFile(zpath) as z:
            with z.open(inner) as f:
                reader = csv.DictReader(f.read().decode().splitlines())
                for row in reader:
                    ep = int(row["epoch"])
                    for k in ["train_loss", "val_f1", "val_dice"]:
                        data.setdefault(k, {}).setdefault(ep, []).append(float(row[k]))
        epochs = sorted(data["train_loss"].keys())
        return (epochs,
                [np.mean(data["train_loss"][e]) for e in epochs],
                [np.mean(data["val_f1"][e]) for e in epochs],
                [np.mean(data["val_dice"][e]) for e in epochs])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for label, (zp, inner, col, ls) in zip_info.items():
        try:
            epochs, loss, f1, dice = load_avg(zp, inner)
            axes[0].plot(epochs, loss, lw=2.2, color=col, linestyle=ls, label=label)
            axes[1].plot(epochs, f1, lw=2.2, color=col, linestyle=ls, label=label)
            axes[2].plot(epochs, dice, lw=2.2, color=col, linestyle=ls, label=label)
        except Exception as e:
            print(f"  Warning loading {label}: {e}")

    titles = ["(a) Combined Multi-Task Loss", "(b) Validation Macro-F1", "(c) Validation Mean Dice"]
    ylabels = ["Total Loss (Kendall Weighted)", "Macro F1-Score", "Mean Dice Coefficient"]
    for ax, t, yl in zip(axes, titles, ylabels):
        ax.set_title(t, fontsize=9.2, fontweight="bold", pad=6)
        ax.set_xlabel("Epoch", fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5)
        ax.tick_params(labelsize=8.0)
        ax.grid(True, linestyle="--", alpha=0.4, lw=0.5)
        ax.set_axisbelow(True)

    axes[0].legend(fontsize=7.2, loc="upper right", frameon=True)
    axes[1].legend(fontsize=7.2, loc="lower right", frameon=True)
    axes[2].legend(fontsize=7.2, loc="lower right", frameon=True)

    fig.tight_layout()
    savefig(fig, "fig4_training_curves.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════
def fig5_confusion_matrix():
    cms = {
        "(a) Exp 1: Baseline 256²\n(Patient-Level, N=629)": np.array([[128, 14, 0], [0, 296, 0], [0, 0, 191]]),
        "(b) Exp 2: +ASPP+FPN 512²\n(Patient-Level, N=629)": np.array([[130, 12, 0], [0, 296, 0], [1, 0, 190]]),
        "(c) Exp 3: +ASPP+FPN 512²\n(Slice-Level*, N=307)": np.array([[69, 0, 2], [0, 143, 0], [0, 0, 93]]),
    }
    classes = ["Meningioma", "Glioma", "Pituitary"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))

    for ax, (title, cm) in zip(axes, cms.items()):
        total = cm.sum(axis=1, keepdims=True)
        cm_norm = cm.astype(float) / total
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7.0)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=8.0)
        ax.set_yticklabels(classes, fontsize=8.0)
        ax.set_xlabel("Predicted Class", fontsize=8.2)
        ax.set_ylabel("True Class", fontsize=8.2)
        ax.set_title(title, fontsize=8.8, fontweight="bold", pad=6)
        thresh = cm_norm.max() / 2.0
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                        ha="center", va="center", fontsize=8.0, fontweight="bold",
                        color="white" if cm_norm[i, j] > thresh else "#1B2631")

    fig.tight_layout()
    savefig(fig, "fig5_confusion_matrix.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
def fig6_roc_curves():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    files = ["exp1_roc_curves.png", "exp2_roc_curves.png", "exp3_roc_curves.png"]
    titles = [
        "(a) Exp 1: Baseline 256² (Patient-Level)\nAUC: Mening=0.9989, Gli=0.9997, Pit=1.000",
        "(b) Exp 2: +ASPP+FPN 512² (Patient-Level)\nAUC: Mening=0.9983, Gli=0.9997, Pit=1.000",
        "(c) Exp 3: +ASPP+FPN 512² (Slice-Level*)\nAUC: Mening=0.9999, Gli=1.000, Pit=0.9998",
    ]
    for ax, fname, title in zip(axes, files, titles):
        fpath = os.path.join(OUT, fname)
        if os.path.exists(fpath):
            img = Image.open(fpath)
            ax.imshow(np.array(img))
            ax.set_title(title, fontsize=8.0, fontweight="bold", pad=5)
        ax.axis("off")

    fig.tight_layout()
    savefig(fig, "fig6_roc_curves.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Per-Class Segmentation Dice
# ══════════════════════════════════════════════════════════════════════════════
def fig7_per_class_dice():
    classes = ["Meningioma", "Glioma", "Pituitary"]
    exp1 = [0.9160, 0.6727, 0.8291]
    exp2 = [0.9216, 0.6802, 0.8318]
    exp3 = [0.9217, 0.7614, 0.8951]
    x = np.arange(len(classes))
    w = 0.24

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    b1 = ax.bar(x - w, exp1, w, label="Exp 1: Baseline 256² (Patient-Level)", color="#2980B9", edgecolor="#1B4F72", lw=0.8)
    b2 = ax.bar(x, exp2, w, label="Exp 2: +ASPP+FPN 512² (Patient-Level)", color="#27AE60", edgecolor="#145A32", lw=0.8)
    b3 = ax.bar(x + w, exp3, w, label="Exp 3: +ASPP+FPN 512² (Slice-Level*)", color="#E74C3C", edgecolor="#78281F", lw=0.8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Mean Dice Coefficient", fontsize=8.8)
    ax.set_ylim(0.55, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True)
    ax.tick_params(labelsize=8.0)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.008, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    fig.tight_layout()
    savefig(fig, "fig7_per_class_dice.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Representative Qualitative Segmentations for All 3 Experiments
# (Matching exact layout from user screenshot)
# ══════════════════════════════════════════════════════════════════════════════
def fig8_segmentations_all_experiments():
    from PIL import ImageDraw, ImageFont

    exp1_p = os.path.join(OUT, "exp1_segmentation_examples.png")
    exp2_p = os.path.join(OUT, "exp2_segmentation_examples.png")
    exp3_p = os.path.join(OUT, "exp3_segmentation_examples.png")

    if not (os.path.exists(exp1_p) and os.path.exists(exp2_p) and os.path.exists(exp3_p)):
        print("  [!] Missing segmentation example images for Fig 8")
        return

    im1 = Image.open(exp1_p).convert("RGB")
    im2 = Image.open(exp2_p).convert("RGB")
    im3 = Image.open(exp3_p).convert("RGB")

    w, h = im1.size  # 1300, 2925
    gap = 70
    header_h = 240
    total_w = w * 3 + gap * 2
    total_h = h + header_h

    comp = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(comp)

    comp.paste(im1, (0, header_h))
    comp.paste(im2, (w + gap, header_h))
    comp.paste(im3, (w * 2 + gap * 2, header_h))

    font_large = None
    font_sub = None
    font_col = None

    for font_path in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if os.path.exists(font_path):
            font_large = ImageFont.truetype(font_path, 42)
            font_sub = ImageFont.truetype(font_path, 34)
            font_col = ImageFont.truetype(font_path, 32)
            break

    if font_large is None:
        font_large = ImageFont.load_default()
        font_sub = font_large
        font_col = font_large

    cols = [
        (0, "(a) Experiment 1: Baseline 256×256", "(Patient-Level Split, 5-Fold CV)"),
        (w + gap, "(b) Experiment 2: MPF-Net + ASPP + FPN 512×512", "(Patient-Level Split, 5-Fold CV)"),
        (w * 2 + gap * 2, "(c) Experiment 3: MPF-Net + ASPP + FPN 512×512", "(Slice-Level Split*, Leakage Benchmark)")
    ]

    for x_offset, t1, t2 in cols:
        bbox1 = draw.textbbox((0, 0), t1, font=font_large)
        w1 = bbox1[2] - bbox1[0]
        draw.text((x_offset + (w - w1) / 2, 25), t1, fill=(20, 20, 20), font=font_large)

        bbox2 = draw.textbbox((0, 0), t2, font=font_sub)
        w2 = bbox2[2] - bbox2[0]
        draw.text((x_offset + (w - w2) / 2, 85), t2, fill=(80, 80, 80), font=font_sub)

        sub_w = w / 3.0
        for idx_sub, sub_title in enumerate(["Input MRI", "Ground Truth", "MPF-Net Pred"]):
            bbox_sub = draw.textbbox((0, 0), sub_title, font=font_col)
            w_sub = bbox_sub[2] - bbox_sub[0]
            x_sub = x_offset + idx_sub * sub_w + (sub_w - w_sub) / 2
            draw.text((x_sub, 160), sub_title, fill=(50, 50, 50), font=font_col)

    for i in [1, 2]:
        x_sep = i * w + (i - 0.5) * gap
        draw.line([(x_sep, 20), (x_sep, total_h - 20)], fill=(210, 210, 210), width=3)

    out_png = os.path.join(OUT, "fig8_segmentations_all_experiments.png")
    out_pdf = os.path.join(OUT, "fig8_segmentations_all_experiments.pdf")
    comp.save(out_png, quality=95)
    comp.save(out_pdf, "PDF", resolution=300.0)
    print("  [OK] Saved: fig8_segmentations_all_experiments.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9 — Multi-Metric Overall Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig9_overall_comparison():
    metrics = ["Accuracy", "Macro-F1", "Mean Dice", "Global Dice", "Mean IoU"]
    exp1_vals  = [0.9777, 0.9750, 0.7751, 0.8131, 0.6840]
    exp2_vals  = [0.9793, 0.9766, 0.7808, 0.8147, 0.6902]
    exp3_vals  = [0.9935, 0.9917, 0.8390, 0.8797, 0.7615]
    sahoo_vals = [0.9742, 0.0,    0.9011, 0.0,    0.0   ]

    x = np.arange(len(metrics))
    w = 0.18

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    b1 = ax.bar(x - 1.5*w, exp1_vals, w, label="Exp 1: Baseline 256² (Patient-Level)", color="#2980B9", edgecolor="#1B4F72", lw=0.8)
    b2 = ax.bar(x - 0.5*w, exp2_vals, w, label="Exp 2: +ASPP+FPN 512² (Patient-Level)", color="#27AE60", edgecolor="#145A32", lw=0.8)
    b3 = ax.bar(x + 0.5*w, exp3_vals, w, label="Exp 3: +ASPP+FPN 512² (Slice-Level*)", color="#E74C3C", edgecolor="#78281F", lw=0.8, alpha=0.85)

    bars_sahoo = ax.bar(x + 1.5*w, sahoo_vals, w, label="Sahoo et al. (DenseNet-201 FCN)", color="#8E44AD", edgecolor="#4A235A", lw=0.8, alpha=0.7)
    for i, v in enumerate([0.9742, None, 0.9011, None, None]):
        if v is None:
            bars_sahoo[i].set_hatch("///")
            bars_sahoo[i].set_alpha(0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8.8, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=8.5)
    ax.set_ylim(0.60, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.4, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, loc="upper right", frameon=True, ncol=2)
    ax.tick_params(labelsize=8.0)

    fig.tight_layout()
    savefig(fig, "fig9_overall_comparison.pdf")


if __name__ == "__main__":
    print("Generating base-paper styled publication-quality figures...")
    fig3_dataset_distribution()
    fig4_training_curves()
    fig5_confusion_matrix()
    fig6_roc_curves()
    fig7_per_class_dice()
    fig8_segmentations_all_experiments()
    fig9_overall_comparison()
    print("Done! All figures generated successfully.")
