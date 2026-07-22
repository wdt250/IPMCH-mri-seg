"""Create the final 2x3 manual-versus-automatic agreement figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FEATURES = [
    ("original_firstorder_Mean", "First-order Mean"),
    ("original_glcm_Contrast", "GLCM Contrast"),
    ("original_shape2D_Sphericity", "Shape2D Sphericity"),
]
POINT = "#3274A1"
BIAS = "#0072B2"
LOA = "#D55E00"
NEUTRAL = "#4A4A4A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def limits(manual: np.ndarray, automatic: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    mean_value = (manual + automatic) / 2
    difference = automatic - manual
    bias = float(difference.mean())
    standard_deviation = float(difference.std(ddof=1))
    return (
        mean_value,
        difference,
        bias,
        bias - 1.96 * standard_deviation,
        bias + 1.96 * standard_deviation,
    )


def paired_panel(ax, manual: np.ndarray, automatic: np.ndarray, title: str) -> None:
    combined = np.concatenate([manual, automatic])
    low, high = float(combined.min()), float(combined.max())
    padding = max((high - low) * 0.06, 1e-9)
    low -= padding
    high += padding
    ax.scatter(manual, automatic, s=27, alpha=0.82, color=POINT, edgecolors="none")
    ax.plot([low, high], [low, high], color=NEUTRAL, linestyle=":", linewidth=1.4)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Manual measurement")
    ax.set_ylabel("Automatic measurement")


def bland_altman_panel(
    ax,
    mean_value: np.ndarray,
    difference: np.ndarray,
    bias: float,
    lower: float,
    upper: float,
) -> None:
    ax.scatter(mean_value, difference, s=27, alpha=0.82, color=POINT, edgecolors="none")
    ax.axhline(bias, color=BIAS, linewidth=1.6)
    ax.axhline(lower, color=LOA, linestyle="--", linewidth=1.3)
    ax.axhline(upper, color=LOA, linestyle="--", linewidth=1.3)
    ax.axhline(0, color=NEUTRAL, linewidth=0.9, alpha=0.75)
    ax.set_xlabel("Mean of manual and automatic measurements")
    ax.set_ylabel("Automatic - manual")


def polish(ax) -> None:
    ax.grid(alpha=0.20, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    args = parse_args()
    values = pd.read_csv(args.result_dir / "pyradiomics_feature_values_long.csv")
    agreement = pd.read_csv(args.result_dir / "pyradiomics_feature_agreement.csv")
    agreement = agreement[agreement["group"].eq("overall")].set_index("feature")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8.8), dpi=180)
    rows = []
    footer = []
    for column, (feature, title) in enumerate(FEATURES):
        subset = values[values["feature"].eq(feature)].dropna(subset=["manual", "auto"])
        manual = subset["manual"].to_numpy(dtype=float)
        automatic = subset["auto"].to_numpy(dtype=float)
        mean_value, difference, bias, lower, upper = limits(manual, automatic)
        paired_panel(axes[0, column], manual, automatic, title)
        bland_altman_panel(axes[1, column], mean_value, difference, bias, lower, upper)
        statistic = agreement.loc[feature]
        row = {
            "feature": feature,
            "n_pairs": int(len(subset)),
            "ICC2": float(statistic["ICC2"]),
            "ICC2_CI95_low": float(statistic["ICC2_CI95_low"]),
            "ICC2_CI95_high": float(statistic["ICC2_CI95_high"]),
            "spearman_rho": float(statistic["spearman_rho"]),
            "BA_bias": bias,
            "BA_lower_LoA": lower,
            "BA_upper_LoA": upper,
            "proportional_bias_p": float(statistic["proportional_bias_p"]),
        }
        rows.append(row)
        footer.append(
            f"{title}: ICC={row['ICC2']:.3f}, rho={row['spearman_rho']:.3f}, "
            f"bias={bias:.3g}, 95% LoA [{lower:.3g}, {upper:.3g}], "
            f"proportional-bias P={row['proportional_bias_p']:.3g}"
        )

    for ax in axes.flat:
        polish(ax)
    handles = [
        Line2D([0], [0], color=NEUTRAL, linestyle=":", linewidth=1.4, label="Identity line"),
        Line2D([0], [0], color=BIAS, linewidth=1.6, label="Bias"),
        Line2D([0], [0], color=LOA, linestyle="--", linewidth=1.3, label="95% limits of agreement"),
        Line2D([0], [0], color=NEUTRAL, linewidth=0.9, label="Zero difference"),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.105), ncol=4, frameon=False)
    figure.text(0.5, 0.018, "\n".join(footer), ha="center", va="bottom", fontsize=7.9, linespacing=1.42)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.95, bottom=0.235, hspace=0.40, wspace=0.28)
    figure.savefig(args.output, dpi=300, facecolor="white")
    plt.close(figure)
    pd.DataFrame(rows).to_csv(args.summary, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
