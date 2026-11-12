"""Forest and funnel plot figures for the Sysys meta-analysis."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats as sp_stats

from process.meta_analysis_bias import trim_and_fill
from process.meta_analysis_models import BiasTestResult, PooledEstimate, StudyEffect

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# p-values below this print as a bound instead of "0.0000" (audit 15-L4).
P_DISPLAY_FLOOR = 1e-4


def format_p_value(p_value: float) -> str:
    """Four-decimal p-value text, or "< 0.0001" for smaller values."""
    if p_value < P_DISPLAY_FLOOR:
        return f"< {P_DISPLAY_FLOOR:.4f}"
    return f"{p_value:.4f}"


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _p_text(p_value: float) -> str:
    """'= 0.0123' or '< 0.0001' for plot annotations."""
    text = format_p_value(p_value)
    return text if text.startswith("<") else f"= {text}"


def create_forest_plot_figure(
    pooled: PooledEstimate,
    title: str = "Forest Plot",
) -> Figure:
    """Create a matplotlib forest plot figure.

    Layout:
    - Left column: study labels
    - Center: horizontal lines (CI) with squares (effect size, sized by weight)
    - Bottom: diamond for pooled estimate
    - Right column: effect size [CI] text
    - Bottom annotation: heterogeneity stats (I^2, Q, tau^2, p)

    Returns a matplotlib Figure (not shown, for embedding in GUI).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    studies = pooled.studies
    k = len(studies)
    if k == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No studies to display", ha="center", va="center",
                fontsize=11, color="#808080")
        ax.axis("off")
        fig.tight_layout()
        return fig

    # Figure sizing
    row_height = 0.45
    fig_height = max(4.0, (k + 3) * row_height + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Y positions: studies top-down, pooled at bottom
    y_positions = list(range(k - 1, -1, -1))  # top to bottom
    y_pooled = -1.5

    # Determine x-axis range
    all_ci_lo = [s.ci_lower for s in studies]
    all_ci_hi = [s.ci_upper for s in studies]
    x_min = min(min(all_ci_lo), pooled.ci_lower, 0) - 0.3
    x_max = max(max(all_ci_hi), pooled.ci_upper, 0) + 0.3

    # Plot each study
    max_weight = max(s.weight for s in studies) if studies else 1.0
    for i, (s, y) in enumerate(zip(studies, y_positions)):
        # CI line
        ax.plot([s.ci_lower, s.ci_upper], [y, y], color="#2c3e50",
                linewidth=1.0, solid_capstyle="butt", zorder=2)

        # Square sized by weight
        sq_size = max(4.0, 12.0 * (s.weight / max_weight if max_weight > 0 else 0.5))
        ax.plot(s.effect_size, y, marker="s", color="#2980b9",
                markersize=sq_size, zorder=3)

        # Study label (left)
        label = s.title[:40] + "..." if len(s.title) > 40 else s.title
        if not label:
            label = s.article_id[:30]
        ax.text(x_min - 0.05, y, label, ha="right", va="center",
                fontsize=8, fontfamily="sans-serif")

        # Effect + CI text (right)
        ci_text = f"{s.effect_size:.2f} [{s.ci_lower:.2f}, {s.ci_upper:.2f}]"
        ax.text(x_max + 0.05, y, ci_text, ha="left", va="center",
                fontsize=8, fontfamily="monospace")

    # Pooled estimate diamond
    diamond_half_h = 0.35
    diamond = Polygon([
        (pooled.ci_lower, y_pooled),
        (pooled.pooled_effect, y_pooled + diamond_half_h),
        (pooled.ci_upper, y_pooled),
        (pooled.pooled_effect, y_pooled - diamond_half_h),
    ], closed=True, facecolor="#e74c3c", edgecolor="#c0392b", linewidth=1.2, zorder=3)
    ax.add_patch(diamond)

    # Pooled label
    ax.text(x_min - 0.05, y_pooled, "Pooled RE", ha="right", va="center",
            fontsize=9, fontweight="bold", fontfamily="sans-serif")
    ci_text = f"{pooled.pooled_effect:.2f} [{pooled.ci_lower:.2f}, {pooled.ci_upper:.2f}]"
    ax.text(x_max + 0.05, y_pooled, ci_text, ha="left", va="center",
            fontsize=8, fontweight="bold", fontfamily="monospace")

    # Vertical null-effect line
    ax.axvline(x=0, color="#7f8c8d", linestyle="--", linewidth=0.8, zorder=1)

    # Dashed vertical line at pooled effect
    ax.axvline(x=pooled.pooled_effect, color="#e74c3c", linestyle=":",
               linewidth=0.6, alpha=0.5, zorder=1)

    # Separator line between studies and pooled
    sep_y = -0.75
    ax.axhline(y=sep_y, color="#bdc3c7", linewidth=0.5)

    # Heterogeneity annotation
    het_text = (
        f"Heterogeneity: I\u00b2 = {pooled.i_squared:.1f}%, "
        f"\u03c4\u00b2 = {pooled.tau_squared:.4f}, "
        f"Q = {pooled.q_statistic:.2f} (p {_p_text(pooled.q_p_value)})"
    )
    test_text = (
        f"Test for overall effect: Z = {pooled.z_value:.2f} "
        f"(p {_p_text(pooled.p_value)})"
    )
    ax.text(0.5, -0.06, het_text, ha="center", va="top",
            fontsize=8, color="#555555", transform=ax.transAxes)
    ax.text(0.5, -0.09, test_text, ha="center", va="top",
            fontsize=8, color="#555555", transform=ax.transAxes)

    # Axis formatting
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_pooled - 1.0, k - 0.5)
    ax.set_xlabel(studies[0].analysis_scale, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.subplots_adjust(left=0.30, right=0.70, bottom=0.15, top=0.93)
    return fig


def create_funnel_plot_figure(
    studies: list[StudyEffect],
    pooled_effect: float,
    bias_result: BiasTestResult | None = None,
    title: str = "Funnel Plot",
) -> Figure:
    """Create a contour-enhanced funnel plot.

    Layout:
    - X-axis: effect size
    - Y-axis: standard error (inverted - larger studies on top)
    - Vertical line at pooled estimate
    - Funnel outline (95% pseudo-CI)
    - Contour regions for significance (p < .05, .01, .001) in light gray shades
    - Points for each study (filled circles)
    - If bias_result has trim-and-fill data: show imputed studies as open circles
    - Bottom annotation: Egger's test result, trim-and-fill adjustment

    Returns a matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not studies:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(0.5, 0.5, "No studies to display", ha="center", va="center",
                fontsize=11, color="#808080")
        ax.axis("off")
        fig.tight_layout()
        return fig

    effects = [s.effect_size for s in studies]
    ses = [s.standard_error for s in studies]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Determine axis ranges
    se_max = max(ses) * 1.15
    se_range = np.linspace(0.001, se_max, 200)

    # Contour regions: significance levels
    for z_crit, alpha_label, shade in [
        (sp_stats.norm.ppf(0.9995), "p < 0.001", "#f0f0f0"),
        (sp_stats.norm.ppf(0.995), "p < 0.01", "#e0e0e0"),
        (sp_stats.norm.ppf(0.975), "p < 0.05", "#d0d0d0"),
    ]:
        left = pooled_effect - z_crit * se_range
        right = pooled_effect + z_crit * se_range
        ax.fill_betweenx(se_range, left, right, alpha=0.4, color=shade,
                         label=alpha_label, zorder=0)

    # 95% pseudo-CI funnel outline
    z95 = 1.96
    funnel_lo = pooled_effect - z95 * se_range
    funnel_hi = pooled_effect + z95 * se_range
    ax.plot(funnel_lo, se_range, "--", color="#7f8c8d", linewidth=0.8, zorder=1)
    ax.plot(funnel_hi, se_range, "--", color="#7f8c8d", linewidth=0.8, zorder=1)

    # Vertical line at pooled effect
    ax.axvline(x=pooled_effect, color="#2c3e50", linestyle="-", linewidth=1.0,
               zorder=1, label=f"Pooled = {pooled_effect:.3f}")

    # Plot observed studies
    ax.scatter(effects, ses, s=40, c="#2980b9", edgecolors="#1a5276",
               linewidths=0.5, zorder=4, label=f"Studies (k={len(studies)})")

    # Plot imputed studies from trim-and-fill
    if bias_result and bias_result.trimfill_n_imputed > 0:
        # Reconstruct imputed positions
        tf_data = trim_and_fill(studies)
        imputed_studies = tf_data.get("imputed_studies", [])
        if imputed_studies:
            imp_effects = [s.effect_size for s in imputed_studies]
            imp_ses = [s.standard_error for s in imputed_studies]
            ax.scatter(imp_effects, imp_ses, s=40, facecolors="none",
                       edgecolors="#e74c3c", linewidths=1.2, zorder=4,
                       label=f"Imputed (k={len(imputed_studies)})")

    # Egger's regression line
    if bias_result and bias_result.eggers_p < 1.0 and len(studies) >= 3:
        # Regression line: standardized_effect = intercept + slope * precision
        # On the funnel plot: effect = intercept*SE + slope
        # This is approximate — draw from se=0 to se=se_max
        # Reconstruct slope from OLS
        precision = np.array([1.0 / s.standard_error for s in studies if s.standard_error > 0])
        std_effect = np.array([s.effect_size / s.standard_error for s in studies if s.standard_error > 0])
        if len(precision) >= 3:
            X = np.column_stack([np.ones(len(precision)), precision])
            beta = np.linalg.lstsq(X, std_effect, rcond=None)[0]
            # On funnel plot coords: effect = beta[0]*SE + beta[1]
            line_effects = beta[0] * se_range + beta[1]
            ax.plot(line_effects, se_range, "-", color="#e74c3c", linewidth=1.0,
                    alpha=0.7, zorder=2, label="Egger's regression")

    # Annotations
    annotations = []
    if bias_result:
        annotations.append(
            f"Egger's: intercept = {bias_result.eggers_intercept:.3f}, "
            f"p {_p_text(bias_result.eggers_p)}"
        )
        if bias_result.trimfill_n_imputed > 0:
            annotations.append(
                f"Trim-and-fill: {bias_result.trimfill_n_imputed} imputed, "
                f"adj. effect = {bias_result.trimfill_adjusted_effect:.3f}"
            )
        annotations.append(
            f"Fail-safe N = {bias_result.failsafe_n}"
        )

    if annotations:
        ann_text = " | ".join(annotations)
        ax.text(0.5, -0.10, ann_text, ha="center", va="top",
                fontsize=7.5, color="#555555", transform=ax.transAxes)

    # Axis formatting
    ax.set_xlabel(studies[0].analysis_scale, fontsize=10)
    ax.set_ylabel("Standard Error", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.invert_yaxis()  # Larger studies (smaller SE) at top
    ax.set_ylim(se_max, 0)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    return fig
