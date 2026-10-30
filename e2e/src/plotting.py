import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

colors = {
    "LLM-Planner": "#D55E00",
    "LLM-Planner with Vision": "#009E73",
    "QUERI (with Vision)": "#0072B2",
}

valid_seen_results = {
    "LLM-Planner": 0.309,
    "LLM-Planner with Vision": 0.260,
    "QUERI (with Vision)": 0.329,
}

valid_unseen_results_Qwen3VL8B = {
    "LLM-Planner": 0.410,
    "LLM-Planner with Vision": 0.349,
    "QUERI (with Vision)": 0.438,
}

valid_unseen_results_GPT54 = {
    "LLM-Planner": 0.571,
    "LLM-Planner with Vision": 0.477,
    "QUERI (with Vision)": 0.570,
}

valid_unseen_results_ClaudeSonnet46 = {
    "LLM-Planner": 0.623,
    "LLM-Planner with Vision": 0.639,
    "QUERI (with Vision)": 0.647,
}

models = {
    "Qwen3VL8B": valid_unseen_results_Qwen3VL8B,
    "GPT5.4": valid_unseen_results_GPT54,
    "ClaudeSonnet4.6": valid_unseen_results_ClaudeSonnet46,
}

# Back-compat default for the single-model plot.
valid_unseen_results = valid_unseen_results_Qwen3VL8B


def plot_valid_unseen_vs_seen(
    output_path: str = "valid_results_comparison.png",
    *,
    unseen: dict | None = None,
    seen: dict | None = None,
    color_map: dict | None = None,
) -> None:
    """Two groups of three bars: valid unseen (left) and valid seen (right)."""
    unseen = unseen if unseen is not None else valid_unseen_results
    seen = seen if seen is not None else valid_seen_results
    color_map = color_map if color_map is not None else colors
    methods = list(color_map.keys())

    bar_width = 0.55
    intra_gap = 0.12
    group_gap = 1.35

    n = len(methods)
    x_unseen = np.arange(n) * (bar_width + intra_gap)
    x_seen = x_unseen[-1] + bar_width + group_gap + np.arange(n) * (bar_width + intra_gap)

    heights_unseen = [unseen[m] for m in methods]
    heights_seen = [seen[m] for m in methods]
    bar_colors = [color_map[m] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_unseen, heights_unseen, width=bar_width, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.bar(x_seen, heights_seen, width=bar_width, color=bar_colors, edgecolor="white", linewidth=0.8)

    ymax = max(*heights_unseen, *heights_seen)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel("Success rate")
    ax.set_title("ALFRED Benchmark")

    for x, h in zip(x_unseen, heights_unseen):
        ax.text(x, h, f"{h * 100:.1f}%", ha="center", va="bottom", fontsize=10)
    for x, h in zip(x_seen, heights_seen):
        ax.text(x, h, f"{h * 100:.1f}%", ha="center", va="bottom", fontsize=10)

    group_centers = (x_unseen.mean(), x_seen.mean())
    ax.set_xticks(group_centers)
    ax.set_xticklabels(["Valid unseen", "Valid seen"])

    ax.legend(
        handles=[Patch(facecolor=color_map[m], edgecolor="white", label=m) for m in methods],
        loc="upper right",
        framealpha=0.92,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_valid_unseen_all_models(
    output_path: str = "valid_unseen_results_comparison_all_models.png",
    *,
    models_unseen: dict[str, dict[str, float]] | None = None,
    color_map: dict[str, str] | None = None,
) -> None:
    """Three groups of three bars: valid unseen for each model."""
    models_unseen = models_unseen if models_unseen is not None else models
    color_map = color_map if color_map is not None else colors

    methods = list(color_map.keys())
    model_names = list(models_unseen.keys())

    bar_width = 0.55
    intra_gap = 0.12
    model_gap = 1.8  # gap between models

    n = len(methods)
    group_width = (n - 1) * (bar_width + intra_gap) + bar_width

    # Build x positions for each group (one group per model).
    group_centers: list[float] = []
    x_positions_by_group: list[np.ndarray] = []
    heights_by_group: list[list[float]] = []
    group_labels: list[str] = []

    start = 0.0
    for model_name in model_names:
        unseen_dict = models_unseen[model_name]

        x_unseen = start + np.arange(n) * (bar_width + intra_gap)
        center_unseen = float(x_unseen.mean())
        x_positions_by_group.append(x_unseen)
        heights_by_group.append([unseen_dict[m] for m in methods])
        group_centers.append(center_unseen)
        group_labels.append(model_name)

        start = start + group_width + model_gap

    bar_colors = [color_map[m] for m in methods]

    fig_w = max(10.0, 2.4 * len(group_labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5.3))

    for x_group, heights in zip(x_positions_by_group, heights_by_group):
        ax.bar(x_group, heights, width=bar_width, color=bar_colors, edgecolor="white", linewidth=0.8)
        for x, h in zip(x_group, heights):
            ax.text(x, h, f"{h * 100:.1f}%", ha="center", va="bottom", fontsize=9)

    ymax = max(h for heights in heights_by_group for h in heights)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel("Success rate")
    ax.set_title("ALFRED Benchmark (Valid unseen)")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels)

    ax.legend(
        handles=[Patch(facecolor=color_map[m], edgecolor="white", label=m) for m in methods],
        loc="upper left",
        framealpha=0.92,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_valid_unseen_all_models()