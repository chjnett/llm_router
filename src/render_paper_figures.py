"""Render publication figures from the checked-in paper result snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


BLUE = "#2563EB"
PURPLE = "#7C3AED"
GREEN = "#15803D"
AMBER = "#B45309"
INK = "#172033"
MUTED = "#64748B"
GRID = "#E3E8EF"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def save_all(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_figure(output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    nodes = [
        (0.2, 1.55, 1.45, 0.9, "Query", "input", "#F7F9FC"),
        (2.0, 1.55, 1.65, 0.9, "Lower SLM", "first answer", "#F0FDF4"),
        (4.0, 1.55, 1.7, 0.9, "Confidence", "output-aware", "#EFF6FF"),
        (6.1, 1.55, 1.8, 0.9, "Answer-only", "verification", "#F5F3FF"),
        (8.35, 2.45, 1.6, 0.9, "Accept", "answers agree", "#F0FDF4"),
        (8.35, 0.65, 1.6, 0.9, "Upper LLM", "escalation", "#FFFBEB"),
        (10.35, 1.55, 1.45, 0.9, "Response", "final output", "#F7F9FC"),
    ]
    for x, y, width, height, title, subtitle, color in nodes:
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08,rounding_size=0.09", facecolor=color, edgecolor=GRID, linewidth=1.1))
        ax.text(x + width / 2, y + 0.57, title, ha="center", va="center", weight="bold", color=INK)
        ax.text(x + width / 2, y + 0.27, subtitle, ha="center", va="center", fontsize=7.5, color=MUTED)
    arrows = [
        ((1.65, 2.0), (2.0, 2.0)), ((3.65, 2.0), (4.0, 2.0)), ((5.7, 2.0), (6.1, 2.0)),
        ((7.9, 2.1), (8.35, 2.75)), ((7.9, 1.9), (8.35, 1.1)),
        ((9.95, 2.75), (10.35, 2.1)), ((9.95, 1.1), (10.35, 1.9)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#94A3B8", "lw": 1.4})
    ax.text(8.0, 2.72, "agree", fontsize=7.5, color=GREEN, ha="right")
    ax.text(8.0, 1.12, "disagree / low confidence", fontsize=7.5, color=AMBER, ha="right")
    ax.set_title("Output-aware SLM–LLM routing with answer-only verification", loc="left", color=INK, pad=2)
    save_all(fig, output_dir, "fig1_architecture")


def latency_figure(data, output_dir: Path):
    latency = data["latency"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    labels = ["Full second pass", "Answer-only"]
    colors = ["#CBD5E1", PURPLE]
    for ax, values, ylabel, reduction in (
        (axes[0], [latency["full_second_pass_ms"], latency["answer_only_ms"]], "Latency per item (ms)", "94.09% lower"),
        (axes[1], [latency["full_second_pass_tokens"], latency["answer_only_tokens"]], "Generated tokens per item", "95.67% fewer"),
    ):
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", weight="bold", color=INK)
        ax.text(0.98, 0.92, reduction, transform=ax.transAxes, ha="right", color=PURPLE, weight="bold")
    fig.suptitle("Answer-only verification removes most second-pass decode cost", x=0.08, ha="left", weight="bold", color=INK)
    fig.tight_layout()
    save_all(fig, output_dir, "fig2_latency_tokens")


def pareto_figure(data, output_dir: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.35))
    points = data["selection_pareto"]
    costs = [point["cost"] for point in points]
    accuracies = [100 * point["accuracy"] for point in points]
    ax.plot(costs, accuracies, "-o", color=BLUE, markerfacecolor="white", linewidth=2, markersize=5, label="Selection Pareto")
    frozen = points[-2]
    ax.scatter([frozen["cost"]], [100 * frozen["accuracy"]], s=95, color=PURPLE, edgecolor="white", linewidth=1.5, zorder=5, label="Frozen policy")
    ax.annotate("0.12 / 0.80", (frozen["cost"], 100 * frozen["accuracy"]), xytext=(-62, 14), textcoords="offset points", color=PURPLE, weight="bold", arrowprops={"arrowstyle": "-", "color": PURPLE})
    ax.axvline(1.0, color="#94A3B8", linestyle="--", linewidth=1, label="Always Upper cost")
    ax.set_xlabel("Normalized cascade cost")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0.18, 1.06)
    ax.set_ylim(70, 90)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Performance–cost frontier on the policy-selection split", loc="left")
    fig.tight_layout()
    save_all(fig, output_dir, "fig3_pareto_frontier")


def comparison_figure(data, output_dir: Path):
    evaluation = data["evaluation"]
    split_keys = ["risk_certification", "official_test"]
    split_labels = ["Certification", "Official test"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    x = np.arange(2)
    width = 0.34
    for index, method in enumerate(("c3", "answer_only")):
        acc = [100 * evaluation[split][method]["accuracy"] for split in split_keys]
        cost = [evaluation[split][method]["cost"] for split in split_keys]
        color = BLUE if method == "c3" else PURPLE
        label = "C3" if method == "c3" else "Answer-only"
        axes[0].bar(x + (index - 0.5) * width, acc, width, label=label, color=color)
        axes[1].bar(x + (index - 0.5) * width, cost, width, label=label, color=color)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(80, 91)
    axes[1].set_ylabel("Normalized cascade cost")
    axes[1].axhline(1.0, color="#94A3B8", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.7, 1.1)
    for ax in axes:
        ax.set_xticks(x, split_labels)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Answer-only verification trades a small accuracy change for lower cost", x=0.08, ha="left", weight="bold", color=INK)
    fig.tight_layout()
    save_all(fig, output_dir, "fig4_method_comparison")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="paper/data/paper_results.json")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    setup_style()
    architecture_figure(output_dir)
    latency_figure(data, output_dir)
    pareto_figure(data, output_dir)
    comparison_figure(data, output_dir)
    print(f"Rendered 4 figures in SVG/PDF/PNG to {output_dir}")


if __name__ == "__main__":
    main()
