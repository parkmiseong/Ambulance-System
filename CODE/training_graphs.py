# -*- coding: utf-8 -*-
"""Training metric CSV writer and PNG dashboard. No SUMO dependency."""
import argparse
import csv
import os
from pathlib import Path

FIELDS = ["episode", "scenario", "patients", "missions", "arrived", "failed",
          "reward", "avg_reward", "avg_loss", "epsilon", "episode_seconds"]


def append_episode_metrics(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def plot_training_metrics(csv_path, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    def values(key):
        return [float(row[key]) for row in rows]
    ep = values("episode")
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = [
        (axs[0, 0], "avg_reward", "Average reward per mission"),
        (axs[0, 1], "avg_loss", "Mean DQN loss"),
        (axs[1, 0], "epsilon", "Epsilon"),
        (axs[1, 1], "episode_seconds", "Episode wall time (seconds)"),
    ]
    for ax, key, title in panels:
        ax.plot(ep, values(key), marker="o", markersize=2.5, linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Ambulance DQN - Training Dashboard", fontsize=15)
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = png_path.with_name(png_path.stem + ".tmp.png")
    fig.savefig(tmp_path, dpi=140)
    plt.close(fig)
    os.replace(tmp_path, png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saved training CSV -> graph PNG")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_training_metrics(args.csv, args.output)
    print(f"Saved: {args.output}")
