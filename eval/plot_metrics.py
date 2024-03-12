import argparse
import pickle
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.colors import LogNorm

from data.config import EXPERIMENT_ROOT


def compute_statistics_from_file(path: Path) -> Dict[str, np.array]:
    with open(path, "r") as f:
        data = np.genfromtxt(f, delimiter=",")

    mean_lat = np.zeros([50, 10], dtype=np.double)
    mean_long = np.zeros([50, 10], dtype=np.double)
    count = np.zeros([50, 10], dtype=np.uint)
    for i in range(data.shape[0]):
        h = int(data[i, 1])
        r = int(data[i, 2]) // 5 - 1
        laterr = np.abs(data[i, 3])
        longerr = np.abs(data[i, 4])

        count[h, r] += 1
        mean_lat[h, r] += (laterr - mean_lat[h, r]) / count[h, r]
        mean_long[h, r] += (longerr - mean_long[h, r]) / count[h, r]
    return dict(
        lat=mean_lat,
        long=mean_long,
        count=count,
    )


def combine_statistics(a, b) -> Dict[str, np.array]:
    count = a["count"] + b["count"]
    c = count.copy()
    c[c < 1] = 1
    lat = (a["lat"] * a["count"] + b["lat"] * b["count"]) / c
    long = (a["long"] * a["count"] + b["long"] * b["count"]) / c
    return dict(
        lat=lat,
        long=long,
        count=count,
    )


def compute_statistics(metrics_dir: Path) -> Dict[str, np.array]:
    statistics = None
    for metrics_file in tqdm.tqdm(metrics_dir.iterdir()):
        if metrics_file.stem == "statistics":
            continue
        stat = compute_statistics_from_file(metrics_file)
        if statistics is None:
            statistics = stat
        else:
            statistics = combine_statistics(statistics, stat)
    return statistics


def load_statistics(metrics_dir: Path, force: bool) -> Dict[str, np.array]:
    path = metrics_dir / "statistics.pickle"
    if not path.exists() or force:
        stats = compute_statistics(metrics_dir)
        with open(path, "wb") as f:
            pickle.dump(stats, f)
    else:
        with open(path, "rb") as f:
            stats = pickle.load(f)
    return stats


def plot_metrics(metrics_dir: Path, force: bool) -> None:
    stats = load_statistics(metrics_dir, force)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 8))

    c0 = axes[0].imshow(
        stats["long"],
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(),
        extent=[0, 10, 10, 0.2],
    )
    fig.colorbar(c0, ax=axes[0])
    axes[0].set_xticks(0.2 * np.arange(5, 55, 5))
    axes[0].set_yticks(0.2 * np.arange(1, 51, 1))
    axes[0].set_xlabel("Rollout Duration (sec)", fontsize=16)
    axes[0].set_ylabel("History Duration (sec)", fontsize=16)
    axes[0].set_title("Longitudinal Error", fontsize=20)

    c1 = axes[1].imshow(
        stats["lat"],
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(),
        extent=[0, 10, 10, 0.2],
    )
    fig.colorbar(c1, ax=axes[1])
    axes[1].set_xticks(0.2 * np.arange(5, 55, 5))
    axes[1].set_yticks(0.2 * np.arange(1, 51, 1))
    axes[1].set_xlabel("Rollout Duration (sec)", fontsize=16)
    axes[1].set_title("Lateral Error", fontsize=20)

    c2 = axes[2].imshow(
        np.sqrt(stats["lat"]**2 + stats["long"]**2),
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(),
        extent=[0, 10, 10, 0.2],
    )
    cb = fig.colorbar(c2, ax=axes[2])
    cb.set_label("Mean Error (meters)", fontsize=16)
    axes[2].set_xticks(0.2 * np.arange(5, 55, 5))
    axes[2].set_yticks(0.2 * np.arange(1, 51, 1))
    axes[2].set_xlabel("Rollout Duration (sec)", fontsize=16)
    axes[2].set_title("Total Error", fontsize=20)

    # Adjust the layout and save the figure
    plt.tight_layout()
    plt.savefig(metrics_dir / "statistics.png", dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    metrics_dir = EXPERIMENT_ROOT / args.experiment_name / "metrics"
    if not metrics_dir.exists():
        raise ValueError(
            f"Metrics have not been generated for {args.experiment_name} yet!"
        )
    plot_metrics(metrics_dir, args.force)


if __name__ == "__main__":
    main()
