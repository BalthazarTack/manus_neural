import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot validation metrics over training steps")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/media/rana/Balthazar/3Dreconstruction/manus/outputs/hand/subject0/test_nan/results/val_results/val_results.csv",
        help="Path to val_results.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to the CSV directory.",
    )
    return parser


def read_metrics(csv_path: str) -> Dict[str, List[float]]:
    steps: List[float] = []
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []

    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                steps.append(float(row["step"]))
                psnr_vals.append(float(row["psnr"]))
                ssim_vals.append(float(row["ssim"]))
                lpips_vals.append(float(row["lpips"]))
            except (KeyError, TypeError, ValueError):
                continue

    order = sorted(range(len(steps)), key=lambda index: steps[index])
    return {
        "step": [steps[index] for index in order],
        "psnr": [psnr_vals[index] for index in order],
        "ssim": [ssim_vals[index] for index in order],
        "lpips": [lpips_vals[index] for index in order],
    }


def plot_single_metric(steps: List[float], values: List[float], title: str, ylabel: str, output_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(steps, values, marker="o", linewidth=2, markersize=3)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_combined(steps: List[float], metrics: Dict[str, List[float]], output_path: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    plots = [
        ("psnr", "PSNR", "PSNR"),
        ("ssim", "SSIM", "SSIM"),
        ("lpips", "LPIPS", "LPIPS"),
    ]

    for axis, (key, title, ylabel) in zip(axes, plots):
        axis.plot(steps, metrics[key], marker="o", linewidth=2, markersize=3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = get_parser().parse_args()
    metrics = read_metrics(args.csv_path)

    if not metrics["step"]:
        raise RuntimeError(f"No valid metric rows found in {args.csv_path}")

    output_dir = args.output_dir or os.path.dirname(args.csv_path)
    os.makedirs(output_dir, exist_ok=True)

    plot_combined(metrics["step"], metrics, os.path.join(output_dir, "val_metrics_over_steps.png"))
    plot_single_metric(metrics["step"], metrics["psnr"], "PSNR over Steps", "PSNR", os.path.join(output_dir, "psnr_over_steps.png"))
    plot_single_metric(metrics["step"], metrics["ssim"], "SSIM over Steps", "SSIM", os.path.join(output_dir, "ssim_over_steps.png"))
    plot_single_metric(metrics["step"], metrics["lpips"], "LPIPS over Steps", "LPIPS", os.path.join(output_dir, "lpips_over_steps.png"))

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()