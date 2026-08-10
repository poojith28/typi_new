#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEP_AL_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DEEP_AL_ROOT, ".."))

import sys

if _DEEP_AL_ROOT not in sys.path:
    sys.path.insert(0, _DEEP_AL_ROOT)

import pycls.datasets.utils as ds_utils


def infer_run_metadata(run_dir):
    summary_path = os.path.join(run_dir, "benchmark_summary.json")
    if not os.path.exists(summary_path):
        episode_paths = sorted(Path(run_dir).glob("episode_*/episode_summary.json"))
        if not episode_paths:
            raise FileNotFoundError(f"Missing benchmark_summary.json and episode_summary.json files in {run_dir}")
        with open(episode_paths[0], "r") as handle:
            episode_summary = json.load(handle)
        run_path = Path(run_dir)
        dataset = str(episode_summary.get("dataset", run_path.parts[-3] if len(run_path.parts) >= 3 else ""))
        seed = episode_summary.get("seed", None)
        if seed is None:
            exp_name = str(episode_summary.get("exp_name", run_path.name))
            import re
            match = re.search(r"(_s|seed[_-]?)(\d+)", exp_name)
            if match:
                seed = int(match.group(2))
        if seed is None:
            raise RuntimeError(f"Could not infer seed from episode summaries in {run_dir}")
        return dataset, int(seed)
    with open(summary_path, "r") as handle:
        summary = json.load(handle)
    dataset = str(summary.get("dataset", ""))
    seed = summary.get("seed", None)
    if seed is None:
        raise RuntimeError(f"Could not infer seed from {summary_path}")
    return dataset, int(seed)


def dataset_root(dataset):
    if dataset in {"CIFAR10", "CIFAR100"}:
        return os.path.join(_PROJECT_ROOT, "data")
    if dataset == "TINYIMAGENET":
        return os.path.join(_PROJECT_ROOT, "scan", "datasets", "TinyImageNet", "tiny-imagenet-200")
    raise NotImplementedError(f"Unsupported dataset for feature evaluation: {dataset}")


def load_labels(dataset):
    from torchvision.datasets import CIFAR10, CIFAR100
    from pycls.datasets.tiny_imagenet import TinyImageNet

    root = dataset_root(dataset)
    if dataset == "CIFAR10":
        train_data = CIFAR10(root=root, train=True, download=False)
        test_data = CIFAR10(root=root, train=False, download=False)
    elif dataset == "CIFAR100":
        train_data = CIFAR100(root=root, train=True, download=False)
        test_data = CIFAR100(root=root, train=False, download=False)
    elif dataset == "TINYIMAGENET":
        train_data = TinyImageNet(root=root, split="train")
        test_data = TinyImageNet(root=root, split="val")
    else:
        raise NotImplementedError(f"Unsupported dataset for feature evaluation: {dataset}")
    train_labels = np.asarray(train_data.targets, dtype=np.int64)
    test_labels = np.asarray(test_data.targets, dtype=np.int64)
    return train_labels, test_labels


def load_episode_dirs(run_dir):
    episode_dirs = []
    for path in Path(run_dir).glob("episode_*"):
        if path.is_dir():
            try:
                episode_id = int(path.name.split("_")[-1])
            except Exception:
                continue
            if (path / "lSet.npy").exists():
                episode_dirs.append((episode_id, path))
    return sorted(episode_dirs, key=lambda x: x[0])


def linear_probe_accuracy(train_x, train_y, test_x, test_y, max_epochs, lr, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    train_x_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.long, device=device)
    test_x_t = torch.tensor(test_x, dtype=torch.float32, device=device)
    test_y_t = torch.tensor(test_y, dtype=torch.long, device=device)

    num_classes = int(max(int(train_y_t.max().item()), int(test_y_t.max().item())) + 1)
    model = nn.Linear(train_x_t.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    batch_size = min(256, len(train_y))

    for _ in range(int(max_epochs)):
        perm = torch.randperm(train_x_t.shape[0], device=device)
        for start in range(0, train_x_t.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            logits = model(train_x_t[idx])
            loss = F.cross_entropy(logits, train_y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        pred = model(test_x_t).argmax(dim=1)
        acc = (pred == test_y_t).float().mean().item() * 100.0
    return float(acc)


def knn_accuracy(train_x, train_y, test_x, test_y, k_value):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x_t = torch.tensor(train_x, dtype=torch.float32, device=device)
    train_y_t = torch.tensor(train_y, dtype=torch.long, device=device)
    test_x_t = torch.tensor(test_x, dtype=torch.float32, device=device)
    test_y_t = torch.tensor(test_y, dtype=torch.long, device=device)

    k_eff = max(1, min(int(k_value), train_x_t.shape[0]))
    num_classes = int(max(int(train_y_t.max().item()), int(test_y_t.max().item())) + 1)
    chunk_size = 512
    preds = []

    with torch.no_grad():
        for start in range(0, test_x_t.shape[0], chunk_size):
            chunk = test_x_t[start:start + chunk_size]
            sim = chunk @ train_x_t.T
            topk_sim, topk_idx = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=False)
            topk_labels = train_y_t[topk_idx]
            class_scores = torch.zeros((chunk.shape[0], num_classes), device=device)
            class_scores.scatter_add_(1, topk_labels, topk_sim.clamp_min(0.0))
            preds.append(class_scores.argmax(dim=1))
        pred = torch.cat(preds, dim=0)
        acc = (pred == test_y_t).float().mean().item() * 100.0
    return float(acc)


def maybe_load_pixel_curve(run_dir):
    summary_path = os.path.join(run_dir, "benchmark_summary.json")
    curve = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as handle:
            summary = json.load(handle)
        for record in summary.get("episode_records", []):
            curve[int(record.get("episode", -1))] = float(record.get("test_accuracy", float("nan")))
        if curve:
            return curve

    for episode_summary_path in sorted(Path(run_dir).glob("episode_*/episode_summary.json")):
        with open(episode_summary_path, "r") as handle:
            record = json.load(handle)
        curve[int(record.get("episode", -1))] = float(record.get("test_accuracy", float("nan")))
    return curve


def evaluate_run(run_dir, mode, knn_k, linear_lr, linear_weight_decay, linear_max_iter, output_subdir):
    dataset, seed = infer_run_metadata(run_dir)

    train_features = ds_utils.load_features(dataset, seed=seed, train=True, normalized=True).astype(np.float32)
    test_features = ds_utils.load_features(dataset, seed=seed, train=False, normalized=True).astype(np.float32)
    train_labels, test_labels = load_labels(dataset)
    pixel_curve = maybe_load_pixel_curve(run_dir)

    rows = []
    for episode_id, episode_dir in load_episode_dirs(run_dir):
        lset = np.load(str(episode_dir / "lSet.npy"), allow_pickle=True).astype(np.int64)
        train_x = train_features[lset]
        train_y = train_labels[lset]
        row = {
            "episode": int(episode_id),
            "labeled_count": int(len(lset)),
            "pixel_test_accuracy": pixel_curve.get(int(episode_id), float("nan")),
        }
        if mode in {"linear", "both"}:
            row["linear_probe_test_accuracy"] = linear_probe_accuracy(
                train_x=train_x,
                train_y=train_y,
                test_x=test_features,
                test_y=test_labels,
                max_epochs=linear_max_iter,
                lr=linear_lr,
                weight_decay=linear_weight_decay,
            )
        if mode in {"knn", "both"}:
            row["knn_test_accuracy"] = knn_accuracy(
                train_x=train_x,
                train_y=train_y,
                test_x=test_features,
                test_y=test_labels,
                k_value=knn_k,
            )
        rows.append(row)

    out_dir = os.path.join(run_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "feature_curve.csv")
    with open(csv_path, "w") as handle:
        headers = sorted(rows[0].keys()) if rows else ["episode"]
        handle.write(",".join(headers) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(key, "")) for key in headers) + "\n")

    summary = {
        "run_dir": run_dir,
        "dataset": dataset,
        "seed": seed,
        "mode": mode,
        "knn_k": int(knn_k),
        "linear_lr": float(linear_lr),
        "linear_weight_decay": float(linear_weight_decay),
        "linear_epochs": int(linear_max_iter),
        "num_points": int(len(rows)),
        "final_pixel_test_accuracy": float(rows[-1].get("pixel_test_accuracy", float("nan"))) if rows else float("nan"),
        "final_linear_probe_test_accuracy": float(rows[-1].get("linear_probe_test_accuracy", float("nan"))) if rows and "linear_probe_test_accuracy" in rows[-1] else float("nan"),
        "final_knn_test_accuracy": float(rows[-1].get("knn_test_accuracy", float("nan"))) if rows and "knn_test_accuracy" in rows[-1] else float("nan"),
    }
    with open(os.path.join(out_dir, "feature_curve_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    if rows:
        mpl_dir = os.path.join(run_dir, output_subdir, ".mpl_cache")
        os.makedirs(mpl_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = [row["episode"] for row in rows]
        plt.figure(figsize=(8, 5))
        if any(np.isfinite(row.get("pixel_test_accuracy", float("nan"))) for row in rows):
            plt.plot(x, [row.get("pixel_test_accuracy", float("nan")) for row in rows], label="pixel_resnet18")
        if mode in {"linear", "both"}:
            plt.plot(x, [row.get("linear_probe_test_accuracy", float("nan")) for row in rows], label="simclr_linear_probe")
        if mode in {"knn", "both"}:
            plt.plot(x, [row.get("knn_test_accuracy", float("nan")) for row in rows], label=f"simclr_knn_k{int(knn_k)}")
        plt.xlabel("AL round")
        plt.ylabel("Test accuracy")
        plt.title(f"{dataset} feature-space evaluation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_curve.png"), dpi=200)
        plt.close()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Post-hoc SimCLR linear-probe / kNN evaluation on saved AL trajectories.")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="One or more existing AL run directories.")
    parser.add_argument("--mode", choices=["linear", "knn", "both"], default="both")
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--linear_lr", type=float, default=1e-2)
    parser.add_argument("--linear_weight_decay", type=float, default=1e-4)
    parser.add_argument("--linear_max_iter", type=int, default=200)
    parser.add_argument("--output_subdir", type=str, default="simclr_feature_eval")
    args = parser.parse_args()

    summaries = []
    for run_dir in args.run_dirs:
        summaries.append(
            evaluate_run(
                run_dir=run_dir,
                mode=args.mode,
                knn_k=args.knn_k,
                linear_lr=args.linear_lr,
                linear_weight_decay=args.linear_weight_decay,
                linear_max_iter=args.linear_max_iter,
                output_subdir=args.output_subdir,
            )
        )
        print(f"Finished feature evaluation for {run_dir}")

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
