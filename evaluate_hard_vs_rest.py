import argparse
import csv
import os

import numpy as np
import torch
from scipy.stats import wilcoxon

from tools.arguments import load_config_as_args
from data.datasets import PineappleH5Dataset
from models.vae import VAE


def per_image_mse(model, dataset, device, n_samples, seed):
    # Test crops are deterministic (center crop), so differences between models
    # come from the weights, not from which window was cropped.
    torch.manual_seed(seed)
    errors = []
    for i in range(len(dataset)):
        sample = dataset[i]
        image = torch.tensor(sample['image']).unsqueeze(0).to(device)
        target = torch.tensor(sample['image_target']).unsqueeze(0).to(device)
        recon_mean, _ = model.sample_reconstructions(image, n_samples=n_samples)
        errors.append(torch.nn.functional.mse_loss(recon_mean, target).item())
    return np.array(errors)


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints on hard vs. rest test images")
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--checkpoints', required=True, nargs='+', help="label=path, in order")
    parser.add_argument('--baseline', required=True, type=str, help="label whose errors define the hard set")
    parser.add_argument('--control', required=True, type=str, help="label every other model is compared against")
    parser.add_argument('--top_fraction', default=0.25, type=float)
    parser.add_argument('--n_samples', default=20, type=int)
    parser.add_argument('--out_dir', required=True, type=str)
    args = parser.parse_args()

    cfg = load_config_as_args(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PineappleH5Dataset(
        cfg.dataset_path, split='test', crop_size=getattr(cfg, 'resize_img', 256),
        augment=False, seed=cfg.seed,
        in_channels=getattr(cfg, 'in_channels', 3), out_channels=getattr(cfg, 'out_channels', 3),
    )

    errors = {}
    for entry in args.checkpoints:
        label, path = entry.split('=', 1)
        model = VAE(
            in_channels=getattr(cfg, 'in_channels', 3),
            out_channels=getattr(cfg, 'out_channels', 3),
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        errors[label] = per_image_mse(model, dataset, device, args.n_samples, seed=0)
        print(f"{label}: mean test MSE = {errors[label].mean():.6f}", flush=True)

    n = len(dataset)
    n_hard = int(n * args.top_fraction)
    hard = np.zeros(n, dtype=bool)
    hard[np.argsort(-errors[args.baseline])[:n_hard]] = True
    print(f"\nHard set = top {args.top_fraction:.0%} test images by '{args.baseline}' error: {n_hard} of {n}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "test_per_image_mse.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "is_hard"] + list(errors))
        for i in range(n):
            writer.writerow([i, int(hard[i])] + [f"{errors[l][i]:.8f}" for l in errors])

    ctrl = errors[args.control]
    print(f"\nMean test MSE (x1e-3) -- every model vs. control '{args.control}'")
    print(f"{'model':16s} {'all':>8s} {'hard':>8s} {'rest':>8s} | {'hard vs ctrl':>13s} {'p':>9s} | {'rest vs ctrl':>13s} {'p':>9s}")
    for label, e in errors.items():
        row = f"{label:16s} {e.mean()*1e3:8.4f} {e[hard].mean()*1e3:8.4f} {e[~hard].mean()*1e3:8.4f} |"
        if label == args.control:
            row += f" {'(control)':>13s} {'':>9s} | {'(control)':>13s} {'':>9s}"
        else:
            for mask in (hard, ~hard):
                diff = e[mask] - ctrl[mask]
                rel = 100 * diff.mean() / ctrl[mask].mean()
                p = wilcoxon(e[mask], ctrl[mask]).pvalue
                row += f" {rel:+12.2f}% {p:9.2g} |"
            row = row.rstrip('|')
        print(row)


if __name__ == "__main__":
    main()
