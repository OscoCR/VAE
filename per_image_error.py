import csv
import os

import torch
from tqdm import tqdm

from tools.arguments import parse_args
from data.datasets import PineappleH5Dataset
from models.vae import VAE


def per_image_mse(model, image, target, n_samples=5):
    recon_mean, _ = model.sample_reconstructions(image, n_samples=n_samples)
    return torch.nn.functional.mse_loss(recon_mean, target).item()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = VAE(
        in_channels=getattr(args, 'in_channels', 3),
        out_channels=getattr(args, 'out_channels', 3),
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path_test, map_location=device))
    model.eval()

    dataset = PineappleH5Dataset(
        args.dataset_path, split='train', crop_size=getattr(args, 'resize_img', 256),
        augment=False, seed=args.seed,
        in_channels=getattr(args, 'in_channels', 3), out_channels=getattr(args, 'out_channels', 3),
    )

    rows = []
    for i in tqdm(range(len(dataset)), desc="Per-image error (train)"):
        sample = dataset[i]
        image = torch.tensor(sample['image']).unsqueeze(0).to(device)
        target = torch.tensor(sample['image_target']).unsqueeze(0).to(device)
        rows.append((i, per_image_mse(model, image, target)))

    os.makedirs(args.output_dir_test, exist_ok=True)
    out_path = os.path.join(args.output_dir_test, "per_image_error_train.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "mse"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
