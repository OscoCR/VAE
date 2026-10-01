import torch

from tools.arguments import parse_args
from data.datasets import PineappleH5Dataset
from models.vae import VAE


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = VAE(
        in_channels=getattr(args, 'in_channels', 3),
        out_channels=getattr(args, 'out_channels', 3),
    ).to(device)
    ckpt = torch.load(args.checkpoint_path_test, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    dataset = PineappleH5Dataset(
        args.dataset_path, split='test', crop_size=getattr(args, 'resize_img', 256),
        augment=False, seed=args.seed,
        in_channels=getattr(args, 'in_channels', 3), out_channels=getattr(args, 'out_channels', 3),
    )

    n_images = 8
    n_samples = 20
    print(f"Checking uncertainty on {n_images} real test images, {n_samples} stochastic passes each\n")

    for idx in range(n_images):
        image = torch.tensor(dataset[idx]['image']).unsqueeze(0).to(device)
        recon_mean, recon_std = model.sample_reconstructions(image, n_samples=n_samples)
        std = recon_std[0]  # (C, H, W)
        print(
            f"image {idx:2d}: std mean={std.mean().item():.6f}  "
            f"std min={std.min().item():.6f}  std max={std.max().item():.6f}  "
            f"std range (max-min)={std.max().item() - std.min().item():.6f}"
        )


if __name__ == "__main__":
    main()
