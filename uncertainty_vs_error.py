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
    sample = dataset[0]
    image = torch.tensor(sample['image']).unsqueeze(0).to(device)
    recon_mean, recon_std = model.sample_reconstructions(image, n_samples=20)
    print(recon_mean.shape, recon_std.shape)

if __name__ == "__main__":
    main()