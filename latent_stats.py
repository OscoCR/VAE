import argparse
import random

import numpy as np
import torch

from tools.arguments import load_config_as_args
from data.datasets import PineappleH5Dataset
from models.vae import VAE


def encoder_stats(model, dataset, device, n_images, seed):
    # Only the encoder matters here (it defines the latent space the generators learn),
    # and it never touched the decoder's variational layer.
    random.seed(seed)
    torch.manual_seed(seed)
    idx = np.random.default_rng(seed).choice(len(dataset), n_images, replace=False)
    sq_mean, post_var, kl, lat_std, spatial = [], [], [], [], []
    with torch.no_grad():
        for i in idx:
            x = torch.tensor(dataset[int(i)]['image']).unsqueeze(0).to(device)
            noise = torch.randn((1, 4, x.shape[2] // 8, x.shape[3] // 8), device=device)
            latent, mean, logvar = model.encoder(x, noise)
            sq_mean.append(mean.pow(2).mean().item())
            post_var.append(logvar.exp().mean().item())
            kl.append((-0.5 * (1 + logvar - mean.pow(2) - logvar.exp())).mean().item())
            lat_std.append(latent.std().item())
            # roughness: mean absolute difference between horizontally adjacent latent cells, relative to latent std
            spatial.append(((latent[..., 1:] - latent[..., :-1]).abs().mean() / latent.std()).item())
    return {
        "rms(mean)": float(np.sqrt(np.mean(sq_mean))),
        "mean posterior std": float(np.mean(np.sqrt(post_var))),
        "SNR rms(mean)/post.std": float(np.sqrt(np.mean(sq_mean)) / np.mean(np.sqrt(post_var))),
        "KL per element": float(np.mean(kl)),
        "latent std (scaled)": float(np.mean(lat_std)),
        "roughness (|d latent| / std)": float(np.mean(spatial)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare encoder/latent statistics across VAE checkpoints")
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', required=True, nargs='+', help="label=path")
    parser.add_argument('--n_images', default=256, type=int)
    args = parser.parse_args()

    cfg = load_config_as_args(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PineappleH5Dataset(
        cfg.dataset_path, split='train', crop_size=getattr(cfg, 'resize_img', 256), augment=False, seed=cfg.seed,
        in_channels=cfg.in_channels, out_channels=cfg.out_channels,
    )

    results = {}
    for entry in args.checkpoints:
        label, path = entry.split('=', 1)
        model = VAE(in_channels=cfg.in_channels, out_channels=cfg.out_channels).to(device)
        ckpt = torch.load(path, map_location=device)
        # load only the encoder: older checkpoints use a different decoder key layout
        enc = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc)
        model.eval()
        results[label] = encoder_stats(model, dataset, device, args.n_images, seed=0)

    labels = list(results)
    print(f"\nEncoder statistics over {args.n_images} train images (same images and noise for every model)\n")
    print(f"{'statistic':32s}" + "".join(f"{l:>16s}" for l in labels))
    for key in results[labels[0]]:
        print(f"{key:32s}" + "".join(f"{results[l][key]:16.4f}" for l in labels))


if __name__ == "__main__":
    main()
