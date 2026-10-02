import torch

from tools.arguments import parse_args
from data.datasets import PineappleH5Dataset
from models.vae import VAE
from scipy.stats import spearmanr



def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mses = []
    std_means = []
    std_maxs = []

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
    for i in range(len(dataset)):
        sample = dataset[i]
        image = torch.tensor(sample['image']).unsqueeze(0).to(device)
        target = torch.tensor(sample['image_target']).unsqueeze(0).to(device)
        mse, std_mean, std_max = per_image_stats(model, image, target)
        mses.append(mse)
        std_means.append(std_mean)
        std_maxs.append(std_max)
    corr, p_value = spearmanr(mses, std_means)
    corr_max, p_value_max = spearmanr(mses, std_maxs)
    print(f"Imagenes evaluadas: {len(mses)}")
    print(f"mse vs std_mean: spearman={corr:.4f}  p-value={p_value:.4g}")
    print(f"mse vs std_max : spearman={corr_max:.4f}  p-value={p_value_max:.4g}")
    return mses, std_means, std_maxs
    
        
def per_image_stats(model, image, target, n_samples=20):
    recon_mean, recon_std = model.sample_reconstructions(image, n_samples=n_samples)
    mse = torch.nn.functional.mse_loss(recon_mean, target).item()
    std_mean = torch.mean(recon_std).item()
    std_max = torch.max(recon_std).item()
    return mse, std_mean, std_max

if __name__ == "__main__":
    main()