import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from tools.arguments import parse_args
from data.datasets import PineappleDataset
from losses.loss import psnr, ssim

# Import your models
from models.vae import VAE
from models.vqvae import VQVAE
from models.dual_vae import DUALVAE

def prepare_test_data(args):
    """Initializes the test dataset and dataloader."""
    if args.dataset_path.endswith('.h5'):
        # dataset_path points directly at the packed HDF5 file -- splits are
        # precomputed inside it (see PineappleH5Dataset), no path_test_ids needed
        from data.datasets import PineappleH5Dataset
        crop_size = getattr(args, 'resize_img', 256)
        testset = PineappleH5Dataset(
            args.dataset_path, split='test', crop_size=crop_size, augment=False, seed=args.seed
        )
    else:
        testset = PineappleDataset(
            path=args.dataset_path,
            split='test',
            test_txt=args.path_test_ids,
            augment=False,
            seed=args.seed
        )
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    return testloader

def load_model(args, device):
    """Instantiates the correct model and loads the checkpoint."""
    if args.model == "vae":
        model = VAE()
    elif args.model == "vqvae":
        model = VQVAE(
            commitment_cost=args.commitment_cost,
            embedding_dim=args.codebook_dim,
            num_embeddings=args.num_embeddings
        )
    elif args.model == "dualvae":
        model = DUALVAE(
            commitment_cost=args.commitment_cost,
            embedding_dim=args.codebook_dim,
            num_embeddings=args.num_embeddings
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model.to(device)
    
    # Load the trained weights
    if not os.path.exists(args.checkpoint_path_test):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path_test}")
    
    model.load_state_dict(torch.load(args.checkpoint_path_test, map_location=device))
    model.eval()
    
    return model

def generate_inferences(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data & Model
    print(f"Loading test data from {args.path_test_ids}...")
    testloader = prepare_test_data(args)
    
    # Extract the underlying dataset object so we can access its 'images' list
    testset = testloader.dataset
    
    print(f"Loading {args.model} model from {args.checkpoint_path_test}...")
    model = load_model(args, device)

    # 2. Create Output Directory
    os.makedirs(args.output_dir_test, exist_ok=True)
    print(f"Inferences will be saved to: {args.output_dir_test}")

    # 3. Inference Loop
    total_psnr, total_ssim, n_images = 0.0, 0.0, 0
    has_filenames = hasattr(testset, "images")

    with torch.no_grad():
        for batch in tqdm(testloader, desc="Generating Inferences"):
            images = batch["image"].to(device)
            indices = batch["idx"] # Grab the original indices from the batch

            # Forward pass depends on what the model returns
            if args.model == "vae":
                recon, _, _ = model(images)
            elif args.model == "vqvae":
                recon, _, _, _ = model(images)
            elif args.model == "dualvae":
                recon, _, _ = model(images, ablation_mode=-1) # -1 = both branches active, matches train_dualvae.py's model(images) call

            # 4. Save the reconstructed images
            # Clamp to [0, 1] just in case to prevent visual artifacts
            recon = recon.clamp(0, 1)

            # Metrics are computed per-image so a batch with a mix of images
            # doesn't average away a single bad reconstruction
            for i in range(images.size(0)):
                total_psnr += psnr(recon[i], images[i])
                total_ssim += ssim(recon[i], images[i])
                n_images += 1

                # 1. Get the dataset index for this specific image in the batch
                dataset_idx = indices[i].item()

                if has_filenames:
                    # 2. Look up the original file path in the dataset's 'images' list
                    original_path = testset.images[dataset_idx]
                    stem = os.path.splitext(os.path.basename(original_path))[0]
                else:
                    # PineappleH5Dataset has no per-file paths (HDF5 rows, not
                    # files on disk) -- name by dataset index instead
                    stem = f"{dataset_idx:05d}"

                vutils.save_image(recon[i], os.path.join(args.output_dir_test, f"{stem}_recon.png"))
                vutils.save_image(images[i], os.path.join(args.output_dir_test, f"{stem}_original.png"))

    avg_psnr = total_psnr / n_images
    avg_ssim = total_ssim / n_images
    print(f"Inference complete! {n_images} images. Test PSNR={avg_psnr:.2f} dB, Test SSIM={avg_ssim:.4f}")

    metrics_path = os.path.join(args.output_dir_test, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"model={args.model}\n")
        f.write(f"checkpoint={args.checkpoint_path_test}\n")
        f.write(f"n_images={n_images}\n")
        f.write(f"psnr={avg_psnr:.4f}\n")
        f.write(f"ssim={avg_ssim:.4f}\n")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    args = parse_args()
    
    generate_inferences(args)