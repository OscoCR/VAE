import os
import torch
import wandb
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from tools.utils import *
from data.datasets import PineappleDataset
from models.vae import VAE
from losses.loss import vae_loss, psnr, ssim

import torchvision.utils as vutils

# ---- helpers.py ----

def get_dataloaders(args):
    generator = torch.Generator().manual_seed(args.seed)

    if args.dataset_path.endswith('.h5'):
        # dataset_path points directly at the packed HDF5 file -- splits are
        # precomputed inside it (see PineappleH5Dataset), no test_txt needed
        from data.datasets import PineappleH5Dataset
        crop_size = getattr(args, 'resize_img', 256)
        in_channels = getattr(args, 'in_channels', 3)
        out_channels = getattr(args, 'out_channels', 3)
        trainset = PineappleH5Dataset(args.dataset_path, split='train', crop_size=crop_size, augment=False, seed=args.seed, in_channels=in_channels, out_channels=out_channels)
        valset = PineappleH5Dataset(args.dataset_path, split='val', crop_size=crop_size, augment=False, seed=args.seed, in_channels=in_channels, out_channels=out_channels)
    else:
        trainset = PineappleDataset(
            path=args.dataset_path,
            split='train', test_txt=args.path_test_ids, augment=False, seed=args.seed
        )
        valset = PineappleDataset(
            path=args.dataset_path,
            split='val', test_txt=args.path_test_ids, augment=False, seed=args.seed
        )

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return trainset, valset, trainloader, valloader

def setup_model_and_optimizer(args):
    model = VAE(
        in_channels=getattr(args, 'in_channels', 3),
        out_channels=getattr(args, 'out_channels', 3),
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer

def train_step(model, dataloader, optimizer, device, beta_kl_loss):
    model.train()
    total_loss, total_recon, total_kl, count = 0, 0, 0, 0

    with tqdm(total=len(dataloader.dataset), desc="Training", unit='img') as pbar:
        for batch in dataloader:
            images = batch["image"].to(device)          # encoder input
            targets = batch["image_target"].to(device)  # what the decoder must reconstruct
            optimizer.zero_grad()
            recon, mu, logvar = model(images)

            loss_dict = vae_loss(recon, targets, mu, logvar, kl_beta=beta_kl_loss)
            loss_dict["total"].backward()
            optimizer.step()

            total_loss += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_kl += loss_dict["kl"].item()
            count += 1

            pbar.set_postfix(loss=loss_dict["total"].item())
            pbar.update(images.size(0))

    return total_loss / count, total_recon / count, total_kl / count

def validation_step(model, dataloader, device, kl_beta):
    model.eval()
    total_loss, total_recon, total_kl, count = 0, 0, 0, 0
    total_psnr, total_ssim = 0, 0
    total_psnr_depth, total_ssim_depth = 0, 0
    has_depth = False

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)          # encoder input
            targets = batch["image_target"].to(device)  # what the decoder must reconstruct
            recon, mu, logvar = model(images)

            loss_dict = vae_loss(recon, targets, mu, logvar, kl_beta=kl_beta)
            total_loss += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_kl += loss_dict["kl"].item()

            # RGB and Depth are different modalities -- never mix them into one
            # PSNR/SSIM number (same reasoning as generate_test_inferences.py).
            # Based on the target (what recon is graded against), not the input --
            # a model fed RGB+Depth but trained to output only RGB has no depth to grade.
            has_depth = targets.shape[1] == 4
            recon_rgb, img_rgb = recon[:, :3], targets[:, :3]
            total_psnr += psnr(recon_rgb, img_rgb)
            total_ssim += ssim(recon_rgb, img_rgb)
            if has_depth:
                recon_depth, img_depth = recon[:, 3:4], targets[:, 3:4]
                total_psnr_depth += psnr(recon_depth, img_depth)
                total_ssim_depth += ssim(recon_depth, img_depth)

            count += 1

    return (
        total_loss / count,
        total_recon / count,
        total_kl / count,
        total_psnr / count,
        total_ssim / count,
        total_psnr_depth / count if has_depth else None,
        total_ssim_depth / count if has_depth else None,
    )

def reconstruct_sample(model, dataset, device):
    sample_img = dataset[0]['image']
    sample_img = torch.tensor(sample_img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _, _ = model(sample_img)
        recon = recon.squeeze(0).cpu().numpy()
        recon = np.transpose(recon, (1, 2, 0)) * 255
    return recon.astype(np.uint8)

def reconstruct_grid(model, dataset, device, n_samples=8):
    model.eval()
    idxs = np.random.choice(len(dataset), n_samples, replace=False)
    samples = [dataset[i] for i in idxs]
    imgs = torch.tensor(np.stack([s["image"] for s in samples])).to(device)           # encoder input
    targets = torch.tensor(np.stack([s["image_target"] for s in samples])).to(device)  # what recon should match

    with torch.no_grad():
        recon, _, _ = model(imgs)

    # RGB and Depth are different modalities -- grid them separately so Depth
    # isn't silently read as an alpha channel on top of RGB (same issue as
    # generate_test_inferences.py before it was split by modality). Based on
    # the target, since that's what recon is actually graded against.
    has_depth = targets.shape[1] == 4
    targets_rgb, recon_rgb = targets[:, :3], recon[:, :3]

    # Denormalize if needed (here assume already in [0,1])
    grid_rgb = vutils.make_grid(torch.cat([targets_rgb, recon_rgb], dim=0), nrow=n_samples, normalize=True, scale_each=True)

    grid_depth = None
    if has_depth:
        targets_depth, recon_depth = targets[:, 3:4], recon[:, 3:4]
        grid_depth = vutils.make_grid(torch.cat([targets_depth, recon_depth], dim=0), nrow=n_samples, normalize=True, scale_each=True)

    return grid_rgb, grid_depth

def log_metrics_to_wandb(epoch, train_losses, val_losses, recon_grid, recon_grid_depth=None):
    train_loss, train_recon, train_kl = train_losses
    val_loss, val_recon, val_kl, val_psnr, val_ssim, val_psnr_depth, val_ssim_depth = val_losses

    log_dict = {
        "epoch": epoch,
        "Sample Reconstructions": wandb.Image(recon_grid, caption=f"Epoch {epoch}"),
        "train/total_loss": train_loss,
        "train/recon_loss": train_recon,
        "train/kl_loss": train_kl,
        "val/total_loss": val_loss,
        "val/recon_loss": val_recon,
        "val/kl_loss": val_kl,
        "val/psnr": val_psnr,
        "val/ssim": val_ssim,
    }
    if recon_grid_depth is not None:
        log_dict["Sample Reconstructions (Depth)"] = wandb.Image(recon_grid_depth, caption=f"Epoch {epoch}")
    if val_psnr_depth is not None:
        log_dict["val/psnr_depth"] = val_psnr_depth
        log_dict["val/ssim_depth"] = val_ssim_depth

    wandb.log(log_dict, step=epoch)


def save_if_best_val(model, loss, best_loss, path, epoch):
    min_delta = 1e-6
    if loss < best_loss - min_delta:
        torch.save(model.state_dict(), os.path.join(path, f"best.pt"))
        print(f"Checkpoint saved at epoch {epoch}.")
        return loss, True
    else:
        print("No improvement in loss.")
        return best_loss, False

# ---- train_vae.py ----

def train_vae(args):
    # Device & seed setup
    device = select_device(args.device)
    set_seed(args.seed, args.deterministic, args.cudnn_benchmark)

    path_to_save_checkpoints = os.path.join(args.checkpoints, f"betaKL@{args.kl_beta}")
    create_directory(path_to_save_checkpoints)
    
    setup_wandb(args)

    trainset, valset, trainloader, valloader = get_dataloaders(args)
    model, optimizer = setup_model_and_optimizer(args)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_losses = train_step(model, trainloader, optimizer, device, args.kl_beta)
        val_losses = validation_step(model, valloader, device, args.kl_beta)

        depth_msg = f", PSNR(D)={val_losses[5]:.2f}, SSIM(D)={val_losses[6]:.3f}" if val_losses[5] is not None else ""
        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_losses[0]:.4f}, Recon={train_losses[1]:.4f}, KL={train_losses[2]:.4f} | "
            f"Val Loss={val_losses[0]:.4f}, Recon={val_losses[1]:.4f}, KL={val_losses[2]:.4f}, "
            f"PSNR={val_losses[3]:.2f}, SSIM={val_losses[4]:.3f}{depth_msg}"
        )

        # Reconstruct and log
        recon_grid, recon_grid_depth = reconstruct_grid(model, valset, args.device, n_samples=8)
        log_metrics_to_wandb(epoch, train_losses, val_losses, recon_grid, recon_grid_depth)

        # Checkpoint and early stopping
        best_val_loss, improved = save_if_best_val(model, val_losses[0], best_val_loss, path_to_save_checkpoints, epoch)
        patience_counter = 0 if improved else patience_counter + 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    wandb.finish()
    return model