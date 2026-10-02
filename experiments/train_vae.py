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
from models.modules.discriminator import Discriminator

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
        trainset = PineappleH5Dataset(
            args.dataset_path, split='train', crop_size=crop_size,
            augment=getattr(args, 'augment', False), seed=args.seed,
            in_channels=in_channels, out_channels=out_channels,
            hard_indices_path=getattr(args, 'hard_indices_path', None),
            hard_repeat=getattr(args, 'hard_repeat', 1),
        )
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

def remap_decoder_checkpoint(state_dict):
    # Old VAE_Decoder was itself an nn.Sequential ("decoder.0...decoder.25").
    # It's now an nn.Module with the same layers nested under self.blocks
    # ("decoder.blocks.0...decoder.blocks.23"), plus the new
    # variational_last_layer replacing the old final conv (decoder.25, no
    # equivalent anymore -- dropped here, left for the caller's strict=False
    # to random-init the new variational_last_layer weights instead).
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("decoder.") and not key.startswith("decoder.blocks."):
            suffix = key[len("decoder."):]
            index_str = suffix.split(".", 1)[0]
            if index_str.isdigit():
                if int(index_str) <= 23:
                    remapped[f"decoder.blocks.{suffix}"] = value
                continue  # old numeric index handled (remapped, or dropped if it was the old final conv)
            # non-numeric suffix under "decoder." that isn't "decoder.blocks." --
            # e.g. decoder.variational_last_layer.* from a checkpoint that's
            # already in the new architecture -- pass through unchanged below
        remapped[key] = value
    return remapped

def setup_model_and_optimizer(args):
    model = VAE(
        in_channels=getattr(args, 'in_channels', 3),
        out_channels=getattr(args, 'out_channels', 3),
    ).to(args.device)

    pretrained_path = getattr(args, 'pretrained_vae_checkpoint', None)
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location=args.device)
        checkpoint = remap_decoder_checkpoint(checkpoint)
        result = model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained VAE from {pretrained_path}")
        print(f"Missing keys (new, random-init): {result.missing_keys}")
        print(f"Unexpected keys (ignored, old architecture): {result.unexpected_keys}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    discriminator = Discriminator(
        in_channels=getattr(args, 'out_channels', 3),
        hidden_channels=64,
    ).to(args.device)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    return model, optimizer, discriminator, optimizer_d

def train_step(model, dataloader, optimizer, discriminator, optimizer_d, device, beta_kl_loss, adv_weight=0.01, kl_beta_z=0.001):
    model.train()
    discriminator.train()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    total_loss, total_recon, total_kl, count = 0, 0, 0, 0

    with tqdm(total=len(dataloader.dataset), desc="Training", unit='img') as pbar:
        for batch in dataloader:
            images = batch["image"].to(device)          # encoder input
            targets = batch["image_target"].to(device)  # what the decoder must reconstruct

            #VAE forward pass
            optimizer.zero_grad()
            recon, mu, logvar, mean_z, logvar_z = model(images)
            loss_dict = vae_loss(
                reconstructed=recon,
                original=targets,
                mean=mu,
                logvar=logvar,
                mean_z=mean_z,
                logvar_z=logvar_z,
                kl_beta=beta_kl_loss,
                kl_beta_z=kl_beta_z,
            )
            fake_logits_for_vae = discriminator(recon) #No .detach() here, we want gradients to flow back to the VAE
            loss_adv = bce_loss(fake_logits_for_vae, torch.ones_like(fake_logits_for_vae))
            total_loss_vae = loss_dict["total"] + adv_weight * loss_adv

            total_loss_vae.backward()
            optimizer.step()

            #Discriminator forward pass
            optimizer_d.zero_grad()   # Reset gradients for discriminator
            real_logits = discriminator(targets)
            fake_logits = discriminator(recon.detach()) #detach to avoid backprop through the VAE
            loss_d = bce_loss(real_logits, torch.ones_like(real_logits)) + bce_loss(fake_logits, torch.zeros_like(fake_logits))
            loss_d.backward()
            optimizer_d.step()

            total_loss += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_kl += loss_dict["kl"].item()
            count += 1

            pbar.set_postfix(loss=loss_dict["total"].item())
            pbar.update(images.size(0))

    return total_loss / count, total_recon / count, total_kl / count

def validation_step(model, dataloader, device, kl_beta, kl_beta_z=0.001):
    model.eval()
    total_loss, total_recon, total_kl, total_kl_z, count = 0, 0, 0, 0, 0
    total_psnr, total_ssim = 0, 0
    total_psnr_depth, total_ssim_depth = 0, 0
    has_depth = False

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)          # encoder input
            targets = batch["image_target"].to(device)  # what the decoder must reconstruct
            recon, mu, logvar, mean_z, logvar_z = model(images)
            loss_dict = vae_loss(
                reconstructed=recon,
                original=targets,
                mean=mu,
                logvar=logvar,
                mean_z=mean_z,
                logvar_z=logvar_z,
                kl_beta=kl_beta,
                kl_beta_z=kl_beta_z,
            )
            total_loss += loss_dict["total"].item()
            total_recon += loss_dict["reconstruction"].item()
            total_kl += loss_dict["kl"].item()
            total_kl_z += loss_dict["kl_z"].item()

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
        total_kl_z / count,
    )

def reconstruct_sample(model, dataset, device):
    sample_img = dataset[0]['image']
    sample_img = torch.tensor(sample_img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon, _, _, _, _ = model(sample_img)
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
        recon, _, _, _, _ = model(imgs)

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
    val_loss, val_recon, val_kl, val_psnr, val_ssim, val_psnr_depth, val_ssim_depth, val_kl_z = val_losses

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
        "val/kl_z_loss": val_kl_z,
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
    model, optimizer, discriminator, optimizer_d = setup_model_and_optimizer(args)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        train_losses = train_step(model, trainloader, optimizer, discriminator, optimizer_d, device, args.kl_beta, args.adv_weight, getattr(args, 'kl_beta_z', 0.001))
        val_losses = validation_step(model, valloader, device, args.kl_beta, getattr(args, 'kl_beta_z', 0.001))

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

    # Always keep the literal final-epoch weights too, alongside the best-by-loss
    # one -- some prefer this for generative pipelines downstream, since a lower
    # validation loss doesn't always mean better results at every use case.
    last_ckpt_path = os.path.join(path_to_save_checkpoints, "last.pt")
    torch.save(model.state_dict(), last_ckpt_path)
    print(f"Final model saved to {last_ckpt_path}")

    wandb.finish()
    return model