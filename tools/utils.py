import os
import wandb
import torch
import numpy as np
import random

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

def setup_wandb(args):
    """Login to Weights & Biases and initialize a new run."""

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    # Without an explicit name, wandb assigns a random one (e.g. "playful-sunrise-3"),
    # making runs indistinguishable on the dashboard without opening each one's config.
    # Encode the variables that actually change between experiments (e.g. RGB vs RGB+Depth).
    model_name = getattr(args, 'model', 'model')
    kl_beta = getattr(args, 'kl_beta', None)
    in_channels = getattr(args, 'in_channels', 3)
    out_channels = getattr(args, 'out_channels', 3)
    run_name = f"{model_name}_klbeta{kl_beta}" if kl_beta is not None else model_name
    if in_channels != 3 or out_channels != 3:
        run_name += f"_in{in_channels}out{out_channels}"

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        config=vars(args),  # Include all args dynamically
    )
    return run

def select_device(cfg_device: str = "cuda") -> str:
    if cfg_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[Device] CUDA not available, falling back to CPU.")
    return device


def set_seed(seed: int = 42, deterministic: bool = True, cudnn_benchmark: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = cudnn_benchmark

    print(f"[Seed] {seed} | deterministic={deterministic} | cudnn.benchmark={torch.backends.cudnn.benchmark}")


def seed_worker(worker_id):
    # Ensure each worker has a different but reproducible seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    