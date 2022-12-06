from typing import Dict, Optional, Tuple
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

# from mindiffusion.unet import NaiveUnet
# from mindiffusion.ddpm import DDPM
from ddpm import DDPM
# from unet import UNet
from naive_unet import NaiveUnet

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/patrickhua/datasets/')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='/home/patrickhua/outputs/')
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    print(f'Using device {args.device}')
    return args


def train_cifar10(
    n_epoch: int = 100, device: str = "cuda", load_pth: Optional[str] = None, batch_size: int = 512, lr: float = 1e-5, data_path: Optional[str] = None, output_dir: str = '../output/'
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)
    # ddpm = DDPM(eps_model=UNet(3), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load("ddpm_cifar.pth"))

    ddpm.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        data_path,
        train=True,
        download=True,
        transform=tf,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    for i in range(n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 32, 32), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)

            image_dir = os.path.join(args.output_dir, 'images')
            os.makedirs(image_dir, exist_ok=True)
            save_image(grid, os.path.join(image_dir, f'ddpm_sample_cifar{i}.png'))

            # save model
            ckpt_dir = os.path.join(args.output_dir, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(ddpm.state_dict(), os.path.join(ckpt_dir, 'ddpm_cifar.pth'))


if __name__ == "__main__":
    args = get_args()
    train_cifar10(
        n_epoch=args.epochs,
        device=args.device,
        data_path=args.data_dir,
        output_dir=args.output_dir,
        lr=args.lr
    )