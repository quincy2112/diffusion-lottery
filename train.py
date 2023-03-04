import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import os

import time

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from datasets import load_dataset
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from torchinfo import summary

from scheduler import DDIMScheduler
from model import UNet
from utils import save_images, normalize_to_neg_one_to_one, plot_losses
from datetime import datetime

n_timesteps = 1000
n_inference_timesteps = 50


def main(args):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    model = UNet(3, image_size=args.resolution, hidden_dims=[16, 32, 64, 128])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                    beta_schedule="cosine", device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    augmentations = Compose([
        Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(args.resolution),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder",
                                data_dir=args.train_data_dir,
                                cache_dir=args.cache_dir,
                                split="train")

    def transforms(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    dataset.set_transform(transforms)

    train_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True)
    
    fid_dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=4096,
                                                   shuffle=False,
                                                    )
    real_image_samples = next(iter(train_dataloader))['input']

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
        
    log_filename = "logs/log_" + str(time.time_ns() // 1_000_000) + ".csv"
    with open(log_filename, "a") as file:
        file.write("epoch,fid\n")

    model = model.to(device)
    summary(model, [(1, 3, args.resolution, args.resolution), (1,)], verbose=1)

    global_step = 0
    losses = []
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        losses_log = 0
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"].to(device)
            clean_images = normalize_to_neg_one_to_one(clean_images)
            
            batch_size = clean_images.shape[0]
            noise = torch.randn(clean_images.shape).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, 
                                     (batch_size,), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps)["sample"]
            loss = F.l1_loss(noise_pred, noise)
            loss.backward()

            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            losses_log += loss.detach().item()
            logs = {
                "loss_avg": losses_log / (step + 1),
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }

            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()
        losses.append(losses_log / (step + 1))

        # Generate sample images for visual inspection
        if epoch % args.save_model_epochs == 0:
            with torch.no_grad():
                # has to be instantiated every time, because of reproducibility
                generator = torch.manual_seed(0)
                generated_images = noise_scheduler.generate(
                    model,
                    num_inference_steps=n_inference_timesteps,
                    generator=generator,
                    eta=1.0,
                    use_clipped_model_output=True,
                    batch_size=args.eval_batch_size,
                    output_type="numpy")
                
                fid = FrechetInceptionDistance(feature=64, normalize=False)
                
                fid.update(torch.tensor(256*np.moveaxis(generated_images['sample'], -1, 1)).to(torch.uint8), real=False)
                fid.update((256*real_image_samples).to(torch.uint8), real=True)
                
                with open(log_filename, "a") as file:
                    file.write(str(epoch) + "," + str(fid.compute().item()) + "\n")
                
                save_images(generated_images, epoch, args)
                plot_losses(losses, f"{args.loss_logs_dir}_{timestamp}/{epoch}/")

                if not os.path.exists("trained_models"):
                    os.mkdir("trained_models")
        
                torch.save(
                    {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir",
                        type=str,
                        default=None,
                        help="A folder containing the training data.")
    parser.add_argument("--output_dir", type=str, default="trained_models/ddpm-model-64.pth")
    parser.add_argument("--samples_dir", type=str, default="test_samples/")
    parser.add_argument("--loss_logs_dir", type=str, default="training_logs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_model_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")


    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory.")

    main(args)