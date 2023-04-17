import os
from datetime import datetime

from PIL import Image
import torch
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt


def plot_losses(losses, out_dir):
    os.makedirs(out_dir)
    plt.plot(losses, label='train')
    plt.legend()
    plt.savefig(f"{out_dir}/losses.png")
    plt.clf()


def save_images(generated_images, epoch, args, title=""):
    images = generated_images["sample"]
    images_processed = (images * 255).round().astype("uint8")
    
    out_dir = f"./final_images/"
    os.makedirs(out_dir)
    for idx, image in enumerate(images_processed):
        image = Image.fromarray(image)
        image.save(f"{out_dir}/{title}_{idx}.png")

    # utils.save_image(generated_images["sample_pt"],
    #                     f"{out_dir}/{title}",
    #                     nrow=args.eval_batch_size // 4)


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()

    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)

    return values


def clip(tensor, min_value=None, max_value=None):
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    elif isinstance(tensor, torch.Tensor):
        return torch.clamp(tensor, min_value, max_value)

    raise ValueError("Tensor format is not valid is not valid - " \
        f"should be numpy array or torch tensor. Got {type(tensor)}.")
