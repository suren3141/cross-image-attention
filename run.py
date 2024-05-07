import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed

sys.path.append(".")
sys.path.append("..")

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images

from pathlib import Path
import os
import os.path as osp
from glob import glob
from tqdm import tqdm

@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)

def run_single_image(model, cfg: RunConfig):
    latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=cfg)
    model.set_latents(latents_app, latents_struct)
    model.set_noise(noise_app, noise_struct)
    print("Running appearance transfer...")
    images = run_appearance_transfer(model=model, cfg=cfg)
    print("Done.")
    return images



def run(cfg: RunConfig) -> List[Image.Image]:
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)
    model = AppearanceTransferModel(cfg)
    if osp.isdir(cfg.app_image_path):
        output_path = cfg.output_path
        app_images = sorted(glob(osp.join(cfg.app_image_path, "*.png")))
        struct_images = sorted(glob(osp.join(cfg.struct_image_path, "*.png")))

        assert len(app_images) == len(struct_images)
        num_images = len(app_images)

        for app_img, struct_img in tqdm(zip(app_images, struct_images), total=num_images):
            file_name = osp.splitext(osp.split(app_img)[-1])[0]
            cfg.app_image_path = Path(app_img)
            cfg.struct_image_path = Path(struct_img)
            cfg.output_path = output_path / f"{file_name}"
            if not osp.exists(cfg.output_path) : os.mkdir(cfg.output_path)
            run_single_image(model, cfg)
    else:
        run_single_image(model, cfg)

    return None

    




def run_appearance_transfer(model: AppearanceTransferModel, cfg: RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
    images = model.pipe(
        prompt=[cfg.prompt] * 3,
        latents=init_latents,
        guidance_scale=1.0,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator('cuda').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
    ).images
    # Save images
    images[0].save(cfg.output_path / f"out_transfer---seed_{cfg.seed}.png")
    images[1].save(cfg.output_path / f"out_style---seed_{cfg.seed}.png")
    images[2].save(cfg.output_path / f"out_struct---seed_{cfg.seed}.png")
    joined_images = np.concatenate(images[::-1], axis=1)
    Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}.png")
    return images


if __name__ == '__main__':
    main()
