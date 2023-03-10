"""Prompt based image generation, using a stable diffusion model hosted by huggingface."""
from diffusers import StableDiffusionPipeline
from torch import autocast
from tqdm import tqdm
import numpy as np
import torch
import os


def mkdir_if_dne(dir_path: str):
    """Make directory if it doesn't already exist.

    Args:
        dir_path (str): Directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def generate_image(pipe: object, prompt: str, guidance_scale: float):
    """Generate images.

    Args:
        pipe (object): Class object: diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline
        prompt (str): Prompt used to generate image.
        guidance_scale (float): Guidance scales determine how much the model should follow the prompt.  Lower values are
            less reflective of the prompt.

    Returns:
        object: Image object.
    """
    with autocast("cuda"):
        image = pipe(prompt, guidance_scale=guidance_scale).images[0]
        return image


def generate_figure_name(
    prompt: str, fig_path: str, guidance_scale: float, img_extension="png"
) -> str:
    """Generate figure name, with complete path and extension/suffix.

    Args:
        prompt (str): Prompt used to generate image.
        fig_path (str): Path to where images will be saved.
        guidance_scale (float): Guidance scales determine how much the model should follow the prompt.  Lower values are
            less reflective of the prompt.
        img_extension (str, optional): Image extension. Defaults to "png".

    Returns:
        str: Figure name, with extension and complete path.
    """
    figname = prompt.replace(" ", "_")
    return f"{fig_path}/{figname}_guidance_scale_{guidance_scale}.{img_extension}"


def save_out_image(image: object, figname: str):
    """Save image object to disk.

    Args:
        image (object): Class object: PIL.Image.Image
        figname (str): Figure name, with extension and complete path.
    """
    image.save(figname)
    return


def choose_hardware(hardware="cpu"):
    """Set what the model should run on: CPU or GPU.

    Args:
        hardware (str, optional): Hardware the model runs on (options are "cpu" or "gpu"). Defaults to "cpu".

    Returns:
        str or object: Hardware the model runs on.
    """
    hardware = hardware.lower()
    if hardware == "cpu":
        device = torch.device(hardware)
    elif hardware == "gpu":
        device = "cuda"
    else:
        device = torch.device(hardware)
        print("Device hardware was set incorrectly. Defaulting to run on CPU.")
    return device


if __name__ == "__main__":
    # Assign and create dir for storing images.
    sub_folder = "craigy"
    fig_path = f"./figs/{sub_folder}"
    mkdir_if_dne(fig_path)

    # Set image generating prompts.
    prompts = ["favicon of an octopus", "goose god"]
    # prompts = ["a cartoon of a goose that is also a greek god"]

    # Guidance scales determine how much the model should follow the prompt.
    # Lower values are less reflective of the prompt.
    # guidance_scales = np.arange(1,11,1)
    guidance_scales = np.arange(9, 11, 1)

    # Choose model.
    model_id = "CompVis/stable-diffusion-v1-4"

    # Choose hardware model is running on: GPU or CPU training.
    # device = torch.device("cpu") # CPU
    # device = "cuda" # GPU
    device = choose_hardware("cpu")

    # Environment settings.
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, use_auth_token=True
    )  # , torch_dtype=torch.float16, revision="fp16")
    pipe = pipe.to(device)

    for prompt in tqdm(prompts):
        for guidance_scale in tqdm(guidance_scales):
            try:
                image = generate_image(
                    pipe=pipe, prompt=prompt, guidance_scale=guidance_scale
                )
                figname = generate_figure_name(
                    prompt=prompt,
                    fig_path=fig_path,
                    guidance_scale=guidance_scale,
                    img_extension="png",
                )
                save_out_image(image=image, figname=figname)
            except:
                print(f"\n\n\nprpmpt '{prompt}' failed\n\n\n")
