import json
import os
import random
import time
from pathlib import Path

import gradio as gr
import requests

from PIL import Image


BASE_FOLDER = Path("D:/AI/ComfyUI_windows_portable/ComfyUI")
MODELS_FOLDER = BASE_FOLDER / "models/checkpoints"
UPSCALE_MODELS_FOLDER = BASE_FOLDER / "models/upscale_models"
INPUT_DIR = BASE_FOLDER / "input"
OUTPUT_DIR = BASE_FOLDER / "output"
URL = "http://127.0.0.1:8188/prompt"
TXT2IMG = "workflow_api_text2img.json"
IMG2IMG = "workflow_api_img2img.json"
HIRES = "workflow_api_HiRes.json"

KSAMPLER_NAMES = [
    "euler",
    "euler_ancestral",
    "dpmpp_sde",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_3m_sde",
    "ddpm",
    "lcm",
]

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]


def checkpoint_list():
    return [model.name for model in MODELS_FOLDER.glob("*.safetensors")]


def upscale_checkpoint_list():
    return [model.name for model in UPSCALE_MODELS_FOLDER.glob("*.pth")]


def get_latest_image():
    image_files = list(OUTPUT_DIR.glob("*.png"))
    image_files.sort(key=lambda x: os.path.getmtime(x))
    latest_image = image_files[-1] if image_files else None
    return latest_image


def start_queue(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode("utf-8")
    requests.post(URL, data=data)


def txt2img_workflow(checkpoint_name, positive_prompt):
    with open(TXT2IMG, "r") as file_json:
        prompt = json.load(file_json)

    prompt["1"]["inputs"]["ckpt_name"] = checkpoint_name
    prompt["2"]["inputs"]["noise_seed"] = random.randint(1, 999999999999)
    prompt["3"]["inputs"]["text"] = positive_prompt

    previous_image = get_latest_image()

    start_queue(prompt)

    while True:
        latest_image = get_latest_image()
        if latest_image != previous_image:
            return latest_image

        time.sleep(1)


def img2img_workflow(checkpoint_name, positive_prompt, input_image, denoising_strength):
    with open(IMG2IMG, "r") as file_json:
        prompt = json.load(file_json)

    prompt["1"]["inputs"]["ckpt_name"] = checkpoint_name
    prompt["3"]["inputs"]["text"] = positive_prompt
    prompt["14"]["inputs"]["seed"] = random.randint(1, 999999999999)
    prompt["14"]["inputs"]["denoise"] = denoising_strength

    image = Image.fromarray(input_image)
    min_side = min(image.size)
    scale_factor = 512 / min_side
    new_size = (round(image.size[0] * scale_factor), round(image.size[1] * scale_factor))
    resized_image = image.resize(new_size)

    resized_image.save(str(INPUT_DIR / "input_api.jpg"))

    previous_image = get_latest_image()

    start_queue(prompt)

    while True:
        latest_image = get_latest_image()
        if latest_image != previous_image:
            return latest_image

        time.sleep(1)


def hires_workflow(
    checkpoint_name,
    positive_prompt,
    negative_prompt,
    input_image,
    upscale_model,
    downscale,
    steps,
    cfg,
    denoising_strength,
    sampler_name,
    scheduler,
):
    with open(HIRES, "r") as file_json:
        prompt = json.load(file_json)

    prompt["22"]["inputs"]["ckpt_name"] = checkpoint_name
    prompt["24"]["inputs"]["text"] = positive_prompt
    prompt["25"]["inputs"]["text"] = negative_prompt
    prompt["19"]["inputs"]["model_name"] = upscale_model
    prompt["32"]["inputs"]["scale_by"] = downscale
    prompt["23"]["inputs"]["seed"] = random.randint(1, 999999999999)
    prompt["23"]["inputs"]["steps"] = steps
    prompt["23"]["inputs"]["cfg"] = cfg
    prompt["23"]["inputs"]["denoise"] = denoising_strength
    prompt["23"]["inputs"]["sampler_name"] = sampler_name
    prompt["23"]["inputs"]["scheduler"] = scheduler

    image = Image.fromarray(input_image)
    min_side = min(image.size)
    scale_factor = 512 / min_side
    new_size = (round(image.size[0] * scale_factor), round(image.size[1] * scale_factor))
    resized_image = image.resize(new_size)

    resized_image.save(str(INPUT_DIR / "input_api.jpg"))

    previous_image = get_latest_image()

    start_queue(prompt)

    while True:
        latest_image = get_latest_image()
        if latest_image != previous_image:
            return latest_image

        time.sleep(1)


def main():
    with gr.Blocks() as demo:
        models = checkpoint_list()
        upscale_models = upscale_checkpoint_list()

        with gr.Row():
            with gr.Column():
                base_checkpoint = gr.Dropdown(choices=models, label="Stable Diffusion Checkpoint")

        with gr.Tab("txt2img"):
            workflow = TXT2IMG
            with gr.Row():
                with gr.Column(scale=3):
                    positive = gr.Textbox(lines=4, placeholder="Positive prompt", container=False)

                with gr.Column(scale=1):
                    generate_btn_txt2img = gr.Button("Generate")

            with gr.Row():
                output_img = gr.Image(label="Output", interactive=False)

        generate_btn_txt2img.click(fn=txt2img_workflow, inputs=[base_checkpoint, positive], outputs=output_img)

        with gr.Tab("img2img"):
            with gr.Row():
                with gr.Column(scale=3):
                    positive = gr.Textbox(lines=4, placeholder="Positive prompt", container=False)

                with gr.Column(scale=1):
                    generate_btn_img2img = gr.Button("Generate")

            with gr.Row():
                with gr.Column():
                    denoising_strength = gr.Slider(
                        minimum=0.00, maximum=1.00, value=0.50, step=0.01, label="Denoising Strength", interactive=True
                    )
                    input_img = gr.Image(label="Input Image")
                with gr.Column():
                    output_img = gr.Image(label="Output", interactive=False)

        generate_btn_img2img.click(
            fn=img2img_workflow, inputs=[base_checkpoint, positive, input_img, denoising_strength], outputs=output_img
        )

        with gr.Tab("HiRes"):
            with gr.Row():
                with gr.Column(scale=3):
                    positive = gr.Textbox(lines=2, placeholder="Positive prompt", container=False)
                    negative = gr.Textbox(lines=2, placeholder="Negative prompt", container=False)

                with gr.Column(scale=1):
                    generate_btn_hires = gr.Button("Generate")

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column(scale=2):
                            upscale_model = gr.Dropdown(choices=upscale_models, label="Upscale Model")
                        with gr.Column(scale=1):
                            downscale_factor = gr.Slider(
                                minimum=0.25,
                                maximum=1.00,
                                value=0.50,
                                step=0.25,
                                label="Downscale factor",
                                interactive=True,
                            )
                    with gr.Column():
                        input_img = gr.Image(label="Input Image")
                        denoising_strength = gr.Slider(
                            minimum=0.00,
                            maximum=1.00,
                            value=0.50,
                            step=0.01,
                            label="Denoising Strength",
                            interactive=True,
                        )
                        step_count = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Steps", interactive=True)
                        cfg = gr.Slider(minimum=0.0, maximum=20.0, value=6, step=0.01, label="CFG", interactive=True)
                        sampler_name = gr.Dropdown(choices=KSAMPLER_NAMES, label="Sampler")
                        scheduler = gr.Dropdown(choices=SCHEDULER_NAMES, label="Scheduler")
                with gr.Column():
                    output_img = gr.Image(label="Output", interactive=False)

        generate_btn_hires.click(
            fn=hires_workflow,
            inputs=[
                base_checkpoint,
                positive,
                negative,
                input_img,
                upscale_model,
                downscale_factor,
                step_count,
                cfg,
                denoising_strength,
                sampler_name,
                scheduler,
            ],
            outputs=output_img,
        )

    demo.launch()


if __name__ == "__main__":
    main()
