# SDXL Turbo ComfyUI Workflows

This repo contains the workflows and Gradio UI from the "How to Use SDXL Turbo in Comfy UI for Fast Image Generation" video tutorial.

## Contents
- text_to_image.json: Text-to-image workflow for SDXL Turbo
- image_to_image.json: Image-to-image workflow for SDXL Turbo
- high_res_fix.json: High-res fix workflow to upscale SDXL Turbo images
- app.py: Gradio app for simplified SDXL Turbo UI
- requirements.txt: Required Python packages

## Usage
- Download and set up Comfy UI and start it
- Install dependencies with pip install -r requirements.txt
- Place the SDXL Turbo checkpoint in Comfy UI models folder
- Change `BASE_FOLDER = Path("D:/AI/ComfyUI_windows_portable/ComfyUI")` to your comfyui folder path. *Make sure it points to the **ComfyUI** folder inside the comfyui_portable folder*
- Run python app.py to start the Gradio app on localhost
- Access the web UI to use the simplified SDXL Turbo workflows

Refer to the [video tutorial](https://youtu.be/FUjBB-2qEUM) for detailed guidance on using these workflows and UI.

Credits
Workflows and Gradio app created by [Code Crafters Corner](https://www.youtube.com/@CodeCraftersCorner). Tutorial published on my YouTube channel.
