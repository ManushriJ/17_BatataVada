import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionInpaintPipeline
from PIL import Image
import gradio as gr

# ✅ Load Stable Diffusion Text-to-Image Model
text2img_pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16, 
    use_safetensors=True
)
text2img_pipeline.to("mps")  
text2img_pipeline.enable_attention_slicing()

# ✅ Load Stable Diffusion Inpainting Model
inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)
inpaint_pipeline.to("mps")  

# ✅ Function to Generate an Image from Prompt
def generate_image(prompt):
    m_prompt = f"white background, front facing, {prompt}"
    generated_image = text2img_pipeline(m_prompt).images[0]  
    generated_image.save("generated_image.png")  
    return generated_image  

# ✅ Function to Apply Inpainting Modification
def modify_image(original_image, mask_image, modification_prompt):
    try:
        # Run inpainting model
        output = inpaint_pipeline(
            prompt=modification_prompt, 
            image=original_image, 
            mask_image=mask_image
        ).images[0]

        # Save & return modified image
        output.save("modified_output.png")
        return output

    except Exception as e:
        return f"Error: {e}"

# ✅ Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 AI Image Generator & Modifier")

    with gr.Row():
        prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Describe the image you want...")
        generate_button = gr.Button("Generate Image")

    with gr.Row():
        generated_image_output = gr.Image(label="Generated Image")  

    with gr.Row():
        gr.Markdown("### 🖌 Upload images for modification")

    with gr.Row():
        original_image_input = gr.Image(label="Upload Original Image", type="pil")
        mask_image_input = gr.Image(label="Upload Mask Image", type="pil")

    with gr.Row():
        modification_prompt_input = gr.Textbox(label="Enter modification description", placeholder="Describe the changes...")
        modify_button = gr.Button("Apply Modification")

    modified_output = gr.Image(label="Modified Image")  

    # ✅ Button Click Actions
    generate_button.click(generate_image, inputs=prompt_input, outputs=generated_image_output)  
    modify_button.click(
        modify_image, 
        inputs=[original_image_input, mask_image_input, modification_prompt_input], 
        outputs=modified_output
    )

# ✅ Launch Gradio App
demo.launch(share=True)
