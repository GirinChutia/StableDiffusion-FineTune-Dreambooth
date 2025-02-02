import argparse
import torch
from PIL import Image
from diffusers import DiffusionPipeline, AutoencoderKL

def image_grid(imgs, rows, cols, resize=1024, spacing=10, bg_color=(255, 255, 255)):
    """
    Create a grid of images with specified rows and cols, 
    optionally resizing each image and spacing them out.

    :param imgs:    List of PIL Image objects.
    :param rows:    Number of rows.
    :param cols:    Number of columns.
    :param resize:  If not None, resize all images to (resize, resize).
    :param spacing: The number of pixels between images.
    :param bg_color:Background color as an (R, G, B) tuple.
    :return:        A single PIL Image containing the grid.
    """
    assert len(imgs) == rows * cols, (
        f"Expected exactly rows*cols={rows*cols} images, but got {len(imgs)}."
    )

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]

    w, h = imgs[0].size
    grid_width = cols * w + (cols - 1) * spacing
    grid_height = rows * h + (rows - 1) * spacing

    grid = Image.new("RGB", (grid_width, grid_height), color=bg_color)

    for i, img in enumerate(imgs):
        r = i // cols
        c = i % cols
        x = c * (w + spacing)
        y = r * (h + spacing)
        grid.paste(img, box=(x, y))

    return grid

def main():
    parser = argparse.ArgumentParser(description="Generate AI images using Stable Diffusion XL with LoRA fine-tuning.")
    
    parser.add_argument("--prompt", type=str, required=True, help="The text prompt for image generation.")
    parser.add_argument("--negative_prompt", type=str, default="distorted, grainy, artifacts, out of focus, low quality, blurry, cartoonish, unrealistic, low-resolution, motion blur, people, top view, bad wheels", help="Negative prompt to avoid unwanted elements.")
    parser.add_argument("--num_steps", type=int, default=25, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=7.7, help="Guidance scale for generation.")
    parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate.")
    parser.add_argument("--rows", type=int, default=2, help="Number of rows in output image grid.")
    parser.add_argument("--cols", type=int, default=2, help="Number of columns in output image grid.")
    parser.add_argument("--output", type=str, default="output_custom_model.png", help="Output filename.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Pretrained model ID.")
    parser.add_argument("--lora_repo", type=str, default="Girin67/train_sdxl", help="LoRA model repository ID.")
    parser.add_argument("--vae_id", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="VAE model ID.")

    args = parser.parse_args()

    print("Loading VAE model...")
    vae = AutoencoderKL.from_pretrained(args.vae_id, torch_dtype=torch.float16)

    print("Loading Stable Diffusion pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    print(f"Loading LoRA weights from {args.lora_repo}...")
    pipe.load_lora_weights(args.lora_repo, weight_name="pytorch_lora_weights.safetensors")

    print("Generating images...")
    image_output = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        guidance=args.guidance,
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=args.num_images
    )

    torch.cuda.empty_cache()

    print("Creating image grid...")
    output_image = image_grid(image_output.images, args.rows, args.cols)

    print(f"Saving image to {args.output}...")
    output_image.save(args.output)
    print("Image saved successfully!")

if __name__ == "__main__":
    main()
    
    
# python script.py --prompt "A red Ford Maverick pickup truck in an epic landscape" --output "maverick.png"


