import argparse
import os
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Train a DreamBooth model using AutoTrain.")

    parser.add_argument("--project_name", type=str, default="Dreambooth_SDXL", help="Name of the project.")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base model to fine-tune.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the training dataset.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID for pushing to Hugging Face Hub.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token for authentication.")
    parser.add_argument("--prompt", type=str, default="A photorealistic image of a <myObject001>, ultra-detailed, natural lighting, 8k resolution.", help="Training prompt for DreamBooth.")
    parser.add_argument("--class_prompt", type=str, default="A photorealistic image of a generic subject, ultra-detailed, natural lighting, 8k resolution.", help="Class prompt for DreamBooth.")
    parser.add_argument("--resolution", type=int, default=1024, help="Training image resolution.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps.")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--use_fp16", action="store_true", help="Enable FP16 training.")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push trained model to Hugging Face Hub.")

    args = parser.parse_args()

    # Construct the AutoTrain command
    autotrain_command = f"""
    autotrain dreambooth \\
        --model {args.model_name} \\
        --project-name {args.project_name} \\
        --image-path {args.data_dir} \\
        --prompt "{args.prompt}" \\
        --class-prompt "{args.class_prompt}" \\
        --resolution {args.resolution} \\
        --batch-size {args.batch_size} \\
        --num-steps {args.num_steps} \\
        --gradient-accumulation {args.gradient_accumulation} \\
        --lr {args.learning_rate} \\
        {"--fp16 \\" if args.use_fp16 else ""} 
        {"--gradient-checkpointing \\" if args.use_gradient_checkpointing else ""} 
        {"--push-to-hub \\" if args.push_to_hub else ""} 
        --token {args.hf_token} \\
        --repo-id {args.repo_id}
    """

    # Remove excessive white spaces and newlines
    autotrain_command = " ".join(autotrain_command.split())

    print("Running AutoTrain DreamBooth command...")
    os.system(autotrain_command)

if __name__ == "__main__":
    main()