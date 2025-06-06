#----------------- CoZ with VLM Prompts, Efficient Memory -----------------#
# REQUIRED ENVIRONMENT: coz

INPUT_FOLDER="samples"
OUTPUT_FOLDER="inference_results/coz_vlmprompt_efficient"

CUDA_VISIBLE_DEVICES=0, python inference_coz.py \
-i $INPUT_FOLDER \
-o $OUTPUT_FOLDER \
--rec_type recursive_multiscale \
--prompt_type vlm \
--lora_path ckpt/SR_LoRA/model_20001.pkl \
--vae_path ckpt/SR_VAE/vae_encoder_20001.pt \
--pretrained_model_name_or_path 'stabilityai/stable-diffusion-3-medium-diffusers' \
--ram_ft_path ckpt/DAPE/DAPE.pth \
--ram_path ckpt/RAM/ram_swin_large_14m.pth \
--efficient_memory \

