#!/usr/bin/env python3
"""
SDXL ONNX Export Script
Exports Stable Diffusion XL components to ONNX format with optimizations.
"""

import os
import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
import onnx
from onnxsim import simplify


def setup_dirs(output_dir: str) -> Path:
    """Create output directory structure."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def load_sdxl_pipeline(model_id: str, device: str = "cpu"):
    """Load SDXL pipeline from HuggingFace."""
    print(f"Loading SDXL pipeline from {model_id}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    )
    pipe.to(device)
    return pipe


def export_text_encoder(encoder, tokenizer, output_path: str, name: str, max_length: int = 77):
    """Export a text encoder to ONNX."""
    print(f"Exporting {name}...")
    
    encoder.eval()
    
    # Create dummy input
    dummy_input_ids = torch.zeros(1, max_length, dtype=torch.int64)
    
    # Export
    torch.onnx.export(
        encoder,
        (dummy_input_ids,),
        output_path,
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
            "pooler_output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    # Simplify
    simplify_onnx(output_path)
    print(f"  Saved to {output_path}")


def export_unet(unet, output_path: str, fp16: bool = True):
    """Export UNet to ONNX with optional FP16 conversion."""
    print("Exporting UNet...")
    
    unet.eval()
    unet.set_attn_processor(AttnProcessor2_0())
    
    # SDXL UNet input shapes
    batch_size = 1
    latent_channels = 4
    latent_height = 128  # 1024 / 8
    latent_width = 128
    
    # Dummy inputs
    sample = torch.randn(batch_size, latent_channels, latent_height, latent_width)
    timestep = torch.tensor([1.0])
    encoder_hidden_states = torch.randn(batch_size, 77, 2048)
    added_cond_kwargs_text_embeds = torch.randn(batch_size, 1280)
    added_cond_kwargs_time_ids = torch.randn(batch_size, 6)
    
    # Wrapper class for SDXL UNet with added conditions
    class UNetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
        
        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
    
    wrapper = UNetWrapper(unet)
    wrapper.eval()
    
    # Export
    torch.onnx.export(
        wrapper,
        (sample, timestep, encoder_hidden_states, 
         added_cond_kwargs_text_embeds, added_cond_kwargs_time_ids),
        output_path,
        input_names=["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"],
        output_names=["out_sample"],
        dynamic_axes={
            "sample": {0: "batch_size", 2: "height", 3: "width"},
            "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
            "text_embeds": {0: "batch_size"},
            "time_ids": {0: "batch_size"},
            "out_sample": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    # Simplify
    simplify_onnx(output_path)
    
    # Convert to FP16 if requested
    if fp16:
        convert_to_fp16(output_path)
    
    print(f"  Saved to {output_path} (FP16={fp16})")


def export_vae_decoder(vae, output_path: str):
    """Export VAE decoder to ONNX."""
    print("Exporting VAE decoder...")
    
    vae.eval()
    
    # Create decoder-only wrapper
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.decoder = vae.decoder
            self.post_quant_conv = vae.post_quant_conv
            self.config = vae.config
        
        def forward(self, latents):
            # Scale latents
            latents = latents / self.config.scaling_factor
            latents = self.post_quant_conv(latents)
            image = self.decoder(latents)
            return image
    
    decoder = VAEDecoderWrapper(vae)
    decoder.eval()
    
    # Dummy input
    latent = torch.randn(1, 4, 128, 128)
    
    # Export
    torch.onnx.export(
        decoder,
        (latent,),
        output_path,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={
            "latent": {0: "batch_size", 2: "height", 3: "width"},
            "image": {0: "batch_size", 2: "height", 3: "width"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    # Simplify
    simplify_onnx(output_path)
    print(f"  Saved to {output_path}")


def simplify_onnx(model_path: str):
    """Simplify ONNX model using onnx-simplifier."""
    print(f"  Simplifying {model_path}...")
    try:
        model = onnx.load(model_path)
        model_simplified, check = simplify(model)
        if check:
            onnx.save(model_simplified, model_path)
            print("  Simplification successful")
        else:
            print("  Simplification check failed, keeping original")
    except Exception as e:
        print(f"  Simplification failed: {e}")


def convert_to_fp16(model_path: str):
    """Convert ONNX model to FP16."""
    from onnxconverter_common import float16
    
    print(f"  Converting to FP16...")
    model = onnx.load(model_path)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
    )
    onnx.save(model_fp16, model_path)


def main():
    parser = argparse.ArgumentParser(description="Export SDXL to ONNX")
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for export",
    )
    parser.add_argument(
        "--fp16-unet",
        action="store_true",
        default=True,
        help="Convert UNet to FP16",
    )
    args = parser.parse_args()
    
    # Setup
    output_dir = setup_dirs(args.output_dir)
    
    # Load pipeline
    pipe = load_sdxl_pipeline(args.model_id, args.device)
    
    # Export components
    export_text_encoder(
        pipe.text_encoder,
        pipe.tokenizer,
        str(output_dir / "text_encoder_1.onnx"),
        "text_encoder_1",
    )
    
    export_text_encoder(
        pipe.text_encoder_2,
        pipe.tokenizer_2,
        str(output_dir / "text_encoder_2.onnx"),
        "text_encoder_2",
    )
    
    export_unet(
        pipe.unet,
        str(output_dir / "unet.onnx"),
        fp16=args.fp16_unet,
    )
    
    export_vae_decoder(
        pipe.vae,
        str(output_dir / "vae_decoder.onnx"),
    )
    
    # Save tokenizer config for inference
    pipe.tokenizer.save_pretrained(str(output_dir / "tokenizer_1"))
    pipe.tokenizer_2.save_pretrained(str(output_dir / "tokenizer_2"))
    
    # Save scheduler config
    pipe.scheduler.save_config(str(output_dir / "scheduler"))
    
    print("\nExport complete!")
    print(f"Models saved to: {output_dir}")
    
    # Print file sizes
    print("\nModel sizes:")
    for f in output_dir.glob("*.onnx"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
