#!/usr/bin/env python3
"""
Export SDXL to ONNX using diffusers' ONNX exporter.

This script delegates ONNX export to diffusers' OnnxStableDiffusionXLPipeline.from_pretrained(..., export=True)
and avoids using torch.onnx.export or any opset version conversion. It requests opset=16 and prefers FP16
for the UNet when requested.

Requirements:
- diffusers providing OnnxStableDiffusionXLPipeline with an export flow
- No onnxscript/version_converter usage
"""

import argparse
import inspect
import sys
from pathlib import Path


def _get_export_kwargs(fn, output_dir: str, device: str, fp16_unet: bool):
    """Build a kwargs dict for OnnxStableDiffusionXLPipeline.from_pretrained by introspecting the signature.

    We only pass keys that the installed diffusers supports so this script is forward/backward compatible.
    """
    sig = inspect.signature(fn)
    candidates = {
        "export": True,
        "opset": 16,
        # common names used by diffusers exporters - only kept if present in the signature
        "output_dir": str(output_dir),
        "export_dir": str(output_dir),
        "fp16": fp16_unet,
        "device": device,
        "use_external_data_format": True,
        "variant": "fp16" if fp16_unet else None,
    }
    kwargs = {k: v for k, v in candidates.items() if k in sig.parameters and v is not None}
    return kwargs


def export_with_diffusers(model_id: str, output_dir: str, device: str = "cpu", fp16_unet: bool = False):
    """Export SDXL components to ONNX using diffusers' ONNX exporter.

    This function uses OnnxStableDiffusionXLPipeline.from_pretrained(..., export=True) when available.
    It makes no additional opset conversions and does not call torch.onnx.export.
    """
    try:
        # Prefer the SDXL-specific ONNX pipeline exporter
        from diffusers import OnnxStableDiffusionXLPipeline as OnnxExporter
    except Exception:
        raise RuntimeError(
            "OnnxStableDiffusionXLPipeline not available in diffusers. "
            "Please upgrade diffusers to a version that provides the ONNX exporter for SDXL."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fn = OnnxExporter.from_pretrained
    kwargs = _get_export_kwargs(fn, output_dir, device, fp16_unet)

    print("Exporting SDXL ONNX models with diffusers ONNX exporter")
    print(f"  model_id: {model_id}")
    print(f"  output_dir: {output_dir}")
    print(f"  device: {device}")
    print(f"  fp16_unet: {fp16_unet}")
    print(f"  exporter kwargs: {kwargs}")

    # Call the exporter; many diffusers ONNX exporters export files as a side-effect and may
    # return a pipeline object or a dict. We don't rely on the return value.
    try:
        _ret = fn(model_id, **kwargs)
    except TypeError as e:
        # If the call fails due to unexpected kwargs, try a minimal call (model_id + export/opset)
        print("  Initial export call failed, retrying with minimal args...", file=sys.stderr)
        minimal = {}
        if "export" in inspect.signature(fn).parameters:
            minimal["export"] = True
        if "opset" in inspect.signature(fn).parameters:
            minimal["opset"] = 16
        if "output_dir" in inspect.signature(fn).parameters:
            minimal["output_dir"] = str(output_dir)
        print(f"  Retrying with: {minimal}")
        _ret = fn(model_id, **minimal)

    # After export, list exported files for user visibility
    exported = sorted([p for p in output_dir.rglob("*")])
    if not exported:
        print("Warning: exporter did not emit any files to output_dir", file=sys.stderr)
    else:
        print("Exported files:")
        for p in exported:
            try:
                print(" ", p.relative_to(output_dir))
            except Exception:
                print(" ", p)

    print("Export complete.")


def main():
    parser = argparse.ArgumentParser(description="Export SDXL to ONNX using diffusers' ONNX exporter")
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default="onnx", help="Directory to write ONNX files to")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Device to run any export-time ops on")
    parser.add_argument("--fp16-unet", action="store_true", help="Request FP16 UNet export where the exporter supports it (RedMagic 11 Pro)")
    args = parser.parse_args()

    export_with_diffusers(args.model_id, args.output_dir, device=args.device, fp16_unet=args.fp16_unet)


if __name__ == "__main__":
    main()
