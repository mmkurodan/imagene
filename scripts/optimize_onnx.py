#!/usr/bin/env python3
"""
ONNX Optimization Script
Additional optimizations for SDXL ONNX models using ONNX Runtime tools.
"""

import os
import argparse
import json
from pathlib import Path
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
from onnxruntime.transformers import optimizer
from onnxruntime.quantization import quantize_dynamic, QuantType


def optimize_transformer_model(input_path: str, output_path: str, model_type: str):
    """Apply ONNX Runtime transformer optimizations."""
    print(f"Optimizing {input_path} as {model_type}...")
    
    try:
        optimized_model = optimizer.optimize_model(
            input_path,
            model_type=model_type,
            num_heads=0,  # Auto-detect
            hidden_size=0,  # Auto-detect
            optimization_options=optimizer.FusionOptions(model_type),
        )
        optimized_model.save_model_to_file(output_path)
        print(f"  Saved optimized model to {output_path}")
        return True
    except Exception as e:
        print(f"  Optimization failed: {e}")
        # Copy original if optimization fails
        import shutil
        shutil.copy(input_path, output_path)
        return False


def quantize_model(input_path: str, output_path: str, quantize_unet: bool = False):
    """Apply INT8 dynamic quantization where possible."""
    print(f"Quantizing {input_path}...")
    
    # Skip UNet quantization by default (quality loss)
    if "unet" in input_path.lower() and not quantize_unet:
        print("  Skipping UNet quantization (use --quantize-unet to enable)")
        import shutil
        shutil.copy(input_path, output_path)
        return
    
    try:
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
            },
        )
        
        # Compare sizes
        orig_size = Path(input_path).stat().st_size / (1024 * 1024)
        new_size = Path(output_path).stat().st_size / (1024 * 1024)
        reduction = (1 - new_size / orig_size) * 100
        print(f"  Quantized: {orig_size:.1f} MB -> {new_size:.1f} MB ({reduction:.1f}% reduction)")
    except Exception as e:
        print(f"  Quantization failed: {e}")
        import shutil
        shutil.copy(input_path, output_path)


def add_metadata(model_path: str, metadata: dict):
    """Add metadata to ONNX model."""
    print(f"Adding metadata to {model_path}...")
    
    model = onnx.load(model_path)
    
    for key, value in metadata.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
    
    onnx.save(model, model_path)
    print(f"  Added {len(metadata)} metadata entries")


def get_model_io_info(model_path: str) -> dict:
    """Extract input/output information from ONNX model."""
    model = onnx.load(model_path)
    
    def get_shape(tensor):
        return [
            dim.dim_value if dim.dim_value > 0 else dim.dim_param
            for dim in tensor.type.tensor_type.shape.dim
        ]
    
    def get_dtype(tensor):
        dtype_map = {
            1: "float32",
            7: "int64",
            10: "float16",
            11: "double",
        }
        return dtype_map.get(tensor.type.tensor_type.elem_type, "unknown")
    
    inputs = {}
    for inp in model.graph.input:
        inputs[inp.name] = {
            "shape": get_shape(inp),
            "dtype": get_dtype(inp),
        }
    
    outputs = {}
    for out in model.graph.output:
        outputs[out.name] = {
            "shape": get_shape(out),
            "dtype": get_dtype(out),
        }
    
    return {"inputs": inputs, "outputs": outputs}


def validate_model(model_path: str) -> bool:
    """Validate ONNX model can be loaded by ONNX Runtime."""
    print(f"Validating {model_path}...")
    
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        
        print(f"  Model valid - {len(session.get_inputs())} inputs, {len(session.get_outputs())} outputs")
        return True
    except Exception as e:
        print(f"  Validation failed: {e}")
        return False


def optimize_for_mobile(model_path: str, output_path: str):
    """Apply mobile-specific optimizations."""
    print(f"Applying mobile optimizations to {model_path}...")
    
    model = onnx.load(model_path)
    
    # Set IR version for mobile compatibility
    if model.ir_version > 8:
        print(f"  Adjusting IR version from {model.ir_version} to 8")
        model.ir_version = 8
    
    onnx.save(model, output_path)


def main():
    parser = argparse.ArgumentParser(description="Optimize SDXL ONNX models")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="onnx",
        help="Input directory with ONNX files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_optimized",
        help="Output directory for optimized ONNX files",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization to text encoders",
    )
    parser.add_argument(
        "--quantize-unet",
        action="store_true",
        help="Also quantize UNet (may reduce quality)",
    )
    parser.add_argument(
        "--mobile",
        action="store_true",
        default=True,
        help="Apply mobile-specific optimizations",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip model validation",
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model configurations
    models = {
        "text_encoder_1.onnx": {"type": "bert", "quantize": args.quantize},
        "text_encoder_2.onnx": {"type": "bert", "quantize": args.quantize},
        "unet.onnx": {"type": "unet", "quantize": args.quantize and args.quantize_unet},
        "vae_decoder.onnx": {"type": "vae", "quantize": False},
    }
    
    # Process each model
    for model_name, config in models.items():
        input_path = input_dir / model_name
        if not input_path.exists():
            print(f"Skipping {model_name} - not found")
            continue
        
        output_path = output_dir / model_name
        temp_path = output_dir / f"temp_{model_name}"
        
        # Step 1: Apply transformer optimizations
        if config["type"] in ["bert", "gpt2"]:
            optimize_transformer_model(str(input_path), str(temp_path), config["type"])
        else:
            import shutil
            shutil.copy(str(input_path), str(temp_path))
        
        # Step 2: Quantize if requested
        if config["quantize"]:
            quantize_model(str(temp_path), str(output_path), args.quantize_unet)
        else:
            import shutil
            shutil.copy(str(temp_path), str(output_path))
        
        # Step 3: Mobile optimizations
        if args.mobile:
            optimize_for_mobile(str(output_path), str(output_path))
        
        # Step 4: Add metadata
        io_info = get_model_io_info(str(output_path))
        metadata = {
            "model_name": model_name.replace(".onnx", ""),
            "model_type": "sdxl",
            "component": config["type"],
            "io_info": io_info,
            "optimized": True,
            "quantized": config["quantize"],
        }
        add_metadata(str(output_path), metadata)
        
        # Step 5: Validate
        if not args.skip_validation:
            validate_model(str(output_path))
        
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()
        
        print()
    
    # Copy tokenizer and scheduler configs
    for config_dir in ["tokenizer_1", "tokenizer_2", "scheduler"]:
        src = input_dir / config_dir
        dst = output_dir / config_dir
        if src.exists():
            import shutil
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Copied {config_dir} config")
    
    # Generate manifest file for Android
    manifest = {
        "models": {},
        "version": "1.0",
        "framework": "onnx_runtime_mobile",
    }
    
    for model_file in output_dir.glob("*.onnx"):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        manifest["models"][model_file.name] = {
            "size_mb": round(size_mb, 2),
            "io_info": get_model_io_info(str(model_file)),
        }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Optimization Summary")
    print("=" * 50)
    total_size = 0
    for model_file in sorted(output_dir.glob("*.onnx")):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  {model_file.name}: {size_mb:.1f} MB")
    print(f"\nTotal model size: {total_size:.1f} MB")


if __name__ == "__main__":
    main()
