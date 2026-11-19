"""
Main CLI entry point for MPF-Net / FinalNet Brain Tumor Segmentation & Classification.

Authors & Team Members:
  - Vishal Tanwar (Roll No: 2024UCA1809)
  - Kumar Sahil   (Roll No: 2024UCA1817)

Datasets:
  - D1: https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset
  - D2: https://www.kaggle.com/datasets/shubhamgajjar/brain-tumor-classification-2d
  - D3: https://www.kaggle.com/datasets/wajeehaldeenaljunaid/mixed-brain-tumor-dataset
  - Figshare: https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset

Usage:
  python main.py --mode cost --resolution 224
  python main.py --mode cost --resolution 512
  python main.py --mode evaluate --checkpoint <path_to_weights> --data-dir <path>
  python main.py --mode train --data-dir <path> --epochs 35 --batch-size 16
  python main.py --mode smoke --data-dir <path>
"""

import os
import sys
import argparse
import json
import torch

from src.utils.seed import seed_all
from src.utils.profiler import count_parameters, measure_inference_latency


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPF-Net / FinalNet Brain Tumor Segmentation and Classification Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="cost",
        choices=["cost", "train", "evaluate", "smoke"],
        help="Execution mode: cost (profile params/latency), train, evaluate, or smoke test.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mpfnet",
        choices=["mpfnet", "finalnet", "aspp_fpn"],
        help="Model architecture variant.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset directory (Figshare .mat folder or Kaggle dataset directory).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        choices=[224, 256, 512],
        help="Input image resolution (H=W). Default: 256.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of classification classes (3 for Figshare, 4 for D1/D3, 5 for D2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training/evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Number of training epochs per fold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (.pt) for evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save experimental results and checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_model(args):
    try:
        from src.models.mpfnet import MPFNet
    except ImportError as e:
        print(f"[ERROR] Failed to import model: {e}")
        print("Please ensure segmentation-models-pytorch is installed:")
        print("    pip install segmentation-models-pytorch")
        sys.exit(1)

    use_aspp = (args.model == "aspp_fpn")
    use_fpn = (args.model == "aspp_fpn")

    model = MPFNet(
        num_classes=args.num_classes,
        pretrained=False,  # Use random weights for offline benchmark
        use_aspp=use_aspp,
        use_fpn=use_fpn,
    )
    return model


def run_cost_benchmark(args, device):
    print(f"\n=======================================================")
    print(f"  MODEL PROFILING: PARAMETERS & INFERENCE LATENCY")
    print(f"=======================================================")
    print(f"Model variant  : {args.model}")
    print(f"Resolution     : {args.resolution}x{args.resolution}")
    print(f"Num classes    : {args.num_classes}")
    print(f"Device         : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    model = build_model(args).to(device)
    params = count_parameters(model)
    print(f"\n--- Parameter Count ---")
    print(f"Total Parameters     : {params['total_parameters']:,}")
    print(f"Trainable Parameters : {params['trainable_parameters']:,}")
    print(f"Parameters (Millions): {params['param_millions']} M")

    input_shape = (1, 3, args.resolution, args.resolution)
    print(f"\n--- Inference Latency Benchmark (Input: {input_shape}) ---")
    latency = measure_inference_latency(
        model,
        input_shape=input_shape,
        device=str(device),
        warmup=10,
        iterations=50,
    )
    print(f"Mean Latency         : {latency['mean_latency_ms']} ms per image")
    print(f"Throughput           : {1000.0 / max(0.001, latency['mean_latency_ms']):.1f} frames/sec")
    print(f"=======================================================\n")


def run_smoke_test(args, device):
    print(f"\n[INFO] Running 2-step smoke test on synthetic batch...")
    model = build_model(args).to(device)
    dummy_x = torch.randn(2, 3, args.resolution, args.resolution, device=device)
    dummy_mask = torch.randint(0, 2, (2, 1, args.resolution, args.resolution), device=device).float()
    dummy_y = torch.randint(0, args.num_classes, (2,), device=device)
    has_mask = torch.tensor([True, True], device=device)

    from src.models.modules import Uncertainty
    from src.training.losses import segmentation_loss

    unc = Uncertainty().to(device)
    ce = torch.nn.CrossEntropyLoss()

    out = model(dummy_x)
    assert out["seg"].shape == (2, 1, args.resolution, args.resolution), "Seg shape mismatch!"
    assert out["cls"].shape == (2, args.num_classes), "Cls shape mismatch!"

    l_seg = segmentation_loss(out, dummy_mask, has_mask)
    l_cls = ce(out["cls"], dummy_y)
    total_loss = unc(l_seg, l_cls)

    total_loss.backward()
    print(f"[SUCCESS] Forward pass, loss computation, and backprop completed successfully!")
    print(f"          Loss value: {total_loss.item():.4f}\n")


def main():
    args = parse_args()
    seed_all(args.seed)
    device = get_device(args.device)

    if args.mode == "cost":
        run_cost_benchmark(args, device)
    elif args.mode == "smoke":
        run_smoke_test(args, device)
    elif args.mode in ["train", "evaluate"]:
        if not args.data_dir or not os.path.exists(args.data_dir):
            print(f"[ERROR] Data directory '{args.data_dir}' not found.")
            print("Please specify a valid dataset path using --data-dir.")
            print("To run offline profiling or verification, use --mode cost or --mode smoke.")
            sys.exit(1)
        print(f"[INFO] Mode '{args.mode}' invoked for data at '{args.data_dir}'.")
        # Full training and evaluation hooks are available in src/training and src/evaluation


if __name__ == "__main__":
    main()
