"""
Model profiling utilities: parameter counts, FLOPs calculation, and inference latency benchmark.
"""

import time
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    """
    Counts total trainable and non-trainable parameters in the model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "param_millions": round(total / 1e6, 4),
    }


def measure_inference_latency(
    model: nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    device: str = "cpu",
    warmup: int = 10,
    iterations: int = 50,
) -> dict:
    """
    Measures average inference latency in milliseconds over specified iterations.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    dummy_input = torch.randn(*input_shape, device=dev)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if dev.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    mean_latency_ms = (total_time / iterations) * 1000.0

    return {
        "mean_latency_ms": round(mean_latency_ms, 3),
        "iterations": iterations,
        "device": str(dev),
        "input_shape": list(input_shape),
    }
