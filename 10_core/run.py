#!/usr/bin/env python3
"""
MoE Reality Check - Core Benchmark Runner
Compares dense vs sparse models on single GPU
"""

import argparse
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from typing import Dict, Any

# Import models from parent directory
import sys
sys.path.append('..')
from alchemist.routing.crucible_router import CrucibleRouter
from alchemist.specialists.base_specialist import SpecialistModel, SpecialistConfig
from alchemist.foundation.tokenizer import SharedTokenizer


class AlchemistMoE(nn.Module):
    """Alchemist MoE model for benchmarking"""
    
    def __init__(self, vocab_size=650, n_specialists=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_specialists = n_specialists
        
        # Initialize specialists
        self.specialists = nn.ModuleList([
            SpecialistModel(SpecialistConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_layers=6,
                num_attention_heads=4
            )) for _ in range(n_specialists)
        ])
        
        # Router
        self.router = CrucibleRouter(
            d_model=256,
            n_experts=n_specialists,
            hidden_size=256,
            entropy_beta=0.005,
            skip_threshold=0.15
        )
        
        # Tokenizer
        self.tokenizer = SharedTokenizer()
        
    def forward(self, input_ids, attention_mask=None):
        # Get specialist outputs
        specialist_outputs = []
        for specialist in self.specialists:
            spec_out = specialist(input_ids, attention_mask)
            specialist_outputs.append(spec_out)
        
        # Route
        router_weights = self.router(input_ids, attention_mask)
        
        # Combine outputs
        final_output = torch.zeros_like(specialist_outputs[0])
        for i, spec_out in enumerate(specialist_outputs):
            final_output += router_weights[:, :, i:i+1] * spec_out
            
        return final_output


def create_dense_model(size_mb):
    """Create a dense model of specified size"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if size_mb == 120:
        model_name = "microsoft/DialoGPT-small"  # ~120M params
    elif size_mb == 300:
        model_name = "microsoft/DialoGPT-medium"  # ~300M params
    else:
        raise ValueError(f"Unknown size: {size_mb}MB")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def create_test_prompts(tokenizer, num_prompts=50):
    """Create test prompts for benchmarking"""
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit.",
        "It was the best of times, it was the worst of times.",
        "To be or not to be, that is the question.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "Call me Ishmael.",
        "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
        "The only way to do great work is to love what you do.",
        "Life is what happens when you're busy making other plans.",
        "The future belongs to those who believe in the beauty of their dreams."
    ]
    
    # Repeat prompts to reach num_prompts
    while len(prompts) < num_prompts:
        prompts.extend(prompts[:num_prompts - len(prompts)])
    
    return prompts[:num_prompts]


def measure_model(model, tokenizer, prompts, model_name):
    """Measure model performance"""
    model.eval()
    model.cuda()
    
    # Warmup
    warm_ids = tokenizer.encode("Hello world")
    warm_tokens = torch.tensor(warm_ids).unsqueeze(0).cuda()
    with torch.no_grad():
        _ = model(warm_tokens)
    
    torch.cuda.synchronize()
    
    # Benchmark
    total_tokens = 0
    total_time = 0
    first_token_times = []
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        
        # Measure first token latency
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        first_token_time = (time.time() - start) * 1000  # ms
        
        total_tokens += outputs.shape[1] - input_ids.shape[1]
        first_token_times.append(first_token_time)
        total_time += first_token_time / 1000  # Convert to seconds
    
    # Calculate metrics
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    first_token_latency_ms = sum(first_token_times) / len(first_token_times)
    ms_per_token = (total_time * 1000) / total_tokens if total_tokens > 0 else 0
    
    # VRAM usage
    vram_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    
    # Active parameters
    if hasattr(model, 'get_active_params'):
        active_params_m = model.get_active_params() / 1e6
    else:
        active_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    
    return {
        'model': model_name,
        'tokens_per_sec': tokens_per_sec,
        'first_token_latency_ms': first_token_latency_ms,
        'ms_per_token': ms_per_token,
        'vram_peak_gb': vram_peak_gb,
        'active_params_m': active_params_m,
        'total_tokens': total_tokens,
        'total_time_s': total_time
    }


def calculate_cost_efficiency(results):
    """Calculate cost per 1M tokens (AWS pricing)"""
    # AWS p3.2xlarge pricing: $3.06/hour
    # Assume 100% GPU utilization
    hourly_cost = 3.06
    tokens_per_hour = results['tokens_per_sec'] * 3600
    cost_per_1m_tokens = (hourly_cost / tokens_per_hour) * 1e6
    
    return cost_per_1m_tokens


def main():
    parser = argparse.ArgumentParser(description='MoE Reality Check Benchmark')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"🚀 MOE REALITY CHECK: {config['model_type']}")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model and tokenizer
    if config['model_type'] == 'alchemist':
        model = AlchemistMoE(vocab_size=650, n_specialists=3)
        tokenizer = model.tokenizer
    elif config['model_type'].startswith('dense'):
        size_mb = int(config['model_type'].replace('dense', ''))
        model, tokenizer = create_dense_model(size_mb)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Create test prompts
    prompts = create_test_prompts(tokenizer, config.get('num_prompts', 10))
    print(f"Created {len(prompts)} test prompts")
    
    # Measure
    print(f"🔬 MEASURING {config['model_type'].upper()}")
    print("=" * 50)
    
    results = measure_model(model, tokenizer, prompts, config['model_type'])
    
    # Calculate cost
    cost_per_1m = calculate_cost_efficiency(results)
    results['aws_cost_per_1m_tokens'] = cost_per_1m
    
    # Print results
    print(f"Tokens/sec:       {results['tokens_per_sec']:.1f}")
    print(f"First token (ms): {results['first_token_latency_ms']:.1f}")
    print(f"CUDA time/token:  {results['ms_per_token']:.3f} ms")
    print(f"VRAM peak (GB):   {results['vram_peak_gb']:.2f}")
    print(f"Active params:    {results['active_params_m']:.1f} M")
    print(f"Cost/1M tokens:   ${cost_per_1m:.3f}")
    
    # Save results
    output_file = f"benchmarks/{config['model_type']}_results.json"
    Path("benchmarks").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to {output_file}")


if __name__ == "__main__":
    main() 