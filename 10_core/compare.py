#!/usr/bin/env python3
"""
Compare MoE vs Dense results and print summary table
"""

import json
import glob
from pathlib import Path


def calculate_cost_efficiency(results):
    """Calculate cost per 1M tokens (AWS pricing)"""
    # AWS p3.2xlarge pricing: $3.06/hour
    # Assume 100% GPU utilization
    hourly_cost = 3.06
    tokens_per_sec = results.get('tokens_per_sec', 0)
    if tokens_per_sec > 0:
        tokens_per_hour = tokens_per_sec * 3600
        cost_per_1m_tokens = (hourly_cost / tokens_per_hour) * 1e6
    else:
        cost_per_1m_tokens = 0.0
    
    return cost_per_1m_tokens


def load_results():
    """Load all benchmark results"""
    results = {}
    benchmark_dir = Path("benchmarks")
    
    for json_file in benchmark_dir.glob("*_results.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            model_name = data['model']
            # Calculate cost if not present
            if 'aws_cost_per_1m_tokens' not in data:
                data['aws_cost_per_1m_tokens'] = calculate_cost_efficiency(data)
            results[model_name] = data
    
    return results


def print_summary_table(results):
    """Print formatted summary table"""
    print("=" * 80)
    print("🏆 FINAL SHOOTOUT RESULTS")
    print("=" * 80)
    
    # Get all models
    models = list(results.keys())
    if not models:
        print("❌ No results found in benchmarks/")
        return
    
    # Define metrics to display
    metrics = [
        'tokens_per_sec',
        'first_token_latency_ms', 
        'ms_per_token',
        'vram_peak_gb',
        'active_params_m',
        'aws_cost_per_1m_tokens'
    ]
    
    # Print header
    header = f"{'Metric':<20}"
    for model in sorted(models):
        if model == 'alchemist':
            display_name = 'Alchemist-MoE'
        elif model == 'dense120m':
            display_name = 'Dense-120M'
        elif model == 'dense300m':
            display_name = 'Dense-300M'
        else:
            display_name = model
        header += f" {display_name:<15}"
    print(header)
    print("-" * 80)
    
    # Print each metric
    for metric in metrics:
        row = f"{metric:<20}"
        for model in sorted(models):
            value = results[model].get(metric, 0)
            
            if 'cost' in metric:
                row += f" ${value:<14.3f}"
            elif 'params' in metric:
                row += f" {value:<14.1f}M"
            elif 'vram' in metric:
                row += f" {value:<14.2f}GB"
            elif 'latency' in metric or 'time' in metric:
                row += f" {value:<14.1f} ms"
            else:
                row += f" {value:<14.1f}"
        print(row)
    
    print("=" * 80)
    
    # Print verdict
    print("🎯 VERDICT")
    print("=" * 80)
    
    if 'alchemist' in results and 'dense120m' in results:
        alchemist_cost = results['alchemist']['aws_cost_per_1m_tokens']
        dense120_cost = results['dense120m']['aws_cost_per_1m_tokens']
        
        if alchemist_cost > dense120_cost:
            overhead = ((alchemist_cost - dense120_cost) / dense120_cost) * 100
            print(f"❌ MoE LOSES: {overhead:.1f}% MORE expensive than dense-120M")
            print("   → Kill the router, go dense")
        else:
            savings = ((dense120_cost - alchemist_cost) / dense120_cost) * 100
            print(f"✅ MoE WINS: {savings:.1f}% cheaper than dense-120M")
            print("   → Router overhead justified")


def main():
    """Main comparison function"""
    print("📊 CREATING SUMMARY...")
    
    results = load_results()
    print_summary_table(results)


if __name__ == "__main__":
    main() 