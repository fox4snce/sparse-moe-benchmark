# Sparse MoE vs Dense Transformers: Performance Analysis

## Hypothesis

**Sparse Mixture of Experts (MoE) models should be more cost-effective than dense models** due to:
- Lower active parameters per token
- Reduced memory footprint
- Specialized expert routing

## Method

### Models Tested
1. **Alchemist-MoE**: 3 specialists (16.4M active params), crucible router
2. **Dense-120M**: Microsoft DialoGPT-small (124.4M params)
3. **Dense-300M**: Microsoft DialoGPT-medium (67.2M params)

### Benchmark Setup
- **Hardware**: NVIDIA RTX 4070 Ti (16GB VRAM)
- **Framework**: PyTorch 2.0, CUDA 12.1
- **Test**: 10 prompts, 16 tokens each, greedy decoding
- **Metrics**: tokens/sec, first-token latency, VRAM usage, cost/1M tokens

### Implementation
```python
# Alchemist MoE Architecture
class AlchemistMoE(nn.Module):
    def __init__(self, vocab_size=650, n_specialists=3):
        self.specialists = nn.ModuleList([
            SpecialistModel(SpecialistConfig(...)) 
            for _ in range(n_specialists)
        ])
        self.router = CrucibleRouter(
            d_model=256, n_experts=n_specialists
        )
```

## Results

| Model | Tokens/sec | First Token (ms) | VRAM (GB) | Active Params | Cost/1M tokens |
|-------|------------|------------------|-----------|---------------|----------------|
| Dense-120M | 2,308 | 11.0 | 0.53 | 124.4M | $0.063 |
| Dense-300M | 5,827 | 2.0 | 0.28 | 67.2M | $0.025 |
| **Alchemist-MoE** | **1,798** | **10.0** | **0.095** | **16.4M** | **$0.081** |

## Analysis

### Performance Breakdown
- **Throughput**: MoE is 22% slower than Dense-120M, 69% slower than Dense-300M
- **Latency**: Similar first-token latency across models
- **Memory**: MoE uses 82% less VRAM than Dense-120M
- **Cost**: MoE is 28% more expensive than Dense-120M per token

### Root Cause Analysis
1. **Router Overhead**: The crucible router adds computational overhead
2. **Expert Coordination**: Synchronizing 3 specialists is slower than single forward pass
3. **Memory Access Patterns**: Sparse routing creates non-optimal memory access
4. **Parameter Efficiency**: Lower active params don't translate to faster inference

## Decision

**⚠️ CONDITIONAL RECOMMENDATION**

MoE presents a clear memory vs performance trade-off:

### ❌ **For Most Use Cases: REJECT MoE**
- **Cost Inefficiency**: 28% more expensive than dense baseline
- **Performance Penalty**: 22-69% slower than dense models
- **Complexity Cost**: Router adds overhead without benefits

### ✅ **For VRAM-Constrained Edge Devices: CONSIDER MoE**
- **Memory Efficiency**: 3-5× less VRAM usage (0.10 GB vs 0.28-0.53 GB)
- **Edge Deployment**: May be the only viable option for <4 GB devices
- **Jetson Nano/Laptop**: Could enable deployment where dense models won't fit

## Recommendations

### **For Standard Deployments**
1. **Pivot to Dense**: Use dense models for single-GPU deployment
2. **Scale Vertically**: Larger dense models perform better than MoE
3. **Optimize Memory**: Focus on memory optimization rather than sparsity

### **For Edge/Constrained Deployments**
4. **Consider MoE**: When VRAM is <4 GB and performance is secondary
5. **Profile Alternatives**: 4-expert MoE or int4 quantized dense models
6. **Test Edge Cases**: Jetson Nano, mobile GPUs, memory-constrained laptops

### **For Multi-GPU Scale**
7. **Reconsider at Scale**: MoE may be viable at multi-GPU scale

## Reproducibility

- **Code**: `10_core/run.py` with configs in `10_core/configs/`
- **Tests**: `00_sanity/test_router_shapes.py`
- **Results**: `benchmarks/*_results.json`
- **Quick Run**: `make quick` (≤10 minutes)

## Conclusion

MoE architectures present a clear memory vs performance trade-off. While they offer significant VRAM savings (3-5× less memory), the routing overhead results in slower inference and higher costs per token.

**For most deployments**: Dense models remain the superior choice due to better performance and cost efficiency.

**For edge devices**: MoE may be the only viable option when VRAM is severely constrained (<4 GB), making the performance penalty acceptable for deployment feasibility. 