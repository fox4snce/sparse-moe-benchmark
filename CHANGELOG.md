# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2024-08-03

### Added
- Initial release of Sparse MoE Benchmark
- Comprehensive benchmark suite comparing MoE vs dense architectures
- Cross-platform support (Windows, Linux, Mac)
- Automated cost calculation and performance analysis
- Sanity tests for router and gradient validation
- Complete documentation with technical analysis

### Features
- **Benchmark Runner**: Measures tokens/sec, latency, VRAM usage, cost efficiency
- **Results Comparison**: Automatic cost calculation and verdict generation
- **Sanity Tests**: Router logits validation and gradient flow verification
- **Cross-Platform**: Windows batch files and Unix Makefile support
- **Documentation**: Technical note with hypothesis → method → results → decision

### Results
- **MoE vs Dense-120M**: 28.4% higher cost, 22% slower throughput
- **MoE vs Dense-300M**: 224% higher cost, 69% slower throughput
- **Conclusion**: Sparse routing overhead exceeds parameter efficiency gains

### Technical Details
- Hardware: NVIDIA RTX 4070 Ti (16GB VRAM)
- Framework: PyTorch 2.0, CUDA 12.1
- Models: Alchemist-MoE (3 specialists), DialoGPT-small/medium
- Metrics: tokens/sec, first-token latency, VRAM usage, cost/1M tokens

## [0.1.0] - 2024-08-03

### Added
- Initial project structure
- Basic MoE implementation
- Benchmark framework
- Results collection and analysis 