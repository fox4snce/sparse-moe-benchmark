.PHONY: quick full sweep test clean

# Quick benchmark (≤10 min)
quick:
	python 10_core/run.py --config 10_core/configs/moe_3expert_quick.yaml

# Full evaluation suite (overnight)
full:
	python 10_core/run.py --config 20_extended/configs/moe_full.yaml

# Hyperparameter sweep
sweep:
	bash 20_extended/sweep.sh

# Run sanity tests
test:
	pytest 00_sanity/ -v

# Clean generated files
clean:
	rm -rf benchmarks/*.csv
	rm -rf __pycache__/
	rm -rf */__pycache__/
	rm -rf .pytest_cache/ 