# Sparse Mixture-of-Experts Reality Check
[![Bench](https://img.shields.io/badge/bench-pass-brightgreen)](https://github.com/fox4snce/sparse-moe-benchmark/blob/main/benchmarks/results.csv)

**I turn a "Could we just...?" AI idea into go / pivot / drop data in 72 hours.**  
This repo reproduces why sparse 3-expert MoE underperforms dense models on a single GPU.

---

## Quick start (≤ 10 min)

```bash
git clone https://github.com/fox4snce/sparse-moe-benchmark
cd sparse-moe-benchmark
pip install -r requirements.txt   # or install PyTorch separately
make quick        # Windows: .\make.bat quick
```

**Reproducibility Guarantee**: All benchmarks use locked GPU clocks, deterministic settings, and 200+ iterations for ±2% run-to-run variance. See `run_benchmark.py` for the bulletproof methodology.

**Note**: GPU clock locking requires administrator privileges. Without admin access, results may vary ±5-10% due to thermal throttling and boost behavior.

**PyTorch Version**: For full model compatibility, use PyTorch 2.6+ (see requirements.txt). Older versions will use fallback models.

## Tokenization: Simple, Reproducible, and Swappable

This repo uses a simple, rule-based tokenizer (`SimpleTokenizer` in `simple_tokenizer.py`) for all benchmarks. It splits on whitespace and punctuation, lowercases text, and requires no external files or dependencies. This ensures zero setup and perfect reproducibility for anyone cloning the repo.

**Why this approach?**
- Zero setup: No need to download or train a tokenizer model.
- 100% reproducibility: No "missing spm.model" errors.
- Good enough for benchmarking speed, VRAM, and cost.

**Limitations:**
- Token count is ~1.5-2× higher than a trained BPE/WordPiece tokenizer.
- Not suitable for production LLMs or fine-tuning on real data.

**How to swap in a trained tokenizer:**
If you want to use a real SentencePiece or HuggingFace tokenizer, just replace the import and initialization in `run.py`:

```python
from alchemist.foundation.tokenizer import SharedTokenizer
# tokenizer = SimpleTokenizer()
tokenizer = SharedTokenizer(model_path="tokenizer.model")
```

and provide a `tokenizer.model` file.

**Core results (i7-14700KF · RTX 4070 Ti · seq 256 · bulletproof methodology)**

### 🔄 Generation Speed (Autoregressive Decoding)
*Real-world usage: one token at a time*

| Model | Tokens/sec | First Token (ms) | VRAM (GB) | Active Params | Cost/1M tokens* |
|-------|------------|------------------|-----------|---------------|----------------|
| Dense-120M | 170.6 | 58.6 | 0.55 | 124M | $1.01 |
| **MoE 3×60M** | **100.3** | **99.7** | **0.09** | **16M** | **$0.60** |

<sub>*Token cost assumes identical hardware & electricity; see docs/tech_note.md.*</sub>

## Verdict

**Memory vs Performance Trade-off:**

✅ **Memory edge** – MoE fits in 0.10 GB; useful for <4 GB edge devices  
❌ **Throughput & cost penalty** – 22–69% slower and ≥28% pricier per token

**For most single-GPU setups**: Kill the router; go dense.  
**For VRAM-constrained edge devices**: MoE may be the only viable option.

## What's inside

```
sparse-moe-benchmark/
├── 00_sanity/      - shape & gradient tests (runs <30s)
├── 10_core/        - quick benchmark (≤10 min)
├── 20_extended/    - full overnight evaluation
├── benchmarks/     - raw JSON & CSV
└── docs/tech_note.md
```

See `docs/dev_guide.md` for the full test matrix and implementation checklist.

---

**Need a 3-day reality-check on your idea?**  
Fixed price US $2,500 · async Q&A · reply in 24h.  
✉ j.caldwell@simplychaos.org 
