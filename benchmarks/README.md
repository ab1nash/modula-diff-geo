# Geometric Transformer Benchmarks

**Main script:** `run_transformer_benchmarks.py` (in project root)

```bash
# Quick test (3 epochs)
python run_transformer_benchmarks.py --quick

# Single benchmark  
python run_transformer_benchmarks.py --benchmark music

# Full run (15 epochs)
python run_transformer_benchmarks.py

# Custom epochs
python run_transformer_benchmarks.py --epochs 50
```

## Benchmarks

| Name | Task | Key Insight |
|------|------|-------------|
| `music` | Detect forward vs reversed sequences | Orientation sensitivity |
| `reaction` | Predict reaction direction | Finsler drift strength |
| `protein` | Secondary structure prediction | L-form vs D-form chirality |

## Cache

Generated data is cached in `.cache/` for faster re-runs.

