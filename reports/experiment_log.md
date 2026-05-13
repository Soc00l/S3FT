# Experiment Log

## 2026-05-13 Summary

### Base-model route

| Model | GSM8K | SVAMP |
| --- | ---: | ---: |
| `base` | 0.37 | 0.76 |
| `base_vanilla_1000` | 0.38 | 0.67 |
| `base_s3ft_1000` | 0.32 | 0.69 |
| `base_vanilla_3000` | 0.34 | 0.66 |
| `base_s3ft_3000` | 0.31 | 0.72 |
| `base_vanilla_full` | 0.36 | 0.68 |
| `base_s3ft_full` | 0.31 | 0.71 |

### Observations

- Vanilla SFT slightly improved or matched GSM8K at small scale, but consistently hurt SVAMP.
- S3FT underperformed vanilla on GSM8K, but retained more SVAMP generalization at `1000`, `3000`, and `full`.
- Increasing `max_new_tokens` during evaluation changed scores only marginally, so truncation was not the main driver of the trend.
