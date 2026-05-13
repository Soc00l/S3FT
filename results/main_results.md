# Main Results

## Datasets

- In-domain: `GSM8K test`
- Cross-domain: `SVAMP`

## Models

- `base`: `Qwen2.5-1.5B-Instruct`
- `base_vanilla_N`: standard SFT on the same gold subset
- `base_s3ft_N`: selective rewrite with model-correct self-answers on the same subset

## Accuracy

| Model | GSM8K | SVAMP |
| --- | ---: | ---: |
| `base` | 0.37 | 0.76 |
| `base_vanilla_1000` | 0.38 | 0.67 |
| `base_s3ft_1000` | 0.32 | 0.69 |
| `base_vanilla_3000` | 0.34 | 0.66 |
| `base_s3ft_3000` | 0.31 | 0.72 |
| `base_vanilla_full` | 0.36 | 0.68 |
| `base_s3ft_full` | 0.31 | 0.71 |

## Notes

- Vanilla SFT produced small or unstable in-domain gains on `GSM8K`, but consistently reduced `SVAMP` accuracy.
- S3FT did not outperform vanilla on `GSM8K`, but it consistently preserved more cross-dataset generalization on `SVAMP`.
- Neither continued-training route exceeded the base model on `SVAMP`, suggesting that further GSM8K-focused finetuning narrows generalization overall, while S3FT narrows it less.
