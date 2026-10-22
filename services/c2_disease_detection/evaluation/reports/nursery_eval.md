# Evaluation Benchmark (T=0.135, tau=0.6)

| Method | K | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---|---|---|---|---|
| Proposed: Meta-Trained Head (MobileNetV3) | 1-shot | 0.500 +/- 0.087 | 0.485 +/- 0.028 | 0.480 +/- 0.112 | 0.441 +/- 0.070 |
| Proposed: Meta-Trained Head (MobileNetV3) | 5-shot | 0.569 +/- 0.038 | 0.523 +/- 0.048 | 0.568 +/- 0.110 | 0.505 +/- 0.053 |
| Proposed: Meta-Trained Head (MobileNetV3) | 10-shot | 0.557 +/- 0.055 | 0.500 +/- 0.044 | 0.529 +/- 0.102 | 0.481 +/- 0.058 |
| Baseline: Raw mobilenet_v3_small (Frozen ImageNet) | 1-shot | 0.441 +/- 0.112 | 0.425 +/- 0.093 | 0.436 +/- 0.134 | 0.394 +/- 0.109 |
| Baseline: Raw mobilenet_v3_small (Frozen ImageNet) | 5-shot | 0.521 +/- 0.065 | 0.484 +/- 0.083 | 0.563 +/- 0.107 | 0.459 +/- 0.079 |
| Baseline: Raw mobilenet_v3_small (Frozen ImageNet) | 10-shot | 0.523 +/- 0.066 | 0.458 +/- 0.056 | 0.522 +/- 0.086 | 0.443 +/- 0.057 |
| Baseline: Raw mobilenet_v2 (Frozen ImageNet) | 1-shot | 0.392 +/- 0.132 | 0.392 +/- 0.089 | 0.415 +/- 0.157 | 0.345 +/- 0.117 |
| Baseline: Raw mobilenet_v2 (Frozen ImageNet) | 5-shot | 0.480 +/- 0.068 | 0.422 +/- 0.078 | 0.500 +/- 0.106 | 0.405 +/- 0.074 |
| Baseline: Raw mobilenet_v2 (Frozen ImageNet) | 10-shot | 0.470 +/- 0.043 | 0.404 +/- 0.052 | 0.485 +/- 0.077 | 0.395 +/- 0.045 |
| Baseline: Raw resnet18 (Frozen ImageNet) | 1-shot | 0.386 +/- 0.130 | 0.395 +/- 0.077 | 0.350 +/- 0.098 | 0.320 +/- 0.086 |
| Baseline: Raw resnet18 (Frozen ImageNet) | 5-shot | 0.482 +/- 0.058 | 0.452 +/- 0.056 | 0.500 +/- 0.088 | 0.410 +/- 0.055 |
| Baseline: Raw resnet18 (Frozen ImageNet) | 10-shot | 0.474 +/- 0.059 | 0.434 +/- 0.059 | 0.459 +/- 0.071 | 0.394 +/- 0.057 |