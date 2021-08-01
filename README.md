# Self-training with Noisy Student improves ImageNet classification
Noisy Student Training is a semi-supervised training method which achieves 88.4% top-1 accuracy on ImageNet
and surprising gains on robustness and adversarial benchmarks.
Noisy Student Training is based on the self-training framework and trained with 4-simple steps:

1. Train a classifier on labeled data (teacher).
2. Infer labels on a much larger unlabeled dataset.
3. Train a larger classifier on the combined set, adding noise (noisy student).
4. Go to step 2, with student as teacher.

