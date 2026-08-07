# Limitations

This repository is a controlled mechanistic benchmark. It should not be read as evidence that error-correcting latent representations solve AI alignment or model self-monitoring in general.

- The encoder is a small MLP with a four-bit supervised semantic bottleneck.
- Hamming(7,4) and repetition codes are deliberately simple classical codes.
- The corruption model acts on discrete latent bits after message formation.
- The benchmark does not model distributed continuous representations in large transformers.
- The parity-check syndrome detects code invalidity, not semantic invalidity.
- The code-consistent drift challenge is constructed adversarially rather than learned strategically by a model.
- Standard Hamming(7,4) cannot reliably correct two-bit errors. Blind single-error repair can make a double error worse.
- The benchmark does not model recursive self-improvement, deceptive alignment, changing objectives, or human value uncertainty.

The useful conclusion is narrower: structured redundancy can provide a strong endogenous signal for some internal corruption classes, while coherent movement between valid representations can evade that signal completely.
