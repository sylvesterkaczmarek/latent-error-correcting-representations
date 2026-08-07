# Related work

This repository is motivated by two adjacent research threads rather than by an existing alignment implementation.

## Error-correcting discrete latents

Martínez-García, Villacrés, Mitchell, and Olmos, *Protect Before Generate: Error Correcting Codes within Discrete Deep Generative Models* (2024), introduces redundancy into discrete latent-variable models using repetition and polar-code structures. The paper focuses on variational inference, reconstruction, generation quality, and uncertainty calibration.

https://arxiv.org/abs/2410.07840

## Adversarial robustness of latent representations

Zhang, Abdi, and Restuccia, *Adversarial Machine Learning in Latent Representations of Neural Networks* (2023), studies robustness of latent representations under adversarial distortion in distributed neural networks.

https://arxiv.org/abs/2309.17401

## Scope of this repository

The present benchmark asks a different question: whether classical redundancy can act as an **endogenous self-monitoring signal** for internal neural-state corruption, and whether that mechanism fails when the internal state moves coherently to another valid representation.

The benchmark does not claim that this formulation is the first use of error-correcting codes in neural networks. Its contribution is the controlled alignment-motivated experiment and the explicit code-consistent drift failure test.
