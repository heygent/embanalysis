---
title: Concrete Numeric Representations in LLM Embeddings
author: Emanuele Gentiletti
---


# What is a Concrete Representation?

- The hypothesis we explore is the presence of geometric structures in the organization
  of numerical embeddings that facilitates mathematical computations.

- Precedent in humans: **Savant Syndrome**
    - Some people are able to perform extraordinary feats of mathematical prowess with
      little effort.
    - It is proposed in the scientific literature that this ability doesn't come through
      algorithmic processes, but through the consultation of encoded geometric
      structures that reveal the answer
        - **Concrete Representations** according to Murray et al., 2005 <?>
    - Sequence-space synesthesia
        - Savants and Synasthetes are able to visualize sequences in space and get
          answers to mathematical questions just by navigating them in space <!---->

# LLM Embeddings

!!!

- String is split in tokens, each token corresponds to a single vector of weights in the
  first layer
- Embeddings are trained through gradient descent, which changes the weights to minimize
  a target error function

![Embeddings example]("public/embeddings_example.png) <!---->

# Bad tokenization schemes

- Most current LLMs tokenize numbers in an unintuitive way.
    - L2R  tokenization: $\underbrace{123} \; \underbrace{456} \; \underbrace{7}$
    - R2L  tokenization: $\underbrace{1} \; \underbrace{234} \; \underbrace{567}$
- R2L is preferrable for numeric computations (think doing additions in column), so why
  do LLMs use L2R?
    - Reason: BPE algorithm

[🔗 Tiktokenizer](https://tiktokenizer.vercel.app/) <!---->

# BPE algorithm
- We construct a list of tokens for the LLM to consider as "units" to work with
    - Start: each singular character gets a token (`a`, `b`, `c`, ...)
    - Until we reach the target `vocabulary_size`:
          - Add the most common occurring pair of tokens as a new token
              - For example, the common preposition `to` is very common in English text,
                so it may get added to the target vocabulary (`a`, `b`, `c`, ..., `to`)
              - After some cycles, the most common occorring pair of tokens might be
                `to` and `m` to form the common name `tom` (`a`, `b`, `c`, ..., `to`,
                ..., `tom`)
              - This is how vocabularies in modern LLMs are built. <!---->
- BPE constructs its vocabulary by associating tokens from left to right.
- Tokenizers split sentences according to the vocabulary with the same logic (L2R).
- GPT-2 and GPT-3 used the same criteria for numbers as it did for words, leading to
  tokenization of only most frequently occurring numbers !!!
- Most open-source LLMs today always tokenize integers from 0 to 999 using a single
  token, but they still group tokens L2R (123,456,7) ![GPT-2 unique numeric tokens. A
  yellow square means the corresponding number is represented with a single
  token.]("public/unique_tokens.png") [🔗 beren.io - Integer Tokenization is
  Insane](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/) <!---->
# Bad Clustering and Inference-time correction

- A study has been conducted to benchmark addition performed with both L2R and R2L
  clustering
      - 🧪 Experiment: ask the same addition problems twice
          - First time neutrally: 1234567 + 654321
          - Second time clustering digits with commas: 1,234,567 + 654,321
              - Commas force token separation at the point of insertion
- R2L tokenization dramatically outperforms standard L2R tokenization
    - GPT-3.5: 75.6% accuracy (L2R) vs 97.8% accuracy (R2L)
    - GPT-4: 84.4% accuracy (L2R) vs 98.9% accuracy (R2L)


[📚 Tokenization counts: the impact of tokenization on arithmetic in frontier
LLMs](https://arxiv.org/abs/2402.14903) <!---->
- The models had been trained with a L2R tokenization scheme, yet forcing a R2L one
  improves performance without needing to train the model again

-  The model weights appear to have learned arithmetic algorithms that are fundamentally
   R2L-oriented, despite being trained predominantly on L2R tokenized data.
    - Inductive biases emerge from mathematical structure, not just data statistics <?>

- This is also a clue that the **underlying embedding representation of numbers is
  representative of general mathematical principles**, and not just rote memorization.
  <!---->
# The Platonic Representation Hypothesis

[📚Huh et al. (2024)](https://arxiv.org/abs/2405.07987) argue that all models are
converging to a shared statistical model of reality.

- **Vision-Vision Alignment**
    - Tested 78 vision models of various architectures and training objectives
    - Models with higher performance on VTAB tasks showed greater mutual alignment
    - Competent models clustered together in representation space

- **Cross-Modal Alignment**
    - Vision models and language models show increasing alignment as they scale
    - Models can be "stitched" together across modalities with simple learned mappings
    - Color representations learned from text match those learned from images

- **Evidence from Model Stitching**
    - Different models can have their intermediate layers successfully swapped
    - This works even across different training objectives and datasets
    - Success indicates compatible internal representations <!----> To recap:

- Concrete structures can provide Savants and synaesthetes access to mathematical
  knowledge
- LLMs generalize over mathematical concepts even when they're trained wrongly
- Representations in LLMs seem to be converging

By looking into the structure of embeddings, we seek to understand the shape these might
approach. To understand them, we represent them using dimensionality reduction
techniques. <!---->
# Dimensionality Reduction
We employ the most commonly used dimensionality reduction techniques:

- **Linear**
    - **Singular Value Decomposition**
        - decomposes a matrix as $A = U \Sigma V^T$ (rotation, stretch, rotation).
          $\Sigma$ contains the singular values, which **measure the data stretch along
          each principal direction**.
          - We then sort the features by their singular values to get the most
            significant ones.

    - **Principal Component Analysis**
          - like SVD, but centers the features (subtracts the mean) first. As a
            consequence, the singular values correspond to **the standard deviations
            along each principal component**.
          - has cleaner statistical interpretation, but can destroy non-centered
            structures

- **Non-Linear**
    - **t-SNE**
          - Models proximity relationships in the data as a probability distribution
          - Recreates the data points in a lower-dimensional space, optimizing them
            through gradient descent to have the same distribution
    - **UMAP**
        - preserves both local and global structure using topological data analysis.
          Faster than t-SNE with better scalability.
        - more stable results, better preservation of global relationships <!---->
# Number Semantics

- We also have an opportunity to check semantics in a way that crosses the symbolic
  barrier.
- We can relate the symbols themselves (0, 1, 2, ..., 999) to the **numeric components
  that constitute the embeddings**
- We explore this idea by checking individual features for all the integer embeddings
  and  their correlation with important mathematical sequences:
    - $n_i = i$ (the numbers themselves)
    - $n_i = \log(i)$
    - Prime numbers
    - Fibonacci numbers
    - Triangular numbers

```python {.marimo}

```
