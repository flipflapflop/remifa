# remifa
Reduced and mixed precision factorization algorithms.

This repository contains a high-performance implementation of various LU
factorization algorithms for NVIDIA GPU devices. In particular, some
of these algorithms are capable of exploiting the mixed precision
floating point units Tensor Cores available on Volta and Turing
architectures.

These codes were used to obtain the experimental results of the article "Mixed Precision LU Factorization on GPU Tensor Cores: Reducing Data Movement and Memory Footprint", co-authored by Florent Lopez and Theo Mary.

