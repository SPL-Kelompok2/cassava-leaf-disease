"""Fungsi loss untuk pelatihan model visi: Taylor Cross Entropy + label smoothing.

Paper (Bagian 4.5) memakai "Taylor Cross Entropy Loss (Feng et al., 2021) with
label smoothing (Mueller et al., 2019)", nilai smoothing 0.05, yang menurut
paper tahan terhadap data berderau (Chen et al., 2022).

Implementasi di sini mengikuti bentuk "Taylor softmax" yang umum dipakai pada
kompetisi Kaggle cassava: softmax diganti ekspansi Taylor orde-n dari exp(x),

    f(x) = 1 + x + x^2/2! + ... + x^n/n!      (n genap -> f(x) > 0)
    p_k  = f(z_k) / sum_j f(z_j)

lalu cross entropy dengan label smoothing dihitung atas log(p). Orde Taylor default n=2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TaylorSoftmax(nn.Module):
    def __init__(self, dim: int = 1, n: int = 2):
        super().__init__()
        if n < 2 or n % 2 != 0:
            raise ValueError("orde Taylor harus genap dan >= 2 agar f(x) > 0")
        self.dim, self.n = dim, n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fn = torch.ones_like(x)
        term = torch.ones_like(x)
        for i in range(1, self.n + 1):
            term = term * x / i
            fn = fn + term
        return fn / fn.sum(dim=self.dim, keepdim=True)


class TaylorCrossEntropyLoss(nn.Module):
    """Taylor CE dengan label smoothing.

    Parameter
    ---------
    num_classes : jumlah kelas (5).
    n           : orde Taylor (genap).
    smoothing   : label smoothing; 0.05 sesuai paper.
    """

    def __init__(self, num_classes: int = 5, n: int = 2, smoothing: float = 0.05):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing harus di [0, 1)")
        self.num_classes, self.smoothing = num_classes, smoothing
        self.taylor = TaylorSoftmax(dim=1, n=n)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = self.taylor(logits.float()).log()  # hitung di float32 (aman untuk AMP)
        with torch.no_grad():
            dist = torch.full_like(logp, self.smoothing / (self.num_classes - 1))
            dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-dist * logp, dim=1))
