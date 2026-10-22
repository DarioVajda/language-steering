"""Steering functions applied to the residual stream during the prompt's forward pass.

A steering function is any object with
  * `layers`: the layers (1..n_layers, see model.py) whose block outputs it modifies;
  * `__call__(layer, h, batch) -> h`: `h` is the [B, T, d] block output and `batch` the
    model.Batch, whose `mask(positions)`, `role` and `lang` tell which tokens are which.
New steering methods only need a new subclass here and a Method in methods.py that builds it.
"""

import torch


class Steering:
    layers = ()

    def __call__(self, layer, h, batch):
        raise NotImplementedError


class AddVector(Steering):
    """h_p <- h_p + alpha * v for every position p in `positions` (Kirtane & Huang, 2026, Eq. 6)."""

    def __init__(self, layer, vector, alpha, positions):
        self.layers = (layer,)
        self.vector, self.alpha, self.positions = torch.as_tensor(vector), alpha, positions

    def __call__(self, layer, h, batch):
        m = batch.mask(self.positions).unsqueeze(-1).to(h.dtype)
        return h + m * (self.alpha * self.vector).to(h.device, h.dtype)


class AddLanguageVector(Steering):
    """h_p <- h_p + alpha * V[lang_p] for every position p in `positions` that has a language:
    each token gets the vector of its own segment's language. `vectors` is [len(batch.langs), d],
    row i for language batch.langs[i]."""

    def __init__(self, layer, vectors, alpha, positions):
        self.layers = (layer,)
        self.vectors, self.alpha, self.positions = torch.as_tensor(vectors), alpha, positions

    def __call__(self, layer, h, batch):
        m = (batch.mask(self.positions) & (batch.lang >= 0)).unsqueeze(-1).to(h.dtype)
        V = (self.alpha * self.vectors).to(h.device, h.dtype)
        return h + m * V[batch.lang.clamp(min=0)]
