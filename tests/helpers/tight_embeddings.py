"""Shared helpers for the tight embeddings tests (used by more than one test module)."""
import numpy as np


DIM = 8


def vec(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v
