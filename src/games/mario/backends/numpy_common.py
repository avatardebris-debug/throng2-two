"""Shared pure-numpy helpers for Mario RL backends (PPO, ICM)."""
from __future__ import annotations

from typing import Optional

import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -10, 10))


def dtanh(x: np.ndarray) -> np.ndarray:
    t = tanh(x)
    return 1.0 - t**2


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + 1e-10)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def init_weight(rows: int, cols: int) -> np.ndarray:
    """Xavier uniform initialization for (rows, cols) weight matrices."""
    limit = np.sqrt(6.0 / (rows + cols))
    return np.random.uniform(-limit, limit, (rows, cols)).astype(np.float32)


class AdamParam:
    """Adam optimizer state for one parameter array."""

    def __init__(
        self,
        shape,
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float = 1.0,
    ):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip

    def step(self, param: np.ndarray, grad: np.ndarray, lr: Optional[float] = None) -> np.ndarray:
        grad = np.clip(grad, -self.grad_clip, self.grad_clip)
        use_lr = self.lr if lr is None else lr
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        param -= use_lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param

    def update(self, param: np.ndarray, grad: np.ndarray, lr: Optional[float] = None, **_) -> np.ndarray:
        """Alias used by ICMModule (same as step)."""
        return self.step(param, grad, lr=lr)
