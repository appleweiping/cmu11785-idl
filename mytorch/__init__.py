"""MyTorch: a from-scratch, NumPy-only deep-learning library.

Implements the CMU 11-785 "Introduction to Deep Learning" HW1-HW4 Part 1
components: MLP + autograd-style backprop, activations, losses, batchnorm,
CNNs (Conv1d/2d, pooling), RNN/GRU cells with CTC loss and greedy/beam
decoding, and scaled dot-product / multi-head attention.
"""

__version__ = "0.1.0"

from . import nn
from . import optim
from . import models

__all__ = ["nn", "optim", "models"]
