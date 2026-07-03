"""MLP reference models for MyTorch (CMU 11-785 HW1P1).

Three fixed-topology MLPs (0, 1 and 4 hidden layers) plus a general ``MLP`` that
accepts an arbitrary list of layer sizes.  ``layers`` is the list of Linear
layers consumed by :class:`~mytorch.optim.sgd.SGD`.
"""

from ..nn.linear import Linear
from ..nn.activation import ReLU


class MLP0:
    """Zero hidden layers: a single linear map + ReLU."""

    def __init__(self, debug=False):
        self.layers = [Linear(2, 3)]
        self.f = [ReLU()]
        self.debug = debug

    def forward(self, A0):
        Z0 = self.layers[0].forward(A0)
        A1 = self.f[0].forward(Z0)
        if self.debug:
            self.Z0, self.A1 = Z0, A1
        return A1

    def backward(self, dLdA1):
        dLdZ0 = self.f[0].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)
        if self.debug:
            self.dLdZ0, self.dLdA0 = dLdZ0, dLdA0
        return dLdA0


class MLP1:
    """One hidden layer."""

    def __init__(self, debug=False):
        self.layers = [Linear(2, 3), Linear(3, 2)]
        self.f = [ReLU(), ReLU()]
        self.debug = debug

    def forward(self, A0):
        Z0 = self.layers[0].forward(A0)
        A1 = self.f[0].forward(Z0)
        Z1 = self.layers[1].forward(A1)
        A2 = self.f[1].forward(Z1)
        if self.debug:
            self.Z0, self.A1, self.Z1, self.A2 = Z0, A1, Z1, A2
        return A2

    def backward(self, dLdA2):
        dLdZ1 = self.f[1].backward(dLdA2)
        dLdA1 = self.layers[1].backward(dLdZ1)
        dLdZ0 = self.f[0].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)
        if self.debug:
            self.dLdZ1, self.dLdA1 = dLdZ1, dLdA1
            self.dLdZ0, self.dLdA0 = dLdZ0, dLdA0
        return dLdA0


class MLP4:
    """Four hidden layers (five linear layers total)."""

    def __init__(self, debug=False):
        self.layers = [
            Linear(2, 4),
            Linear(4, 8),
            Linear(8, 8),
            Linear(8, 4),
            Linear(4, 2),
        ]
        self.f = [ReLU() for _ in range(len(self.layers))]
        self.debug = debug

    def forward(self, A):
        if self.debug:
            self.A = [A]
        for i in range(len(self.layers)):
            Z = self.layers[i].forward(A)
            A = self.f[i].forward(Z)
            if self.debug:
                self.A.append(A)
        return A

    def backward(self, dLdA):
        if self.debug:
            self.dLdA = [dLdA]
        for i in reversed(range(len(self.layers))):
            dLdZ = self.f[i].backward(dLdA)
            dLdA = self.layers[i].backward(dLdZ)
            if self.debug:
                self.dLdA.append(dLdA)
        return dLdA


class MLP:
    """General MLP over an arbitrary list of sizes with a chosen activation.

    ``sizes = [in, h1, ..., out]`` builds ``len(sizes) - 1`` linear layers, each
    followed by ``activation`` (a class, instantiated fresh per layer).
    """

    def __init__(self, sizes, activation=ReLU, debug=False):
        self.layers = [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.f = [activation() for _ in self.layers]
        self.debug = debug

    def forward(self, A):
        for lin, act in zip(self.layers, self.f):
            A = act.forward(lin.forward(A))
        return A

    def backward(self, dLdA):
        for lin, act in zip(reversed(self.layers), reversed(self.f)):
            dLdA = lin.backward(act.backward(dLdA))
        return dLdA
