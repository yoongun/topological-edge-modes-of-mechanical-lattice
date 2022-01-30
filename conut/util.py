import numpy as np
import torch


class pauli:
    @property
    def i(self):
        return torch.eyes(2, dtype=torch.cdouble)

    @property
    def x(self):
        return torch.tensor([
            [0., 1.],
            [1., 0.]
        ], dtype=torch.cdouble)

    @property
    def y(self):
        return torch.tensor([
            [0., -1.j],
            [1.j, 0.]
        ], dtype=torch.cdouble)

    @property
    def z(self):
        return torch.tensor([
            [1., 0.],
            [0., -1.]
        ], dtype=torch.cdouble)


σ = pauli()


class dirac_impl:
    def __init__(self, N) -> None:
        self.N = N

    def bra(self, m: int):
        return self.ket(m).T.conj()

    def ket(self, m):
        state = torch.tensor([[0.]] * self.N, dtype=torch.cdouble)
        state[m][0] = 1.
        return state


class dirac:
    def __init__(self, *Ns) -> None:
        self.ds = []
        for N in Ns:
            self.ds.append(dirac_impl(N))

    def bra(self, *ms):
        return self.ket(*ms).T.conj()

    def ket(self, *ms):
        state = torch.tensor([[1.]], dtype=torch.cdouble)
        for i, m in enumerate(ms):
            state = torch.kron(state, self.ds[i].ket(m))
        return state
