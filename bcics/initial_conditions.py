__all__ = ["IC"]

import torch
import numpy as np
from torch.autograd import grad

class IC:
    """Initial conditions: y([x, t0]) = func([x, t0])."""

    def __init__(self, geom, func, initial_point=0.):
        self.geom = geom
        self.func = func
        self.initial_point = initial_point
        assert np.isclose(self.geom[-1][0], self.initial_point)

    def error(self, inputs, outputs):
        assert np.allclose(inputs[:, -1].cpu().detach().numpy(), self.initial_point)
        values = self.func(inputs)
        if values.dim() == 2 and list(values.shape)[1] != 1:
            raise RuntimeError(
                "IC function should return an array of shape N by 1 for each component."
                "Use argument 'component' for different output components."
            )
        return outputs - values

class Derivative_IC:
    """Initial conditions: dydt(t0) = func(x)."""

    def __init__(self, geom, initial_point=0., func=None):
        self.geom = geom
        self.func = func
        self.initial_point = initial_point
        assert func is not None
        assert np.isclose(self.geom[-1][0], self.initial_point)

    def time_derivative(self, inputs, outputs):
        return grad(outputs, inputs[:, -1],
            grad_outputs=torch.ones_like(outputs),
            retain_graph=True,
            create_graph=True)[0]

    def error(self, inputs, outputs):
        assert np.allclose(inputs[:, -1].cpu().detach().numpy(), self.initial_point)
        values = self.func(inputs)
        return self.time_derivative(inputs, outputs) - values
