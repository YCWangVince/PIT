"""Boundary conditions."""

__all__ = [
    "BC",
    "DirichletBC",
    "NeumannBC",
    "OperatorBC",
    "PeriodicBC",
]

import numbers
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.autograd import grad


class BC(ABC):
    """
    Boundary condition base class.
    """

    def __init__(self, geom, boundary_dim, boundary_point, time_dim=True):
        """
        :param geom: list of tuples, where each tuple represents the (min, max) value of a dimension in an order of (x, y, z, (t))
            -If time_dim is True, then the last dimension is the time dimension
        :param boundary_dim: boundary_dim: the dimension of the dimension. e.g., geom = [(0, 1), (0, 2)] and boundary_dim=1, then
            it's a boundary condition on the y (0, 2) dimension.
        :param boundary_point: The boundary point of the dimension. e.g, 0 then it's for the y=0 points
        :param time_dim: time_dim: bool, if there's a time domain.
        """

        self.geom = geom
        self.boundary_dim = boundary_dim
        self.boundary_point = boundary_point
        self.time_dim = time_dim

        if self.time_dim is True:
            assert self.boundary_dim <= len(geom) - 2
        else:
            assert self.boundary_dim <= len(geom) - 1

        assert np.isclose(self.boundary_point, self.geom[self.boundary_dim][0]) \
               or np.isclose(self.boundary_point, self.geom[self.boundary_dim][1])


    def derivative(self, inputs, outputs):
        return grad(outputs, inputs[:, self.boundary_dim],
            grad_outputs=torch.ones_like(outputs),
            retain_graph=True,
            create_graph=True)[0]

    @abstractmethod
    def error(self, inputs, outputs):
        """Returns the loss."""


class DirichletBC(BC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom,  boundary_dim, boundary_point, time_dim=True, func=None):
        super().__init__(geom, boundary_dim, boundary_point, time_dim)
        assert func is not None
        self.func = func

    def error(self, inputs, outputs):
        assert np.allclose(inputs[:, self.boundary_dim].cpu().detach().numpy(), self.boundary_point)
        values = self.func(inputs)
        if values.dim() == 2 and list(values.shape)[1] != 1:
            raise RuntimeError(
                "DirichletBC function should return an array of shape N by 1 for each "
                "component. Use argument 'component' for different output components."
            )
        return outputs - values


class NeumannBC(BC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom,  boundary_dim, boundary_point, time_dim=True, func=None):
        super().__init__(geom, boundary_dim, boundary_point, time_dim)
        assert func is not None
        self.func = func

    def error(self, inputs, outputs):
        assert np.allclose(inputs[:, self.boundary_dim].cpu().detach().numpy(), self.boundary_point)
        values = self.func(inputs)
        return self.derivative(inputs, outputs) - values


class PeriodicBC(BC):
    """Periodic boundary conditions on component_x."""
    def __init__(self, geom, boundary_dim, boundary_point, time_dim=True, derivative_order=0):
        super().__init__(geom, boundary_dim, boundary_point, time_dim,)
        self.derivative_order = derivative_order
        if derivative_order > 1:
            raise NotImplementedError(
                "PeriodicBC only supports derivative_order 0 or 1."
            )

    def error(self, inputs, outputs):
        inputs_left = inputs[0]
        inputs_right = inputs[1]
        outputs_left = outputs[0]
        outputs_right = outputs[1]

        assert np.allclose(inputs_left[:, self.boundary_dim].cpu().detach().numpy(), self.boundary_point) or \
               np.allclose(inputs_right[:, self.boundary_dim].cpu().detach().numpy(), self.boundary_point)

        if self.derivative_order == 0:
            yleft = outputs_left
            yright = outputs_right
        else:
            yleft = grad(outputs_left, inputs_left,
                         grad_outputs=torch.ones_like(outputs_left),
                         retain_graph=True,
                         create_graph=True)[0]

            yright = grad(outputs_right, inputs_right,
                         grad_outputs=torch.ones_like(outputs_right),
                         retain_graph=True,
                         create_graph=True)[0]
        return yleft - yright


class OperatorBC(BC):
    """General operator boundary conditions: func(inputs, outputs, X) = 0.

    Args:
        geom: ``Geometry``.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of `inputs`.
            `inputs` and `outputs` are the network input and output tensors,
            respectively; `X` are the NumPy array of the `inputs`.
        on_boundary: (x, Geometry.on_boundary(x)) -> True/False.

    """

    def __init__(self, geom, boundary_dim, boundary_point, time_dim, func):
        super().__init__(geom, boundary_dim, boundary_point, time_dim)
        self.func = func

    def error(self, inputs, outputs):
        return self.func(inputs, outputs)
