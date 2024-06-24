from __future__ import annotations
from typing import Optional, Tuple, TypeVar, Union, overload
import warnings

import torch
import torch.nn as nn

from torch import Tensor

from ..solver import *

__all__ = [
    'Parameter'
]

def _raise(error):
    raise error

T = TypeVar('T', bound='Parameter')
class Parameter(nn.Parameter):
    def __new__(cls, data, requires_symbolic=False, solver: Optional[Solver]=None):
        assert isinstance(solver, Solver) or solver is None
        if isinstance(data, Parameter):
            obj = super().__new__(cls, data.data, data.requires_grad)
            assert isinstance(data.solver, Solver) or data.solver is None
            solver = solver or data.solver
            requires_symbolic = requires_symbolic or data.requires_symbolic

        elif isinstance(data, nn.Parameter):
            obj = super().__new__(cls, data.data, data.requires_grad)

        elif isinstance(data, Tensor):
            obj = super().__new__(cls, data, data.requires_grad)

        else:
            raise NotImplementedError(
                f"unsupported creating symbolic.Parameter from {type(data)}"
            )

        obj._requires_symbolic = False
        obj._solver = None
        obj._lb = None
        obj._ub = None
        obj._symbolic_data = None
        obj._mask = None
        obj._concrete_cache = None
        obj._delta = None
        obj._ordered_indices = None
        obj._fused_for_argmax = dict()
        obj._name =None
        obj.to(solver=solver).requires_symbolic_(requires_symbolic)
        return obj

    def __repr__(self):
        return (
            'Symbolic Parameter containing:\n'
            f'{super(nn.Parameter, self).__repr__()},\n'
            f'requires_symbolic={self.requires_symbolic}, '
            f'solver={self.solver},'
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(
                nn.Parameter(self.data.clone(memory_format=torch.preserve_format), self.requires_grad),
                requires_symbolic=self.requires_symbolic,
                solver=self.solver
            )
            result._lb = self._lb
            result._ub = self._ub
            result._mask = self._mask
            memo[id(self)] = result
            return result

    def update_(self, src=None):
        if src is None:
            src = self

        with torch.no_grad():
            if self._concrete_cache is not None:
                assert (self.cpu().detach().numpy() == self._concrete_cache).all(), \
                    "network weights is modified during repair."
            self.data[...] = src.symbolic().evaluate().to(self.data.device, self.data.dtype)
            self._concrete_cache = None
            self._delta = None

        return self

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        if self.requires_symbolic and solver is None:
            raise NotImplementedError(
                "Please unset a parameter's requires_symbolic before reset "
                "its solver to `None`. For example: "
                "`.requires_symbolic_(False).to(solver=None)`"
            )
        self._solver = solver

    @property
    def delta(self) -> SymbolicLPArray:
        if self._delta is None:
            self._delta = self.symbolic()[self.mask] - self.concrete()[self.mask]
            # out = np.zeros(self.shape).astype(object)
            # out[self.mask] = self.symbolic()[self.mask] - self.concrete()[self.mask]
            # self._delta = out.view(type(self.symbolic())).to(self.solver)

        return self._delta

    @property
    def LBs(self):
        return np.fromiter(map(
            (lambda v: v.LB if isinstance(v, LightningVar) else v),
            self.symbolic().flat
        ), dtype=torch_dtype_to_numpy(self.dtype)).reshape(tuple(self.shape))

    @property
    def UBs(self):
        return np.fromiter(map(
            (lambda v: v.UB if isinstance(v, LightningVar) else v),
            self.symbolic().flat
        ), dtype=torch_dtype_to_numpy(self.dtype)).reshape(tuple(self.shape))

    @property
    def symbolic_data(self):
        assert self.solver is not None and self.requires_symbolic, \
            "accessing the symbolic data of a non-symbolic parameter."

        """ Create symbolic data if not exits.
        """
        if self._symbolic_data is None or self._symbolic_data.solver is not self.solver:
            self._fused_for_argmax = dict()
            self._symbolic_data = self.solver.reals(tuple(self.shape), name=self._name, mask=self.mask, lb=self.lb, ub=self.ub)

            # NOTE(anonymous): setting those bounds is super slow.
            if self.lb is not None:
                for v, c in zip(self._symbolic_data[self.mask].flat, self.concrete()[self.mask].flat):
                    v.LB += c
            if self.ub is not None:
                for v, c in zip(self._symbolic_data[self.mask].flat, self.concrete()[self.mask].flat):
                    v.UB += c

            self.solver.update()

            """ Fill non-symbolic entries with the latest concrete data. """
            if self.mask is not None:
                self._symbolic_data[~self.mask] = self.concrete()[~self._mask]

            self._delta = None

        # """ Translate to self.solver if the existing symbolic data is not on it. """
        # if self._symbolic_data.solver is not self.solver:
        #     self._symbolic_data = self._symbolic_data.to(self.solver)

        return self._symbolic_data

    def concrete(self):
        if self.device != torch.device('cpu'):
            if self._concrete_cache is None:
                self._concrete_cache = self.cpu().detach().numpy()
            if is_debugging():
                assert (self.cpu().detach().numpy() == self._concrete_cache).all()
            return self._concrete_cache
        else:
            return self.detach().numpy()

    @property
    def masked(self):
        if self.mask is not None:
            return self[torch.from_numpy(self.mask)]
        else:
            return self

    @masked.setter
    def masked(self, val):
        if self.mask is not None:
            self[torch.from_numpy(self.mask)] = val
        else:
            self[...] = val

    def symbolic(self):
        return self.symbolic_data

    def array_if_symbolic(self) -> SymbolicArray:
        if self.requires_symbolic:
            return self.symbolic_data

        return self

    def array(self):
        if self.requires_symbolic:
            return self.array_if_symbolic()

        return self.concrete()

    def to(self, *args, **kwargs) -> T:
        if len(args) == 1 and isinstance(args[0], Solver):
            self.solver = args[0]
            return self

        elif 'solver' in kwargs:
            self.solver = kwargs['solver']
            return self

        return super().to(*args, **kwargs)

    @property
    def requires_symbolic(self):
        return self._requires_symbolic

    @requires_symbolic.setter
    def requires_symbolic(self, mode):
        self._requires_symbolic = mode
        if self.requires_symbolic:
            self.to(solver=Solver.override() or self.solver or Solver.fallback() or _raise(
                RuntimeError(
                    "No available solver for `.requires_symbolic_(True)`. Try "
                    "`.to(solver).requires_symbolic_()` or to create a (override "
                    "or fallback) solver context `with solver: ...`. "
                )
            ))

    @property
    def lb(self):
        return self._lb

    # @lb.setter
    # def lb(self, value): ...
    #     if self.lb != value:
    #         self.outdated = True
    #     self._lb = value

    @property
    def ub(self):
        return self._ub

    # @ub.setter
    # def ub(self, value): ...
    #     if self.ub != value:
    #         self.outdated = True
    #     self._ub = value

    @property
    def bound(self):
        return self.lb, self.ub

    @bound.setter
    def bound(self, value):
        self.lb, self.ub = value

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    def requires_symbolic_(self: T, mode: bool=True, lb=None, ub=None, mask=None, name=None) -> T:
        """ Mark this parameter as symbolic or not. """
        self.requires_symbolic = mode
        self._lb = lb
        self._ub = ub
        self._name=name
        self.mask = mask
        return self
