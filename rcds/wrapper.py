"""
Objective wrapper module.

"""

import torch

from typing import Tuple
from typing import Callable
from functools import wraps

class Wrapper():
    """
    Objective function wrapper.
    Can be used as a decorator for objective wrapping.

    Original objective is expected to return a (value, error) tuple of tensors
    If sf parameter is True, sign of the original objective returned value will be flipped
    If returned objective value is tensor(nan), upto additional nr evaluations will be performed
    The case when final objective value is tensor(nan) is expected to be handled elsewhere
    If error parameter is None, returned error is used, else it is replaced by error parameter in all subsequent evaluations
    If regularization parameters alpha_l1 and/or alpha_l2 are not zero, regularization penalty is added to the returned value

    Given an original objective: knobs -> (value, error), wrapped objective (expected to act on unit knobs)

    @Wrapper(...)
    def objective(knobs) -> (value, error):
        ...

    Wrapped objective must be first evaluated with None input or without arguments to attach attributes and methods, original objective is not evaluated

    Significance steps parameter dk is not used in objective evaluation
    It is attached to wrapper in units of rescaled knobs
    Parameter dk can be used during optimization to decide whether a particular knob is worthy to change
    For example, if suggested change for knob k_i is less than dk_i, old value can be kept
    This might be useful if knob change itself is expensive

    Parameters
    ----------
    cache: bool
        flag to cache knobs (original box), value and error
    sf: bool
        flag to flip objective value sign
    nr: int
        maximum number of retrials if returned objective value is tensor(nan)
    nk: int
        number of knobs
    lb: torch.Tensor
        lower bounds
    ub: torch.Tensor
        upper bounds
    dk: torch.Tensor
        significance steps
    error: float
        objective error value, if not None, replace original objective returned error
    alpha_l1: float
        l1 regularization factor
    alpha_l2: float
        l2 regularization factor
    dtype: torch.dtype
        data type
    device: torch.device
        data device

    Attributes
    ----------
    cache: bool
        flag to cache knobs (original box), value and error
    sf: bool
        flag to flip objective value sign
    nr: int
        maximum number of retrials if returned objective value is tensor(nan)
    nk: int
        number of knobs (attached)
    lb: torch.Tensor
        lower bounds (attached)
    ub: torch.Tensor
        upper bounds (attached)
    dk: torch.Tensor
        rescaled significance steps (attached)
    error: float
        objective error value, if not None, replace original objective returned error (attached)
    alpha_l1: float
        l1 regularization factor (attached)
    alpha_l2: float
        l2 regularization factor (attached)
    dtype: torch.dtype
        data type
    device: torch.device
        data device
    nan: torch.Tensor
        nan

    Methods
    ----------
    forward(self, knobs:torch.Tensor) -> torch.Tensor
        Rescale original input knobs into unit ouput knobs (attached).
    inverse(self, knobs:torch.Tensor) -> torch.Tensor
        Rescale unit input knobs into original output knobs (attached).
    save(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Return cached knobs, value and error data as a tuple of tensors (attached).
    __call__(self, objective:Callable[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Callable[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Return wrapped objective.

    """
    def __init__(self, *, cache:bool=False, sf:bool=False, nr:int=0, nk:int=1, lb:torch.Tensor=None, ub:torch.Tensor=None, dk:torch.Tensor=None,
                 error:float=None, alpha_l1:float=0.0, alpha_l2:float=0.0, dtype:torch.dtype=torch.float64, device:torch.device='cpu') -> None:
        """
        Objective function wrapper initialization.

        Parameters
        ----------
        cache: bool
            flag to cache knobs (original box), value and error
        sf: bool
            flag to flip objective value sign
        nr: int
            maximum number of retrials if returned objective value is tensor(nan)
        nk: int
            number of knobs
        lb: torch.Tensor
            lower bounds
        ub: torch.Tensor
            upper bounds
        dk: torch.Tensor
            significance steps
        error: float
            objective error value, if not None, replace original objective returned error
        alpha_l1: float
            l1 regularization factor
        alpha_l2: float
            l2 regularization factor
        dtype: torch.dtype
            data type
        device: torch.device
            data device

        Returns
        -------
        None

        """
        self.dtype, self.device = dtype, device

        self.nan = torch.tensor(float('nan'), dtype=self.dtype, device=self.device)

        self.cache = cache
        if not isinstance(self.cache, bool):
            raise TypeError(f'expected bool value for cache')

        self.sf = sf
        if not isinstance(self.sf, bool):
            raise TypeError(f'expected bool value for sf')

        self.nr = nr
        if not isinstance(self.nr, int):
            raise TypeError(f'expected int value for nr')
        if self.nr < 0:
            raise ValueError(f'expected nr >= 0')

        self.nk = nk
        if not isinstance(self.nk, int):
            raise TypeError(f'expected int value for nk')
        if self.nk < 1:
            raise ValueError(f'expected nk >= 1')

        self.lb = lb
        if self.lb is not None:
            if len(self.lb) != self.nk:
                raise Exception(f'lb size mismatch, expected length {self.nk}, got {len(self.lb)} on input')
            if not isinstance(self.lb, torch.Tensor):
                try:
                    self.lb = torch.tensor(self.lb, dtype=self.dtype, device=self.device)
                except TypeError:
                    raise TypeError(f'failed to convert lb data to torch')
            else:
                self.lb = self.lb.to(dtype=self.dtype, device=self.device)
        else:
            self.lb = torch.zeros(self.nk, dtype=self.dtype, device=self.device)

        self.ub = ub
        if self.ub is not None:
            if len(self.ub) != self.nk:
                raise Exception(f'ub size mismatch, expected length {self.nk}, got {len(self.ub)} on input')
            if not isinstance(self.ub, torch.Tensor):
                try:
                    self.ub = torch.tensor(self.ub, dtype=self.dtype, device=self.device)
                except TypeError:
                    raise TypeError(f'failed to convert ub data to torch')
            else:
                self.ub = self.ub.to(dtype=self.dtype, device=self.device)
        else:
            self.ub = torch.ones(self.nk, dtype=self.dtype, device=self.device)

        if (self.ub - self.lb <= 0).sum() != 0:
            raise Exception(f'bounds mismatch')

        self.dk = dk
        if self.dk is not None:
            if len(self.dk) != self.nk:
                raise Exception(f'dk size mismatch, expected length {self.nk}, got {len(self.dk)} on input')
            if not isinstance(self.dk, torch.Tensor):
                try:
                    self.dk = torch.tensor(self.dk, dtype=self.dtype, device=self.device)
                except TypeError:
                    raise Exception(f'failed to convert dk data to torch')
            else:
                self.dk = self.dk.to(dtype=self.dtype, device=self.device)
        else:
            self.dk = torch.zeros(self.nk, dtype=self.dtype, device=self.device)

        if (self.ub - self.lb <= self.dk).sum() != 0:
            raise Exception(f'dk values mismatch')

        self.dk /= self.ub - self.lb

        self.error = error
        if self.error is not None:
            if not isinstance(self.error, float):
                raise TypeError(f'expected float for error parameter')
            self.error = torch.tensor(self.error, dtype=self.dtype, device=self.device)

        self.alpha_l1 = alpha_l1
        if self.alpha_l1 is not None:
            if not isinstance(self.alpha_l1, float):
                raise TypeError(f'expected float for alpha_l1 parameter')
            self.alpha_l1 = torch.tensor(self.alpha_l1, dtype=self.dtype, device=self.device)

        self.alpha_l2 = alpha_l2
        if self.alpha_l2 is not None:
            if not isinstance(self.alpha_l2, float):
                raise TypeError(f'expected float for alpha_l2 parameter')
            self.alpha_l2 = torch.tensor(self.alpha_l2, dtype=self.dtype, device=self.device)


    def forward(self, knobs:torch.Tensor) -> torch.Tensor:
        """
        Rescale original input knobs into unit ouput knobs.

        Parameters
        ----------
        knobs: torch.Tensor
            input knobs to rescale

        Returns
        -------
        rescaled output knobs (torch.Tensor)

        """
        return (knobs - self.lb)/(self.ub - self.lb)


    def inverse(self, knobs:torch.Tensor) -> torch.Tensor:
        """
        Rescale unit input knobs into original output knobs.

        Parameters
        ----------
        knobs: torch.Tensor
            input knobs to rescale

        Returns
        -------
        rescaled output knobs (torch.Tensor)

        """
        return knobs*(self.ub - self.lb) + self.lb


    def save(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return cached knobs, value and error data as a tuple of tensors.

        Parameters
        ----------
        None

        Returns
        -------
        knobs, value, error (Tuple[torch.Tensor, torch.Tensor, torch.Tensor])

        """
        if self.cache:
            knobs = torch.tensor(self.cache_knobs, dtype=self.dtype, device=self.device)
            value = torch.tensor(self.cache_value, dtype=self.dtype, device=self.device)
            error = torch.tensor(self.cache_error, dtype=self.dtype, device=self.device)
            return knobs, value, error


    def __call__(self, objective:Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]):
        """
        Return wrapped objective.

        Evaluate result without arguments to attach attributes
        Evaluate result with unit knobs to return (value, error) tuple

        Parameters
        ----------
        objective: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            original objective, input knobs are in original cube

        Returns
        -------
        wrapped objective (Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]), input knobs are in unit cube

        """
        @wraps(objective)
        def wrapper(knobs:torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:

            if knobs is None:

                wrapper.__dict__ = self.__dict__

                wrapper.forward = self.forward
                wrapper.inverse = self.inverse

                wrapper.save = self.save

                wrapper.counter = 0

                wrapper.n:int = 0
                wrapper.m:int = 0

                wrapper.cache_knobs:list = []
                wrapper.cache_value:list = []
                wrapper.cache_error:list = []

            if knobs is not None:

                knobs = wrapper.inverse(knobs)
                value = wrapper.nan

                wrapper.n += 1

                while torch.isnan(value):
                    wrapper.m += 1
                    value, error = objective(knobs)
                    wrapper.counter += 1
                    if wrapper.counter > wrapper.nr:
                        break

                wrapper.counter = 0

                if wrapper.error is not None:
                    error = wrapper.error

                if not torch.isnan(value):
                    value += wrapper.alpha_l1*(knobs**1).abs().sum()
                    value += wrapper.alpha_l2*(knobs**2).sum()

                value = value if not wrapper.sf else -value

                if wrapper.cache:
                    wrapper.cache_knobs.append(knobs.cpu().numpy().tolist())
                    wrapper.cache_value.append(value.cpu().numpy().tolist())
                    wrapper.cache_error.append(error.cpu().numpy().tolist())

                return value, error

        return wrapper

def main():
    pass

if __name__ == '__main__':
    main()