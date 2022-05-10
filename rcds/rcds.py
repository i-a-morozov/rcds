"""
RCDS minimization module.

"""

import torch
import numpy
import logging
import warnings

from typing import Tuple
from typing import Callable
from time import sleep

from statsmodels.api import WLS
from scipy.optimize import minimize

from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .statistics import median, biweight_midvariance
from .wrapper import Wrapper

class RCDS():
    """
    RCDS single (noisy) objective minimization.

    Parameters
    ----------
    objective: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        (wrapped) objective
    ni: int
        max number of iterations
    np: int
        number of points around current bracketed minimum (on each side) to use for parabola fit
    ns: int
        number of points to add with uniform sampling in bracketed interval (defined by np)
    sf: float
        initial step fraction
    sf_min: float
        min step fraction
    sf_max: float
        max step fraction
    dr: float
        step fraction decay rate (applied after each iteration)
    fc: float
        factor to use in comparison of objective values
    ft: float
        threshold factor to for oulier detection
    fr: float
        factor to use in comparison of close points
    otol: float
        objective tolerance
    ktol: float
        knobs tolerance
    epsilon: float
        float epsilon
    file: str
        output file name

    Attributes
    ----------
    objective: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        (wrapped) objective
    ni: int
        max number of iterations
    np: int
        number of points around current bracketed minimum (on each side) to use for parabola fit
    ns: int
        number of points to add with uniform sampling in bracketed interval (defined by np)
    sf: float
        initial step fraction
    sf_min: float
        min step fraction
    sf_max: float
        max step fraction
    dr: float
        step fraction decay rate (applied after each iteration)
    fc: float
        factor to use in comparison of objective values
    ft: float
        threshold factor to for oulier detection
    fr: float
        factor to use in comparison of close points
    otol: float
        objective tolerance
    ktol: float
        knobs tolerance
    epsilon: float
        float epsilon
    file: str
        output file name
    nk: int
        number of knobs
    lb: torch.Tensor
        lower bounds
    ub: torch.Tensor
        upper bounds
    error: torch.Tensor
        objective error value, if not None, replace original objective returned error
    alpha_l1: torch.Tensor
        l1 regularization factor
    alpha_l2: torch.Tensor
        l2 regularization factor
    golden: torch.Tensor
        golden ratio
    nan: torch.Tensor
        nan
    eval_count: int
        number of objective evaluations
    iter_count: int
        number of iterations
    cache_knobs: list
        knobs cache
    cache_value: list
        value cache
    cache_error: list
        error cache

    Methods
    ----------
    forward(self, knobs:torch.Tensor) -> torch.Tensor
        Rescale original input knobs into unit ouput knobs.
    inverse(self, knobs:torch.Tensor) -> torch.Tensor
        Rescale unit input knobs into original output knobs.
    append(self, knobs:torch.Tensor, value:torch.Tensor, error:torch.Tensor) -> None
        Append input knobs, value and error to cache.
    close_knobs(self, probe:torch.Tensor, other:torch.Tensor) -> bool
        Check whether knobs are close based on knobs significance defined by objective.
    alter_knobs(self, probe:torch.Tensor, other:torch.Tensor) -> torch.Tensor
        Alter knobs components based on knobs significance defined by objective.
    termination_knobs(self, probe:torch.Tensor, other:torch.Tensor) -> bool
        Check termination condition for knobs.
    termination_value(self, value_probe:torch.Tensor, value_other:torch.Tensor, error_probe:torch.Tensor, error_other:torch.Tensor) -> bool
        Check termination condition for objective value.
    interval(knobs:torch.Tensor, vector:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        Return interval of parameter values (neg, pos) for which all knobs remain in the unit cube (staticmethod).
    to_cube(knobs:torch.Tensor) -> torch.Tensor
        Move knobs to the unit cube (staticmethod).
    on_cube(self, knobs:torch.Tensor) -> bool
        Check whether knobs are on the unit cube.
    bracket(self, sf:torch.Tensor, knobs:torch.Tensor, value:torch.Tensor, error:torch.Tensor, vector:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Bracket objective minimum for given initial knobs along a given search direction.
    parabola(self, vector:torch.Tensor, table_alpha:torch.Tensor, table_knobs:torch.Tensor, table_value:torch.Tensor, table_error:torch.Tensor, *, detector:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Find 1d minimum using (weighted) parabola fit.
    minimize_parabola(self, sf:torch.Tensor, knobs:torch.Tensor, value:torch.Tensor, error:torch.Tensor, vector:torch.Tensor, *, sample:bool=True, detector:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Parabola line minimization (bracket and 1d minimization using parabola fit).
    minimize_gp(self, sf:torch.Tensor, knobs:torch.Tensor, value:torch.Tensor, error:torch.Tensor, vector:torch.Tensor, *, no_ei:int=8, no_ucb:int=2, nr:int=64, rs:int=256, beta:float=0.5, use_parabola:bool=True, np:int=1, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        GP line minimization (GP and 1d minimization using 3 point parabola fit).
    fit_rcds(self, knobs:torch, matrix:torch.Tensor, *, minimize:Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]=None, termination:bool=True, verbose:bool=False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        RCDS minimization.
    adjust_cube(self, *,  data:torch.Tensor=None, extend:bool=False, factor:float=5.0, center_estimator:Callable[[torch.Tensor], torch.Tensor]=median, spread_estimator:Callable[[torch.Tensor], torch.Tensor]=biweight_midvariance) -> Tuple[torch.Tensor, torch.Tensor]
        Adjust search box using cached knobs dispersion.
    fit_scipy(self, knobs:torch.Tensor, method:str='powell', **kwargs)
        Scipy minimization wrapper for bounded methods.

    """
    def __init__(self, /, objective:Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], *,
                 ni:int=32, np:int=2, ns:int=4, sf:float=0.01, sf_min:float=0.001, sf_max:float=0.1,
                 dr:float=1.0, fc:float=3.0, ft:float=5.0, fr:float=0.1,
                 otol:float=1.0E-06, ktol:float=1.0E-6, epsilon:float=1.0E-16, file:str=None) -> None:
        """
        RCDS initialization.

        Note, objective is assumed to be pre-evaluated (not evaluated in __init__)

        Parameters
        ----------
        objective: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            (wrapped) objective
        ni: int
            max number of iterations
        np: int
            number of points around current bracketed minimum (on each side) to use for parabola fit
        ns: int
            number of points to add with uniform sampling in bracketed interval (defined by np)
        sf: float
            initial step fraction
        sf_min: float
            min step fraction
        sf_max: float
            max step fraction
        dr: float
            step fraction decay rate (applied after each iteration)
        fc: float
            factor to use in comparison of objective values
        ft: float
            threshold factor to for oulier detection
        fr: float
            factor to use in comparison of close points
        otol: float
            objective tolerance
        ktol: float
            knobs tolerance
        epsilon: float
            float epsilon
        file: str
            output file name

        Returns
        -------
        None

        """
        self.objective = objective

        self.dtype, self.device = self.objective.dtype, self.objective.device

        self.nan = self.objective.nan

        self.nk = self.objective.nk
        self.lb = self.objective.lb
        self.ub = self.objective.ub
        self.dk = self.objective.dk

        self.error = self.objective.error

        self.alpha_l1 = self.objective.alpha_l1
        self.alpha_l2 = self.objective.alpha_l2

        self.forward = self.objective.forward
        self.inverse = self.objective.inverse

        self.golden = 0.5*(1.0 + torch.tensor(5.0, dtype=self.dtype, device=self.device).sqrt())

        self.ni = ni
        if not isinstance(self.ni, int):
            raise TypeError(f'expected int value for ni')
        if self.ni < 1:
            raise ValueError(f'expected ni >= 1')

        self.np = np
        if not isinstance(self.np, int):
            raise TypeError(f'expected int value for np')
        if self.np < 1:
            raise ValueError(f'expected np >= 1')

        self.ns = ns
        if not isinstance(self.ns, int):
            raise TypeError(f'expected int value for ns')
        if self.ns < 0:
            raise ValueError(f'expected ns >= 0')

        self.sf = sf
        if not isinstance(self.sf, float):
            raise TypeError(f'expected float value for sf')

        self.sf = torch.tensor(self.sf, dtype=self.dtype, device=self.device)
        if self.sf <= 0:
            raise ValueError(f'expected sf > 0.0')

        self.sf_min = sf_min
        if not isinstance(self.sf_min, float):
            raise TypeError(f'expected float value for sf_min')

        self.sf_min = torch.tensor(self.sf_min, dtype=self.dtype, device=self.device)
        if self.sf_min <= 0:
            raise ValueError(f'expected sf_min > 0.0')

        self.sf_max = sf_max
        if not isinstance(self.sf_max, float):
            raise TypeError(f'expected float value for sf_max')

        self.sf_max = torch.tensor(self.sf_max, dtype=self.dtype, device=self.device)
        if self.sf_max <= 0:
            raise ValueError(f'expected sf_max > 0.0')

        if self.sf_min >= self.sf_max:
            raise Exception(f'expected sf_min < sf_max')

        if self.sf >= self.sf_max:
            raise Exception(f'expected sf < sf_max')

        if self.sf <= self.sf_min:
            raise Exception(f'expected sf > sf_min')

        self.dr = dr
        if not isinstance(self.dr, float):
            raise TypeError(f'expected float value for dr')

        self.dr = torch.tensor(self.dr, dtype=self.dtype, device=self.device)
        if self.dr < 0.0:
            raise ValueError(f'expected dr >= 0.0')
        if self.dr > 1.0:
            raise ValueError(f'expected dr <= 1.0')

        self.fc = fc
        if not isinstance(self.fc, float):
            raise TypeError(f'expected float value for fc')

        self.fc = torch.tensor(self.fc, dtype=self.dtype, device=self.device)
        if self.fc <= 0.0:
            raise ValueError(f'expected fc > 0.0')

        self.ft = ft
        if not isinstance(self.ft, float):
            raise TypeError(f'expected float value for ft')

        self.ft = torch.tensor(self.ft, dtype=self.dtype, device=self.device)
        if self.ft <= 0.0:
            raise ValueError(f'expected ft > 0.0')

        self.fr = fr
        if not isinstance(self.fr, float):
            raise TypeError(f'expected float value for fr')

        self.fr = torch.tensor(self.fr, dtype=self.dtype, device=self.device)
        if self.fr <= 0.0:
            raise ValueError(f'expected fr > 0.0')
        if self.fr >= 1.0:
            raise ValueError(f'expected fr < 1.0')

        self.otol = otol
        if not isinstance(self.otol, float):
            raise TypeError(f'expected float value for otol')

        self.otol = torch.tensor(self.otol, dtype=self.dtype, device=self.device)
        if self.otol <= 0.0:
            raise ValueError(f'expected otol > 0.0')

        self.ktol = ktol
        if not isinstance(self.ktol, float):
            raise TypeError(f'expected float value for ktol')

        self.ktol = torch.tensor(self.ktol, dtype=self.dtype, device=self.device)
        if self.ktol <= 0.0:
            raise ValueError(f'expected ktol > 0.0')

        self.epsilon = epsilon
        if not isinstance(self.epsilon, float):
            raise TypeError(f'expected float value for epsilon')

        self.epsilon = torch.tensor(self.epsilon, dtype=self.dtype, device=self.device)
        if self.epsilon <= 0.0:
            raise ValueError(f'expected epsilon > 0.0')

        self.file = file
        if not isinstance(self.file, str) and self.file is not None:
            raise TypeError(f'expected file value for file')
        if self.file:
            logging.basicConfig(filename=self.file, format='%(asctime)s %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

        self.eval_count:int = 0
        self.iter_count:int = 0

        self.cache_knobs:list = []
        self.cache_value:list = []
        self.cache_error:list = []


    def append(self, knobs:torch.Tensor, value:torch.Tensor, error:torch.Tensor) -> None:
        """
        Append input knobs, value and error to cache.

        Note, increments objective evaluation counter, optional logging

        Parameters
        ----------
        knobs, value, error: torch.Tensor
            input knobs, value and error

        Returns
        -------
        None

        """
        self.eval_count += 1

        if self.file:
            logging.info(''.join(f'{knob:>.9E} ' for knob in knobs) + f'{value:>.9E} ' + f'{error:>.9E} ')

        self.cache_knobs.append(knobs.cpu().numpy().tolist())
        self.cache_value.append(value.cpu().numpy().tolist())
        self.cache_error.append(error.cpu().numpy().tolist())


    def close_knobs(self, probe:torch.Tensor, other:torch.Tensor) -> bool:
        """
        Check whether knobs are close based on knobs significance defined by objective.

        Parameters
        ----------
        probe, other: torch.Tensor
            knobs to compare

        Returns
        -------
        check result (bool)

        """
        return all((probe - other).abs() <= self.epsilon + self.dk)


    def alter_knobs(self, probe:torch.Tensor, other:torch.Tensor) -> torch.Tensor:
        """
        Alter knobs components based on knobs significance defined by objective.

        Replace close elements of other with corresponding elements of probe

        Parameters
        ----------
        probe: torch.Tensor
            input knobs
        other: torch.Tensor
            output knobs

        Returns
        -------
        altered knobs (torch.Tensor)

        """
        mask = (probe - other).abs() <= self.epsilon + self.dk
        other[mask] = probe[mask]
        return other


    def termination_knobs(self, probe:torch.Tensor, other:torch.Tensor) -> bool:
        """
        Check termination condition for knobs.

        Parameters
        ----------
        probe, other: torch.Tensor
            knobs to compare

        Returns
        -------
        check result (bool)

        """
        return self.close_knobs(probe, other) or ((probe - other).norm() <= self.epsilon + self.ktol).cpu().item()


    def termination_value(self, value_probe:torch.Tensor, value_other:torch.Tensor,
                          error_probe:torch.Tensor, error_other:torch.Tensor) -> bool:
        """
        Check termination condition for objective value.

        Parameters
        ----------
        value_probe, value_other: torch.Tensor
            values to compare
        error_probe, error_other: torch.Tensor
            corresponding errors

        Returns
        -------
        check result (bool)

        """
        return ((value_probe.abs() - value_other.abs()).abs() <= self.epsilon + self.otol + (error_probe**2 + error_other**2).sqrt()).cpu().item()


    @staticmethod
    def interval(knobs:torch.Tensor, vector:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return interval of parameter values (neg, pos) for which all knobs remain in the unit cube.

        Parameters
        ----------
        knobs: torch.Tensor
            initial knobs (in the unit cube)
        vector: torch.Tensor
            direction vector

        Returns
        -------
        valid parameter interval (Tuple[torch.Tensor, torch.Tensor])

        """
        interval = torch.stack([(0.0 - knobs)/vector, (1.0 - knobs)/vector]).T.sort().values
        interval = interval[torch.any(interval.isnan(), dim=-1).logical_not()]
        interval = interval[torch.any(interval.isinf(), dim=-1).logical_not()]
        neg, pos = interval.T
        return neg.max(), pos.min()


    @staticmethod
    def to_cube(knobs:torch.Tensor) -> torch.Tensor:
        """
        Move knobs to the unit cube.

        Parameters
        ----------
        knobs: torch.Tensor
            input knobs

        Returns
        -------
        changed knobs (torch.Tensor)

        """
        knobs[knobs < 0.0] = 0.0
        knobs[knobs > 1.0] = 1.0
        return knobs


    def on_cube(self, knobs:torch.Tensor) -> bool:
        """
        Check whether knobs are on the unit cube.

        Parameters
        ----------
        knobs: torch.Tensor
            input knobs

        Returns
        -------
        check result (bool)

        """
        return any((knobs - 0.0).abs() < self.epsilon) or any((knobs - 1.0).abs() < self.epsilon)


    def bracket(self, sf:torch.Tensor, knobs:torch.Tensor, value:torch.Tensor, error:torch.Tensor,
                vector:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Bracket objective minimum for given initial knobs along a given search direction.

        Parameters
        ----------
        sf: torch.Tensor
            initial step fraction
        knobs: torch.Tensor
            initial knobs
        value: torch.Tensor
            initial value
        error: torch.Tensor
            initial error
        vector: torch.Tensor
            search direction vector

        Returns
        -------
        table_alpha, table_knobs, table_value, table_error (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])

        """
        alpha_lb, alpha_ub = self.interval(knobs, vector)

        step_min = self.sf_min*(alpha_ub - alpha_lb)
        step_max = self.sf_max*(alpha_ub - alpha_lb)

        step = sf*(alpha_ub - alpha_lb)

        table_alpha, table_knobs, table_value, table_error = [], [], [], []

        table_alpha.append(torch.zeros_like(step))
        table_knobs.append(knobs)
        table_value.append(value)
        table_error.append(error)

        alpha_min = torch.clone(torch.zeros_like(step))
        knobs_min = torch.clone(knobs)
        value_min = torch.clone(value)
        error_min = torch.clone(error)

        if step >= step_max:
            step = step_max
        if step <= step_min:
            step = step_min

        knobs_pos = self.to_cube(knobs + vector*step)
        knobs_pos = self.alter_knobs(knobs, knobs_pos)
        value_pos, error_pos = self.objective(knobs_pos)

        self.append(knobs_pos, value_pos, error_pos)
        table_alpha.append(torch.clone(step))
        table_knobs.append(torch.clone(knobs_pos))
        table_value.append(torch.clone(value_pos))
        table_error.append(torch.clone(error_pos))

        if torch.isnan(value_pos):
            table_alpha = torch.stack(table_alpha) - alpha_min
            table_knobs = torch.stack(table_knobs)
            table_value = torch.stack(table_value)
            table_error = torch.stack(table_error)
            return table_alpha, table_knobs, table_value, table_error

        if value_pos - value_min < self.epsilon + (error_pos**2 + error_min**2).sqrt():
            alpha_min = torch.clone(step)
            knobs_min = torch.clone(knobs_pos)
            value_min = torch.clone(value_pos)
            error_min = torch.clone(error_pos)

        while value_pos - value_min <  self.epsilon + self.fc*(error_pos**2 + error_min**2).sqrt():

            if step == alpha_ub:
                break

            if step < step_max:
                step *= 1.0 + self.golden
            else:
                step += step_max

            if step > alpha_ub:
                step = torch.clone(alpha_ub)

            knobs_pos = self.to_cube(knobs + vector*step)
            knobs_pos = self.alter_knobs(knobs, knobs_pos)
            value_pos, error_pos = self.objective(knobs_pos)

            self.append(knobs_pos, value_pos, error_pos)
            table_alpha.append(torch.clone(step))
            table_knobs.append(torch.clone(knobs_pos))
            table_value.append(torch.clone(value_pos))
            table_error.append(torch.clone(error_pos))

            if torch.isnan(value_pos):
                table_alpha = torch.stack(table_alpha) - alpha_min
                table_knobs = torch.stack(table_knobs)
                table_value = torch.stack(table_value)
                table_error = torch.stack(table_error)
                return table_alpha, table_knobs, table_value, table_error

            if value_pos - value_min < self.epsilon + (error_pos**2 + error_min**2).sqrt():
                alpha_min = torch.clone(step)
                knobs_min = torch.clone(knobs_pos)
                value_min = torch.clone(value_pos)
                error_min = torch.clone(error_pos)

        if value - value_min > self.epsilon + self.fc*(error**2 + error_min**2).sqrt():
            table_alpha = torch.stack(table_alpha) - alpha_min
            table_knobs = torch.stack(table_knobs)
            table_value = torch.stack(table_value)
            table_error = torch.stack(table_error)
            return table_alpha, table_knobs, table_value, table_error

        step = -sf*(alpha_ub - alpha_lb)

        if step.abs() >= step_max:
            step = -step_max
        if step.abs() <= step_min:
            step = -step_min

        knobs_neg = self.to_cube(knobs + vector*step)
        knobs_neg = self.alter_knobs(knobs, knobs_neg)
        value_neg, error_neg = self.objective(knobs_neg)

        self.append(knobs_neg, value_neg, error_neg)
        table_alpha.append(torch.clone(step))
        table_knobs.append(torch.clone(knobs_neg))
        table_value.append(torch.clone(value_neg))
        table_error.append(torch.clone(error_neg))

        if torch.isnan(value_neg):
            table_alpha = torch.stack(table_alpha) - alpha_min
            table_knobs = torch.stack(table_knobs)
            table_value = torch.stack(table_value)
            table_error = torch.stack(table_error)
            return table_alpha, table_knobs, table_value, table_error

        if value_neg - value_min < self.epsilon + (error_neg**2 + error_min**2).sqrt():
            alpha_min = torch.clone(step)
            knobs_min = torch.clone(knobs_neg)
            value_min = torch.clone(value_neg)
            error_min = torch.clone(error_neg)

        while value_neg - value_min <  self.epsilon + self.fc*(error_neg**2 + error_min**2).sqrt():

            if step == alpha_lb:
                break

            if step.abs() < step_max:
                step *= 1.0 + self.golden
            else:
                step -= step_max

            if step < alpha_lb:
                step = torch.clone(alpha_lb)

            knobs_neg = self.to_cube(knobs + vector*step)
            knobs_neg = self.alter_knobs(knobs, knobs_neg)
            value_neg, error_neg = self.objective(knobs_neg)

            self.append(knobs_neg, value_neg, error_neg)
            table_alpha.append(torch.clone(step))
            table_knobs.append(torch.clone(knobs_neg))
            table_value.append(torch.clone(value_neg))
            table_error.append(torch.clone(error_neg))

            if torch.isnan(value_neg):
                table_alpha = torch.stack(table_alpha) - alpha_min
                table_knobs = torch.stack(table_knobs)
                table_value = torch.stack(table_value)
                table_error = torch.stack(table_error)
                return table_alpha, table_knobs, table_value, table_error

            if value_neg - value_min < self.epsilon + (error_neg**2 + error_min**2).sqrt():
                alpha_min = torch.clone(step)
                knobs_min = torch.clone(knobs_neg)
                value_min = torch.clone(value_neg)
                error_min = torch.clone(error_neg)

        table_alpha = torch.stack(table_alpha) - alpha_min
        table_knobs = torch.stack(table_knobs)
        table_value = torch.stack(table_value)
        table_error = torch.stack(table_error)

        index = table_alpha.argsort()

        return table_alpha[index], table_knobs[index], table_value[index], table_error[index]


    def parabola(self, vector:torch.Tensor,
                 table_alpha:torch.Tensor, table_knobs:torch.Tensor, table_value:torch.Tensor, table_error:torch.Tensor, *,
                 sample:bool=True,
                 detector:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find 1d minimum using (weighted) parabola fit.

        If detector is not None, detector function is applied to fit residuals and is assumed to return a mask
        This mask is used to alter weights and refit parabola

        Parameters
        ----------
        vector: torch.Tensor
            search direction
        table_alpha: torch.Tensor
            alpha table
        table_knobs: torch.Tensor
            knobs table
        table_value: torch.Tensor
            value table
        table_error: torch.Tensor
            error table
        sample: bool
            flag to add additional samples
        detector:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            detector(residual, self.ft, self.otol)
            function to use for outlier cleaning, applied to fit residuals, assumed to return weights for each sample point

        Returns
        -------
        knobs, value, error (Tuple[torch.Tensor, torch.Tensor, torch.Tensor])

        """
        index = table_value.argmin()
        start = table_knobs[index]

        table_alpha = table_alpha[max(0, index - self.np) : index + self.np + 1]
        table_knobs = table_knobs[max(0, index - self.np) : index + self.np + 1]
        table_value = table_value[max(0, index - self.np) : index + self.np + 1]
        table_error = table_error[max(0, index - self.np) : index + self.np + 1]

        alpha_lb, *_, alpha_ub = table_alpha

        if sample:

            alpha_size = self.fr*(alpha_ub - alpha_lb)
            alpha_grid = torch.linspace(alpha_lb, alpha_ub, self.ns + 2, dtype=self.dtype, device=self.device)[1:-1]
            alpha_mask = torch.stack([(table_alpha - alpha).abs().min() > alpha_size for alpha in alpha_grid])
            alpha_grid = alpha_grid[alpha_mask]

            for alpha in alpha_grid:

                knobs = self.to_cube(start + vector*alpha)
                knobs = self.alter_knobs(start, knobs)
                value, error = self.objective(knobs)

                self.append(knobs, value, error)
                table_alpha = torch.cat((table_alpha, alpha.unsqueeze(0)))
                table_knobs = torch.cat((table_knobs, knobs.unsqueeze(0)))
                table_value = torch.cat((table_value, value.unsqueeze(0)))
                table_error = torch.cat((table_error, error.unsqueeze(0)))

                if torch.isnan(value):
                    return knobs, value, error

        X = torch.stack([table_alpha**2, table_alpha, torch.ones_like(table_alpha)]).T.cpu().numpy()
        y = table_value.cpu().numpy()
        w = (1.0/table_error**2).nan_to_num(posinf=1.0)
        w = w/w.sum().cpu().numpy()
        fit = WLS(y, X, weights=w).fit()
        a, b, c = fit.params

        if detector is not None:

            residual = torch.tensor(fit.resid, dtype=self.dtype, device=self.device)
            mask = detector(residual, self.ft, self.otol)

            if mask.sum() == 0:
                mask = mask.logical_not()

            w = (mask/table_error**2).nan_to_num(posinf=1.0)
            w = w/w.sum().cpu().numpy()
            fit = WLS(y, X, weights=w).fit()
            a, b, c = fit.params

        if abs(a) >= self.epsilon:
            alpha = torch.tensor(-b/(2.0*a), dtype=self.dtype, device=self.device)
        else:
            index = table_value.argmin()
            alpha = table_alpha[index]

        if alpha > alpha_ub:
            alpha = alpha_ub
        if alpha < alpha_lb:
            alpha = alpha_lb

        knobs = self.to_cube(start + vector*alpha)
        knobs = self.alter_knobs(start, knobs)
        value, error = self.objective(knobs)

        self.append(knobs, value, error)

        return knobs, value, error


    def minimize_parabola(self, sf:torch.Tensor, knobs:torch.Tensor, value:torch.Tensor, error:torch.Tensor, vector:torch.Tensor, *,
                          detector:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]=None,
                          **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parabola line minimization (bracket and 1d minimization using parabola fit).

        Parameters
        ----------
        sf: torch.Tensor
            initial step fraction
        knobs: torch.Tensor
            initial knobs
        value: torch.Tensor
            initial value
        error: torch.Tensor
            initial error
        vector: torch.Tensor
            search direction
        detector:Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
            function to use for outlier cleaning, assumed to return weights for each sample point

        Returns
        -------
        knobs, value, error (Tuple[torch.Tensor, torch.Tensor, torch.Tensor])

        """
        alpha, knobs, value, error = self.bracket(sf, knobs, value, error, vector)
        return self.parabola(vector, alpha, knobs, value, error, detector=detector)


    def minimize_gp(self, sf:torch.Tensor, knobs:torch.Tensor, value:torch.Tensor, error:torch.Tensor, vector:torch.Tensor, *,
                    no_ei:int=8, no_ucb:int=2, nr:int=64, rs:int=256, beta:float=1.0, use_parabola:bool=True, np:int=1,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GP line minimization (GP and 1d minimization using 3 point parabola fit).

        Parameters
        ----------
        sf: torch.Tensor
            initial step fraction (not used)
        knobs: torch.Tensor
            initial knobs
        value: torch.Tensor
            initial value
        error: torch.Tensor
            initial error
        vector: torch.Tensor
            search direction
        no_ei: int
            number of observations to perform with ei af
        no_ucb: int
            number of observations to perform with ucb af
        nr: int
            number of restarts
        rs: int
            number of raw samples
        beta: float
            ucb beta factor
        use_parabola: bool
            flag to perfrom parabola fit
        np: int
            number of points to use in parabola fit
        kwargs:
            passed to FixedNoiseGP

        Returns
        -------
        knobs, value, error (Tuple[torch.Tensor, torch.Tensor, torch.Tensor])
        alpha_lb, alpha_ub = self.interval(knobs, vector)
        """
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        alpha_lb, alpha_ub = self.interval(knobs, vector)

        @Wrapper(nk=1, lb=alpha_lb.flatten(), ub=alpha_ub.flatten())
        def target(alpha:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.objective(knobs + alpha*vector)

        target()

        if isinstance(beta, float):
            beta = beta*torch.ones(no_ucb, dtype=self.dtype, device=self.device)

        if len(beta) != no_ucb:
            raise ValueError('beta length mismatch')

        x = torch.zeros((1, 1))
        X = torch.stack([target.forward(alpha) for alpha in x])

        y = torch.clone(value).flatten()
        s = torch.clone(error).flatten()

        Y = ((y - y.mean())/y.std()).nan_to_num()
        S = (s/y.std()).nan_to_num()

        gp = FixedNoiseGP(X, Y.reshape(-1, 1), S.reshape(-1, 1)**2, **kwargs)
        ll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(ll)

        bounds = torch.tensor([[0.0], [1.0]], dtype=self.dtype, device=self.device)

        for i in range(no_ei + no_ucb):
            best = torch.min(Y)
            af = ExpectedImprovement(gp, best, maximize=False) if i < no_ei else UpperConfidenceBound(gp, beta[i - no_ei], maximize=False)
            candidate, _ = optimize_acqf(af, bounds = bounds, num_restarts = nr, q = 1, raw_samples = rs)
            candidate = candidate.flatten()
            value, error = target(candidate)
            x = torch.cat([x, target.inverse(candidate).reshape(-1, 1)])
            X = torch.cat([X, candidate.reshape(-1, 1)])
            y = torch.cat([y, value.flatten()])
            s = torch.cat([s, error.flatten()])
            self.append(knobs + x[-1]*vector, y[-1], s[-1])
            Y = ((y - y.mean())/y.std()).nan_to_num()
            S = (s/y.std()).nan_to_num()
            gp = FixedNoiseGP(X, Y.reshape(-1, 1), S.reshape(-1, 1)**2, **kwargs)
            ll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(ll)

        alpha = x.flatten()
        knobs = torch.stack([knobs + a*vector for a in alpha])
        value = y
        error = s

        index = alpha.argsort()
        alpha = alpha[index] - alpha[value.argmin()]
        knobs = knobs[index]
        value = value[index]
        error = error[index]

        index = value.argmin()

        if use_parabola:
            index = range(index - np, index + np + 1)
            return self.parabola(vector, alpha[index], knobs[index], value[index], error[index], sample=False)

        return knobs[index], value[index], error[index]


    def fit_rcds(self, knobs:torch, matrix:torch.Tensor, *,
                 minimize:Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]=None,
                 termination:bool=True, verbose:bool=False, pause:float=0.0, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        RCDS minimization.

        Parameters
        ----------
        knobs: torch.Tensor
            initial knobs
        matrix: torch.Tensor
            initial search directions
        minimize:Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            minimizer function
        termination: bool
            flag to use knobs and value termination
        verbose: bool
            verbose flag
        pause: float
            pause after each iteration
        kwargs:
            passed to minimize

        Returns
        -------
        fraction, knobs, value, error, matrix (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])

        """
        minimize = self.minimize_parabola if minimize is None else minimize

        sf = torch.clone(self.sf)

        knobs = torch.clone(knobs)
        if not torch.allclose(knobs, self.to_cube(torch.clone(knobs))):
            print(f'warning: initial knobs are not in the unit cube, values are adjusted')
            knobs = self.to_cube(knobs)

        matrix = torch.clone(matrix)
        if not torch.linalg.matrix_rank(matrix) == self.nk:
            print(f'warning: initial matrix is not full rank, changed to identity')
            matrix = torch.eye(self.nk, dtype=self.dtype, device=self.device)

        k = 0

        xk = torch.zeros((self.ni + 1, *knobs.shape), dtype=self.dtype, device=self.device)
        fk = torch.zeros(self.ni + 1, dtype=self.dtype, device=self.device)
        sk = torch.zeros(self.ni + 1, dtype=self.dtype, device=self.device)
        mk = torch.zeros((self.ni + 1, *matrix.shape), dtype=self.dtype, device=self.device)

        xk[k] = torch.clone(knobs)
        fk[k], sk[k] = self.objective(xk[k])
        mk[k] = torch.clone(matrix)

        self.append(xk[k], fk[k], sk[k])

        if torch.isnan(fk[k]):
            print(f'exit: objective value is nan')
            return sf, xk[:k+1], fk[:k+1], sk[:k+1], mk[:k+1]

        if verbose:
            print(f'-----------------------------')
            print(f'Iteration {k}:')
            print(f'-----------------------------')
            print(f'delta: {sf.cpu().numpy()}')
            print(f'knobs: {self.objective.inverse(xk[k]).cpu().numpy()}')
            print(f'value: {fk[k].cpu().numpy()}')
            print(f'error: {sk[k].cpu().numpy()}')
            print(f'count: {self.eval_count}')
            print()

        i = 0
        n = self.nk

        xi = torch.zeros((n + 1, *knobs.shape), dtype=self.dtype, device=self.device)
        fi = torch.zeros(n + 1, dtype=self.dtype, device=self.device)
        si = torch.zeros(n + 1, dtype=self.dtype, device=self.device)

        xi[i] = torch.clone(xk[k])
        fi[i] = torch.clone(fk[k])
        si[i] = torch.clone(sk[k])

        while k != self.ni:

            while i != n:
                xi[i + 1], fi[i + 1], si[i + 1] = minimize(sf, xi[i], fi[i], si[i], matrix[i], **kwargs)
                if torch.isnan(fi[i + 1]):
                    print(f'exit: objective value is nan')
                    return sf, xk[:k + 1], fk[:k + 1], sk[:k + 1], mk[:k + 1]
                i += 1

            f1, s1 = fk[k], sk[k]
            f2, s2 = fi[n], si[n]
            adjust = self.to_cube(2.0*xi[n] - xk[k])
            f3, s3 = self.objective(adjust)
            self.append(adjust, f3, s3)

            if torch.isnan(f3):
                print(f'exit: objective value is nan')
                return sf, xk[:k + 1], fk[:k + 1], sk[:k + 1], mk[:k + 1]

            delta = -fi.diff()
            m = delta.argmax()
            delta = delta[m]

            v = xi[n] - xk[k]
            v = v/v.norm()

            if f3 >= f1 or (f1 - 2*f2 + f3)*(f1 - f2 - delta)**2 >= 0.5*delta*(f1 - f3)**2 or torch.isnan(v.sum()):
                xk[k + 1] = xi[n] if f2 < f3 else adjust
                fk[k + 1] = f2 if f2 < f3 else f3
                sk[k + 1] = s2 if f2 < f3 else s3
            else:
                xk[k + 1], fk[k + 1], sk[k + 1] = minimize(sf, xi[n], fi[n], si[n], v, **kwargs)
                if torch.isnan(fk[k + 1]):
                    print(f'exit: objective value is nan')
                    return sf, xk[:k + 1], fk[:k + 1], sk[:k + 1], mk[:k + 1]
                matrix = torch.cat((matrix[:m], matrix[m + 1:], v.reshape(1, -1)))

            xi[0] = xk[k + 1]
            fi[0] = fk[k + 1]
            mk[k + 1] = torch.clone(matrix)

            if sf != self.sf_min:
                sf *= self.dr
            if sf < self.sf_min:
                sf = torch.clone(self.sf_min)
                print(f'warning: minimum step fraction {sf} is reached at iteration {k + 1}')

            if verbose:
                print(f'-----------------------------')
                print(f'Iteration {k + 1}:')
                print(f'-----------------------------')
                print(f'delta: {sf.cpu().numpy()}')
                print(f'knobs: {self.objective.inverse(xk[k + 1]).cpu().numpy()}')
                print(f'value: {fk[k + 1].cpu().numpy()}')
                print(f'error: {sk[k + 1].cpu().numpy()}')
                print(f'count: {self.eval_count}')
                print()

            if any((xk[k + 1] - 0.0).abs() < self.epsilon) or any((xk[k + 1] - 1.0).abs() < self.epsilon):
                print(f'warning: value on box encounted at iteration {k + 1}')

            if termination and self.termination_knobs(xk[k + 1], xk[k]):
                print(f'exit: triggered knobs termination at iteration {k + 1}')
                k += 1
                break

            if termination and self.termination_value(fk[k + 1], fk[k], sk[k + 1], sk[k]):
                print(f'exit: triggered value termination at iteration {k + 1}')
                k += 1
                break

            if k + 1 == self.ni:
                print(f'warning: maximum number of iterations {k + 1} is reached')

            k += 1
            i = 0
            self.iter_count = k
            sleep(pause)

        return sf, xk[:k+1], fk[:k+1], sk[:k+1], mk[:k+1]


    def adjust_cube(self, *,
               data:torch.Tensor=None,
               extend:bool=False,
               factor:float=5.0,
               center_estimator:Callable[[torch.Tensor], torch.Tensor]=median,
               spread_estimator:Callable[[torch.Tensor], torch.Tensor]=biweight_midvariance) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust search box using cached knobs dispersion.

        Note, returned bounds are in original cube, use to redefine objective with new values

        Parameters
        ----------
        data: torch.Tensor
            knobs data
        extend: bool
            flag to allow cube size increase
        factor: bool
            spread multiplication factor
        center_estimator: Callable[[torch.Tensor], torch.Tensor]
            center estimator
        spread_estimator: Callable[[torch.Tensor], torch.Tensor]
            spread estimator

        Returns
        -------
        lb, ub (Tuple[torch.Tensor, torch.Tensor])

        """
        if data is None:
            data = torch.stack([self.inverse(knobs) for knobs in torch.tensor(self.cache_knobs, dtype=self.dtype, device=self.device)])

        if not isinstance(data, torch.Tensor):
            try:
                data = torch.tensor(data, dtype=self.dtype, device=self.device)
            except TypeError:
                raise TypeError(f'failed to convert data to torch')

        data = data.T

        center = center_estimator(data)
        spread = self.epsilon + spread_estimator(data).sqrt()

        lb = center - factor*spread
        ub = center + factor*spread

        if not extend:
            lb[lb < self.lb] = self.lb[lb < self.lb]
            ub[ub > self.ub] = self.ub[ub > self.ub]

        return lb, ub


    def fit_scipy(self, knobs:torch.Tensor, method:str='powell', **kwargs):
        """
        Scipy minimization wrapper for bounded methods.

        Parameters
        ----------
        knobs: torch.Tensor
            initial knobs
        method: str
            minimization method ('powell', 'Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr')
        kwargs:
            passed to minimize

        Returns
        -------
        minimize result

        """
        def target(knobs:numpy.array, *, dtype:torch.dtype=self.dtype, device:torch.device=self.device) -> numpy.array:
            knobs = torch.tensor(knobs, dtype=dtype, device=device)
            value, _ = self.objective(knobs)
            return value.cpu().numpy()

        if method not in ['powell', 'Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']:
            raise ValueError(f'unsupported method {method}')

        return minimize(
            target,
            knobs.cpu().numpy(),
            bounds=numpy.array([numpy.zeros(self.nk), numpy.ones(self.nk)]).T,
            method=method,
            tol=self.otol.cpu().numpy(),
            **kwargs
        )


def main():
    pass

if __name__ == '__main__':
    main()