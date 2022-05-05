"""
Statistics module.
Compute statistics (center/spread) for 1D and 2D data.
Standardize and rescale 1D and 2D data.

"""

import torch


def mean(data:torch.Tensor) -> torch.Tensor:
    """
    Compute the mean for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data

    Returns
    -------
    mean (torch.Tensor)

    """
    return data.mean(-1)


def variance(data:torch.Tensor) -> torch.Tensor:
    """
    Compute the variance for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data

    Returns
    -------
    variance (torch.Tensor)

    """
    return data.var(-1)


def weighted_mean(data:torch.Tensor, weight:torch.Tensor=None) -> torch.Tensor:
    """
    Compute the weighted mean for given input data and weight.

    Parameters
    ----------
    data: torch.Tensor
        input data
    weight: torch.Tensor
        input weight

    Returns
    -------
    weighted mean (torch.Tensor)

    """
    if weight is None:
        return mean(data)

    return 1.0/weight.sum(-1)*(weight*data).sum(-1).squeeze(0)


def weighted_variance(data:torch.Tensor, weight:torch.Tensor=None, *,
                      center:torch.Tensor=None) -> torch.Tensor:
    """
    Compute the weighted variance for given input data and weight.

    Parameters
    ----------
    data: torch.Tensor
        input data
    weight: torch.Tensor
        input weight
    center: torch.Tensor
        precomputed weighted mean

    Returns
    -------
    weighted variance (torch.Tensor)

    """
    if weight is None:
        return variance(data)

    center = weighted_mean(data, weight) if center is None else center
    return weight.sum(-1)/((weight.sum(-1))**2 - (weight**2).sum(-1))*(weight*(data - center.reshape(-1, 1))**2).sum(-1).squeeze(0)


def trimmed_mean(data:torch.Tensor, *,
                 min_fraction:float=0.1, max_fraction:float=0.1) -> torch.Tensor:
    """
    Compute the trimmed mean for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data
    min_fraction: float
        fraction of min values to drop
    max_fraction: float
        fraction of max values to drop

    Returns
    -------
    trimmed mean (torch.Tensor)

    """
    *_, size = data.shape
    min_drop, max_drop = int(min_fraction*size), int(max_fraction*size)
    return mean(data.sort().values.reshape(len(data.shape), -1)[:, min_drop:-max_drop].squeeze(0))


def trimmed_variance(data:torch.Tensor, *,
                     min_fraction:float=0.1, max_fraction:float=0.1) -> torch.Tensor:
    """
    Compute the trimmed variance for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data
    min_fraction: float
        fraction of min values to drop
    max_fraction: float
        fraction of max values to drop

    Returns
    -------
    trimmed variance (torch.Tensor)

    """
    *_, size = data.shape
    min_drop, max_drop = int(min_fraction*size), int(max_fraction*size)
    return variance(data.sort().values.reshape(len(data.shape), -1)[:, min_drop:-max_drop].squeeze(0))


def winsorized_mean(data:torch.Tensor, *,
                    min_fraction:float=0.1, max_fraction:float=0.1) -> torch.Tensor:
    """
    Compute the winsorized mean for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data
    min_fraction: float
        fraction of min values to replace
    max_fraction: float
        fraction of max values to replace

    Returns
    -------
    winsorized mean (torch.Tensor)

    """
    *_, size = data.shape
    min_drop, max_drop = int(min_fraction*size), int(max_fraction*size)
    data = data.sort().values.reshape(len(data.shape), -1)
    min_value, max_value = data[:, +min_drop], data[:, -max_drop - 1]
    data[:, :+min_drop] = min_value.reshape(len(data), -1)
    data[:, -max_drop:] = max_value.reshape(len(data), -1)
    return mean(data.squeeze(0))


def winsorized_variance(data:torch.Tensor, *,
                        min_fraction:float=0.1, max_fraction:float=0.1) -> torch.Tensor:
    """
    Compute the winsorized variance for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data
    min_fraction: float
        fraction of min values to replace
    max_fraction: float
        fraction of max values to replace

    Returns
    -------
    winsorized variance (torch.Tensor)

    """
    *_, size = data.shape
    min_drop, max_drop = int(min_fraction*size), int(max_fraction*size)
    data = data.sort().values.reshape(len(data.shape), -1)
    min_value, max_value = data[:, +min_drop], data[:, -max_drop - 1]
    data[:, :+min_drop] = min_value.reshape(len(data), -1)
    data[:, -max_drop:] = max_value.reshape(len(data), -1)
    return variance(data.squeeze(0))


def median(data:torch.Tensor) -> torch.Tensor:
    """
    Compute the median for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data

    Returns
    -------
    median (torch.Tensor)

    """
    return data.median(-1).values


def median_deviation(data:torch.Tensor, *,
                     center:torch.Tensor=None) -> torch.Tensor:
    """
    Compute the median absolute deviation from the median for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data
    center: torch.Tensor
        precomputed median

    Returns
    -------
    median deviation (torch.Tensor)

    """
    center = median(data) if center is None else center
    return median((data - center.reshape(-1, 1)).abs().squeeze(0))


def biweight_midvariance(data:torch.Tensor, scale:torch.Tensor=10.0, *,
                         center:torch.Tensor=None, deviation:torch.Tensor=None) -> torch.Tensor:
    """
    Compute the biweight midvariance for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data
    scale: torch.Tensor
        scaling parameter
    center: torch.Tensor
        precomputed median
    deviation: torch.Tensor
        precomputed median absolute deviation

    Returns
    -------
    biweight midvariance (torch.Tensor)

    """
    *_, size = data.shape
    center = median(data) if center is None else center
    deviation = median_deviation(data, center=center) if deviation is None else deviation
    weight = 1.0 - ((data - center.reshape(-1, 1))/(scale*deviation.reshape(-1, 1)))**2
    weight[weight < 0.0] = 0.0
    return (size*torch.sum(weight**4*(data - center.reshape(-1, 1))**2, -1)/torch.sum(weight*(5.0*weight - 4.0), -1)**2).squeeze(0)


def quantile(data:torch.Tensor, fraction:torch.Tensor) -> torch.Tensor:
    """
    Compute the quantile for given input data and list of fractions.

    Parameters
    ----------
    data: torch.Tensor
        input data
    fraction: torch.Tensor
        list of fractions

    Returns
    -------
    quantile (torch.Tensor)

    """
    return torch.quantile(data, fraction, dim=-1).swapaxes(0, -1)


def quartiles(data:torch.Tensor) -> torch.Tensor:
    """
    Compute (0.25, 0.50, 0.75) quartiles for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data

    Returns
    -------
    quartiles (torch.Tensor)

    """
    return quantile(data, torch.tensor([0.25, 0.50, 0.75], dtype=data.dtype, device=data.device))


def interquartile_range(data:torch.Tensor) -> torch.Tensor:
    """
    Compute interquartile range for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data

    Returns
    -------
    interquartile range (torch.Tensor)

    """
    q_min, _, q_max = quartiles(data).swapaxes(0, -1)
    return (q_max - q_min)


def whiskers(data:torch.Tensor, *,
            factor:float=1.5) -> torch.Tensor:
    """
    Compute whiskers for given input data.

    Parameters
    ----------
    data: torch.Tensor
        input data

    Returns
    -------
    whiskers (torch.Tensor)

    """
    q_min, _, q_max = quartiles(data).swapaxes(0, -1)
    iqr = (q_max - q_min)
    return torch.stack([q_min - factor*iqr, q_max + factor*iqr])


def standardize(data:torch.Tensor, *,
                center_estimator=mean, spread_estimator=variance) -> torch.Tensor:
    """
    Standardize input 1D or 2D data.

    Note, default center/spread are mean/variance

    Parameters
    ----------
    data: torch.Tensor
        input 1D or 2D data
    center_estimator: callable
        center estimator
    spread_estimator: callable
        spread estimator

    Returns
    -------
    standardized data (torch.Tensor)

    """
    center = center_estimator(data).reshape(-1, 1)
    spread = spread_estimator(data).sqrt().reshape(-1, 1)
    return ((data - center)/spread).squeeze(0)


def rescale(data:torch.Tensor, *,
            range_min:float=None, range_max:float=None,
            scale_min:float=None, scale_max:float=None) -> torch.Tensor:
    """
    Rescale input 1D or 2D data.

    Note, default scaled interval is (-1, +1), input data min/max are different for each row if None

    Parameters
    ----------
    data: torch.Tensor
        input 1D or 2D data
    range_min: float
        input data min
    range_max: float
        input data max
    scale_min: float
        scaled data min
    scale_max: float
        scaled data max

    Returns
    -------
    scaled data (torch.Tensor)

    """
    range_min = data.min(-1).values.reshape(-1, 1) if range_min is None else range_min
    range_max = data.max(-1).values.reshape(-1, 1) if range_max is None else range_max

    scale_min = -1.0 if scale_min is None else scale_min
    scale_max = +1.0 if scale_max is None else scale_max

    return (data*(scale_max - scale_min)/(range_max - range_min) + (range_max*scale_min - range_min*scale_max)/(range_max - range_min)).squeeze(0)


def main():
    pass

if __name__ == '__main__':
    main()