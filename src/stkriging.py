"""Spatio-temporal universal kriging with a separable Gaussian covariance.

Covariance model for two space-time points (s_i, t_i) and (s_j, t_j):

    C_ij = sigma2 * exp(-dlon^2/a^2 - dlat^2/b^2 - dt^2/c^2) + tau2 * 1{i = j}

with a linear mean  Z(s, t) = b0 + b1*lat + b2*lon + b3*t.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import norm


# --------------------------------------------------------------------------
# Design matrix and covariance
# --------------------------------------------------------------------------
def design_matrix(lon, lat, day):
    """Columns: [1, lat, lon, day]."""
    lon, lat, day = np.asarray(lon), np.asarray(lat), np.asarray(day)
    return np.column_stack([np.ones(len(lon)), lat, lon, day])


def squared_differences(lon, lat, day):
    """Pairwise squared differences in longitude, latitude and time."""
    dlon2 = (lon[:, None] - lon[None, :]) ** 2
    dlat2 = (lat[:, None] - lat[None, :]) ** 2
    dt2 = (day[:, None] - day[None, :]) ** 2
    return dlon2, dlat2, dt2


def covariance_matrix(dlon2, dlat2, dt2, theta):
    """Covariance matrix (signal + nugget on the diagonal)."""
    sigma2, a, b, c, tau2 = theta
    C = sigma2 * np.exp(-dlon2 / a**2 - dlat2 / b**2 - dt2 / c**2)
    np.fill_diagonal(C, C.diagonal() + tau2)
    return C


# --------------------------------------------------------------------------
# Maximum likelihood estimation
# --------------------------------------------------------------------------
def neg_log_likelihood(log_params, Z, X, dlon2, dlat2, dt2):
    """Profile negative Gaussian log-likelihood (trend profiled out by GLS).

    Parameters are optimised on the log scale to keep them positive.
    """
    theta = np.exp(log_params)
    n = len(Z)
    C = covariance_matrix(dlon2, dlat2, dt2, theta)
    try:
        L, low = cho_factor(C)
        Cinv_X = cho_solve((L, low), X)
        Cinv_Z = cho_solve((L, low), Z)
        beta_gls = np.linalg.solve(X.T @ Cinv_X, X.T @ Cinv_Z)
        resid = Z - X @ beta_gls
        Cinv_res = cho_solve((L, low), resid)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return 0.5 * (n * np.log(2.0 * np.pi) + log_det + resid @ Cinv_res)
    except (np.linalg.LinAlgError, ValueError):
        return 1e10


def gls_coefficients(Z, X, dlon2, dlat2, dt2, theta):
    """GLS trend coefficients for a given covariance parameter vector."""
    C = covariance_matrix(dlon2, dlat2, dt2, theta)
    L, low = cho_factor(C)
    return np.linalg.solve(X.T @ cho_solve((L, low), X),
                           X.T @ cho_solve((L, low), Z))


# --------------------------------------------------------------------------
# Universal kriging (nearest-neighbour approximation)
# --------------------------------------------------------------------------
def scaled_distance(lon1, lat1, day1, lon2, lat2, day2, theta):
    """Distance in units of the covariance length-scales (a, b, c)."""
    _, a, b, c, _ = theta
    return np.sqrt((lon1 - lon2) ** 2 / a**2
                   + (lat1 - lat2) ** 2 / b**2
                   + (day1 - day2) ** 2 / c**2)


def krige_point(g_lon, g_lat, g_day, lon, lat, day, Z, X, theta,
                k=200, beta_fallback=None):
    """Universal kriging prediction and kriging variance at one location.

    Uses the k nearest observations in length-scale-scaled space-time
    distance. Returns (prediction, variance). The variance is that of the
    latent field (it excludes the nugget tau2); add tau2 for a prediction
    interval on a new *observation*.
    """
    sigma2, a, b, c, tau2 = theta
    x0 = np.array([1.0, g_lat, g_lon, g_day])

    d = scaled_distance(lon, lat, day, g_lon, g_lat, g_day, theta)
    nn = np.argsort(d)[:k]
    lon_n, lat_n, day_n, Z_n, X_n = lon[nn], lat[nn], day[nn], Z[nn], X[nn]

    C_nn = covariance_matrix(*squared_differences(lon_n, lat_n, day_n), theta)
    c0 = sigma2 * np.exp(-(lon_n - g_lon) ** 2 / a**2
                         - (lat_n - g_lat) ** 2 / b**2
                         - (day_n - g_day) ** 2 / c**2)
    try:
        L, low = cho_factor(C_nn)
        Cinv_X = cho_solve((L, low), X_n)
        Cinv_Z = cho_solve((L, low), Z_n)
        beta_local = np.linalg.solve(X_n.T @ Cinv_X, X_n.T @ Cinv_Z)
        resid = Z_n - X_n @ beta_local
        pred = x0 @ beta_local + c0 @ cho_solve((L, low), resid)

        Cinv_c0 = cho_solve((L, low), c0)
        delta = x0 - X_n.T @ Cinv_c0
        var = (sigma2 - c0 @ Cinv_c0
               + delta @ np.linalg.inv(X_n.T @ Cinv_X) @ delta)
        return pred, var
    except (np.linalg.LinAlgError, ValueError):
        if beta_fallback is None:
            raise
        return x0 @ beta_fallback, sigma2 + tau2


# --------------------------------------------------------------------------
# Moran's I
# --------------------------------------------------------------------------
def morans_i(values, lon, lat, power=1.0):
    """Moran's I with row-standardised inverse-distance weights w = 1/d**power.

    Returns (I, expected I, z-score, two-sided p-value) using the
    normality-assumption variance. Locations must be distinct.
    """
    coords = np.column_stack([lon, lat])
    dist = cdist(coords, coords)
    if np.any(dist[~np.eye(len(dist), dtype=bool)] == 0):
        raise ValueError("Coincident locations give infinite inverse-distance "
                         "weights; aggregate to one value per location first.")
    with np.errstate(divide="ignore"):
        W = 1.0 / dist**power
    np.fill_diagonal(W, 0.0)
    W = W / W.sum(axis=1, keepdims=True)

    n = len(values)
    dev = values - values.mean()
    S0 = W.sum()
    I = n * (dev @ W @ dev) / ((dev @ dev) * S0)
    E_I = -1.0 / (n - 1)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)
    var_I = (n**2 * S1 - n * S2 + 3 * S0**2) / (S0**2 * (n**2 - 1)) - E_I**2
    z = (I - E_I) / np.sqrt(var_I)
    return I, E_I, z, 2 * (1 - norm.cdf(abs(z)))
