import numpy as np


def crps(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean CRPS over columns of a draw-by-column array."""
    count, columns = draws.shape
    if count == 0 or columns == 0:
        return 0.0
    accuracy = np.abs(draws - truth).mean(axis=0)
    if count == 1:
        return float(accuracy.mean())
    ordered = np.sort(draws, axis=0)
    ranks = np.arange(count, dtype=np.float64)[:, None]
    spread = ((2.0 * ranks - count + 1.0) * ordered).sum(axis=0)
    return float((accuracy - spread / (count * (count - 1.0))).mean())


def mae(draws: np.ndarray, truth: np.ndarray) -> float:
    """Return mean absolute error over draws and predicted quantities."""
    count, columns = draws.shape
    return float(np.abs(draws - truth).mean()) if count and columns else 0.0


def coverage_gap(draws: np.ndarray, truth: np.ndarray, *, level: float) -> float:
    """Return empirical minus nominal central-interval coverage."""
    count, columns = draws.shape
    if columns == 0:
        return 0.0
    if count == 0:
        return -level
    low, high = np.quantile(draws, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], axis=0)
    return float(((low <= truth) & (truth <= high)).mean()) - level
