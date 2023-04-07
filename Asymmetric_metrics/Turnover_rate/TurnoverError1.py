import numpy as np

def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    residual = (y_pred - y_true).astype(float)
    error = np.where(residual < 0, 2 * residual ** 2, residual ** 2)
    return np.mean(error)
