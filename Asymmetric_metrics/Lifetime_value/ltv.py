import numpy as np

def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    residual = (y_pred - y_true).astype(float)
    loss = np.where(residual > 0, 10 * residual ** 2, residual ** 2)
    error = np.mean(loss)
    return error