import math
import numpy as np

from extended_network_model import STATES
from sklearn.metrics import mean_squared_error, mean_absolute_error


def detected_active_counts(model):
    counts = [] 
    for detected_active in STATES.detected: 
        counts.append(model.model.get_state_count(detected_active).values[1:])
    return sum(counts)

def model_rmse(model, y_true):
    infected_count = detected_active_counts(model)
    return math.sqrt(mean_squared_error(y_true, infected_count))


def model_mae(model, y_true):
    infected_count = detected_active_counts(model)
    return mean_absolute_error(y_true, infected_count)


def model_r_squared(model, y_true):
    infected_count = detected_active_counts(model)
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2) + np.finfo(np.float32).eps
    rss = np.sum((y_true - infected_count) ** 2)
    return -(1 - rss / tss)


return_func_zoo = {
    'rmse': model_rmse,
    'mae': model_mae,
    'r2': model_r_squared
}
