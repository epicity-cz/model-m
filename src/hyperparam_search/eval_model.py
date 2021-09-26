import math
import numpy as np

from models.states import STATES
from sklearn.metrics import mean_squared_error, mean_absolute_error


def detected_active_counts(model, fit_column):
    counts = model.get_df()
    counts = counts[fit_column][1:].to_numpy()

    return counts


def model_rmse(model, y_true, fit_column='I_d'):
    infected_count = detected_active_counts(model, fit_column)
    return math.sqrt(mean_squared_error(y_true, infected_count))


def model_mae(model, y_true, fit_column='I_d'):
    infected_count = detected_active_counts(model, fit_column)
    return mean_absolute_error(y_true, infected_count)


def model_r_squared(model, y_true, fit_column='I_d'):
    infected_count = detected_active_counts(model, fit_column)
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2) + np.finfo(np.float32).eps
    rss = np.sum((y_true - infected_count) ** 2)
    return -(1 - rss / tss)


return_func_zoo = {
    'rmse': model_rmse,
    'mae': model_mae,
    'r2': model_r_squared
}
