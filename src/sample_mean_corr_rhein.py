"""Implement sample analysis methods."""
from datetime import datetime
from functools import reduce
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas


def my_mean(data_sample) -> float:
    """Implement a function to find the mean of an input List."""
    # TODO: 1.1 Implement me.
    return 0.


def my_std(data_sample) -> float:
    """Implement a function to find the standard deviation of a sample in a List."""
    # TODO: 1.2 Implement me.
    return 0.


def auto_corr(x: np.ndarray) -> np.ndarray:
    """Impement a function to compute the autocorrelation of x.
    
    Args:
        x (np.ndarray): Normalized input signal array of shape (signal_length,).

    Returns:
        np.ndarray: Autocorrelation of input signal of shape (signal_length*2 - 1,).
    """
    # TODO: 2.1 Implement me.
    # TODO: 2.2 Check your implementation via nox -s test.
    return np.zeros_like(x)


if __name__ == "__main__":
    rhein = pandas.read_csv("./data/pegel.tab", sep=" 	")
    levels = np.array([int(pegel.split(" ")[0]) for pegel in rhein["Pegel"]])

    timestamps = [ts[:-4] for ts in rhein["Zeit"]]
    datetime_list = []
    for ts in timestamps:
        ts_date, ts_time = ts.split(",")
        day, month, year = ts_date.split(".")
        hour, minute = ts_time.split(":")
        datetime_list.append(datetime(int(year), int(month), int(day)))

    before_2000 = [
        level
        for level, timepoint in zip(levels, datetime_list)
        if timepoint < datetime(2000, 1, 1)
    ]
    after_2000 = [
        level
        for level, timepoint in zip(levels, datetime_list)
        if timepoint > datetime(2000, 1, 1)
    ]


    # TODO: 1.3 Compute the mean and standard deviation before 2000.
    # TODO: 1.4 Compute the mean and standatd deviation after 2000.

    #----------------------------------------------------------------------------------------------#
    # TODO: 2.3 Normalize the data of the Rhine level measurements since 2000.
    # TODO: 2.4 Compute and plot the autocorrelation.

    # TODO: 2.5 Create a random signal.
    # TODO: 2.6 Normalize the random signal.
    # TODO: 2.7 Compute the autocorrelation.
    # TODO: 2.8. Plot both autocorrelations and compare the results.
