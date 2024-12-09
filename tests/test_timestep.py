import xarray as xr
from utils import find_different_datasets, get_consensus_check_msg, get_filename
from datetime import timedelta
import numpy as np
from colorama import Fore, Style
from utils import get_filename
import collections
import pandas as pd 

def get_timedelta(ds: xr.Dataset):
    ''' 
    For a dataset, return the timedelta that should be between each timestep 
    For example, if it's daily data, return a string that represents 1 day 

    Here is where you can learn more about how frequencies are encoded in strings:
    https://pandas.pydata.org/docs/dev/user_guide/timeseries.html#timeseries-offset-aliases
    and https://business-science.github.io/pytimetk/guides/03_pandas_frequency.html 

    This method also uses M_ and Y_ to represent monthly and yearly data that is not standard.
    '''

    freq = xr.infer_freq(ds.time.to_index())

    if freq is not None: 
        return freq 

    # The frequency is not standard, so we will do our best guess to infer it :( 

    # Check if it's yearly-ish 
    times = ds.time.to_index().to_datetimeindex().to_period("Y")
    time_diffs = times.diff().to_series()
    freq = time_diffs.mode()[0].n

    if freq != 0: 
        return "Y_"
    
    # Check if it's monthly-ish 
    times = ds.time.to_index().to_datetimeindex().to_period("M")
    time_diffs = times.diff().to_series()
    freq = time_diffs.mode()[0].n
    
    if freq != 0: 
        return "M_"

    return None


def check_timestep(ds1: xr.Dataset, ds2: xr.Dataset, verbose = False):
    timestep_1 = get_timedelta(ds1)
    timestep_2 = get_timedelta(ds2)

    if verbose and timestep_1 != timestep_2:
        print(Fore.CYAN + f"Timestep Err Output: ")
        print(f"Comparing majority opinion {get_filename(ds1)} with {get_filename(ds2)}")
        print(f"The first has timestep {timestep_1} and the second has timestep {timestep_2}" + Style.RESET_ALL)
        if(timestep_1 is None or timestep_2 is None):
            print(Fore.CYAN + "WARNING: A timestep of None means that a timestep for the dataset could not be inferred. This means the timestamps are somehow irregular." + Style.RESET_ALL)
        print()

    return timestep_1 == timestep_2


def test_timestep(datasets: list, verbose = False, checks = None) -> str:
    different_datasets = find_different_datasets(datasets, check_timestep, verbose)
    msgs = ["Timesteps are not equivalent across all datasets.", 
            "Timesteps are equivalent across all datasets."]
    return get_consensus_check_msg(different_datasets, "Timestep Check", msgs, checks, len(datasets))