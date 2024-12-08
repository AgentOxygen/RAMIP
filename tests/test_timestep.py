import xarray as xr
from utils import find_different_datasets, get_consensus_check_msg, get_filename
from datetime import timedelta
import numpy as np
from colorama import Fore, Style
from utils import get_filename
import collections

def get_timedelta(ds: xr.Dataset):
    # For a dataset, return the timedelta that should be between each timestep 
    # For example, if it's daily data, return 1 day

    # Find the most common timedelta between timesteps
    time_diffs = ds.time.to_index().diff().to_series()
    return time_diffs.mode()[0]


def check_timestep(ds1: xr.Dataset, ds2: xr.Dataset, verbose = False):
    timestep_1 = get_timedelta(ds1)
    timestep_2 = get_timedelta(ds2)

    if verbose and timestep_1 != timestep_2:
        print(Fore.CYAN + f"Timestep Err Output: ")
        print(f"Comparing majority opinion {get_filename(ds1)} with {get_filename(ds2)}")
        print(f"The first has timestep {timestep_1} and the second has timestep {timestep_2}" + Style.RESET_ALL)
        print()

    return timestep_1 == timestep_2


def test_timestep(datasets: list, verbose = False, checks = None) -> str:
    different_datasets = find_different_datasets(datasets, check_timestep, verbose)
    msgs = ["Timesteps are not equivalent across all datasets.", 
            "Timesteps are equivalent across all datasets."]
    return get_consensus_check_msg(different_datasets, "Timestep Check", msgs, checks, len(datasets))

