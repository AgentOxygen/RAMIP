import xarray as xr
from utils import find_wrong_datasets, get_indiv_check_msg
from datetime import timedelta
import numpy as np
from colorama import Fore, Style
from utils import get_filename
import collections
from test_timestep import get_timedelta

def find_gaps(ds: xr.Dataset, expected_delta: np.timedelta64):
    # If there are any gaps that are greater than the expected timedelta, return them as a list of tuples
    # Each tuple should be (start_index, start_time, end_time)
    # Where start_index is the index right before the timeskip and start_time is the first time  
    # that is skipped, and end_time is the last time that is skipped 

    time_diffs = ds.time.to_index().diff().to_series()
    gaps = []
    inds = np.where(time_diffs > expected_delta)[0]
    for ind in inds:
        start_time = ds.time[ind-1].values
        end_time = ds.time[ind].values
        gaps.append((ind, start_time, end_time))

    return gaps

def check_time_gaps(ds: xr.Dataset, verbose = False):
    timedelta = get_timedelta(ds)
    
    gaps = find_gaps(ds, timedelta)

    if verbose and len(gaps) > 0:
        print(Fore.CYAN + f"Time Gaps Check Err Output: " + Style.RESET_ALL)
        if len(gaps) > 10:
            print(f"The time values for {get_filename(ds)} dataset have gaps that are greater than the expected timedelta of {timedelta}. Here are the first 10 time gaps: ")
            gaps = gaps[:10]
        else:
            print(f"The time values for {get_filename(ds)} dataset have gaps that are greater than the expected timedelta of {timedelta}. Here are the time gaps: ")
        for gap in gaps:
            print(f"{gap[1]} -> {gap[2]} (index {gap[0]-1} -> {gap[0]})")
        print()

    return len(gaps) == 0

def test_time_gaps(datasets: list, verbose = False, checks = None) -> str:
    wrong_datasets = find_wrong_datasets(datasets, check_time_gaps, verbose)
    msgs = ["There exist time gaps in the datasets.", 
            "There are no time gaps in the datasets."]
    return get_indiv_check_msg(wrong_datasets, "Time Gaps Check", msgs, checks, len(datasets))

