import xarray as xr
from utils import find_different_datasets, get_consensus_check_msg, get_filename


def check_calendar(ds1: xr.Dataset, ds2: xr.Dataset, verbose = False):
    if 'calendar' not in ds1.time.encoding or 'calendar' not in ds2.time.encoding:
        if 'calendar' not in ds1.time.encoding and 'calendar' not in ds2.time.encoding:
            return True
        else:
            no_calendar_ds = get_filename(ds1) if 'calendar' not in ds1.time.encoding else get_filename(ds2)
            if verbose:
                print(f"Dataset {no_calendar_ds} has no calendar attribute.")
            return False

    return ds1.time.encoding['calendar'] == ds2.time.encoding['calendar']


def test_calendar(datasets: list, verbose = False, checks = None):  
    different_datasets = find_different_datasets(datasets, check_calendar, verbose)
    msgs = ["Time coordinates do not use the same calendar across all datasets.", 
            "Time coordinates use the same calendar across all datasets."]
    return get_consensus_check_msg(different_datasets, "Calendar Check", msgs, checks, len(datasets))

