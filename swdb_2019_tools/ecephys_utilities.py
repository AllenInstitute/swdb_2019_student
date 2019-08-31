
import pandas as pd


def spike_times_to_dataframe(spike_times):
    """ Convert a dictionary of spike times to a pandas dataframe of spike times

    Parameters
    ==========
    spike_times : dict
        Keys are unit ids, values are arrays of spike times

    Returns
    =======
    pandas.DataFrame :
        Index is unit ids, "spike_times" column contain arrays of spike times for each unit

    Usage
    =====
    >>> spike_times_to_dataframe({100: [10, 11, 12], 200: [12, 13, 14, 15]})
                spike_times
    unit_id                  
    100          [10, 11, 12]
    200      [12, 13, 14, 15]


    """
    
    unit_ids = []
    spike_times_out = []
    
    for unit_id, unit_spike_times in spike_times.items():
        unit_ids.append(unit_id)
        spike_times_out.append(unit_spike_times)
        
    return pd.DataFrame({
        "spike_times": spike_times_out
    }, index=pd.Index(data=unit_ids, name="unit_id"))