"""
Helpful functions for accessing data
"""
import pandas as pd
def get_dg_response_filter_from_saskia():
    """
    Returns a responsiveness df, with two cols:
      [
        'cell_specimen_id',
        'responsive': bool
      ]
    The source is from saskia's swdb_dg_table.csv dataset.
    """
    # Read data from file 'filename.csv' 
    # (in the same directory that your python process is based)
    # Control delimiters, rows, column names with read_csv (see later) 
    data = pd.read_csv("../data/swdb_dg_table.csv")
    return data[['cell_specimen_id', 'responsive_dg']].rename(
        columns = {'responsive_dg': 'responsive'},
        inplace = False)