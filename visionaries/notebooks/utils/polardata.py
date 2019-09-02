"""
Useful functions for wrangling with polar data
"""

import random

def cell_df_to_median_polar(cell_df):
    """
    Input:
    - cell_df (DataFrame) - each row is one cell. columns = [angle, magnitude].
    Output:
    - list<(angle in degree, median intensity for this angle)>
    """
    grouped_df = cell_df.groupby('angle').median()
    result = []
    for index, row in grouped_df.iterrows():
        result.append((index, row['magnitude']))
    return result

def get_mock_polardata(experiment_id, cell_specimen_id):
  """
  Generate mock polar plot data
  """

  return [
      (0, random.uniform(0, 1)),
      (45, random.uniform(0, 1)),
      (90, random.uniform(0, 1)),
      (135, random.uniform(0, 1)),
      (180, random.uniform(0, 1)),
      (225, random.uniform(0, 1)),
      (270, random.uniform(0, 1))
  ]