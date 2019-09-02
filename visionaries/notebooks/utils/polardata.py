"""
Useful functions for wrangling with polar data
"""

import random
import numpy as np

def cell_df_to_median_polar(cell_df):
    """
    Input:
    - cell_df (DataFrame). columns = [angle, magnitude]. A concatenation of multiple cells' preferences for each angle.
    Output:
    - list<(angle in degree, median intensity for this angle)>
    """
    grouped_df = cell_df.groupby('angle').median()
    result = []
    for angle, row in grouped_df.iterrows():
        result.append((angle, row['magnitude']))
    return result

def get_median_polar(thetas, rs):
  """
  Given multiple polar data, get the median polar.
  Input:
  thetas[i] and rs[i] correspond to a single cell's polar data.
  That is... cell i likes orientation thetas[i][j] with a magnitude of rs[i][j]
  
  Output:
  the median polar data.
  """
  theta_to_rs = {}
  for cell_i in range(len(thetas)):
      cell_thetas = thetas[cell_i]
      cell_rs = rs[cell_i]
      for ori_i in range(len(cell_thetas)):    
          theta = cell_thetas[ori_i]
          r = cell_rs[ori_i]
          if theta not in theta_to_rs:
              theta_to_rs[theta] = []
          theta_to_rs[theta].append(r)
  median_theta = []
  median_r = []
  for theta, rs in theta_to_rs.items():
      median_theta.append(theta)
      median_r.append(np.median(rs))
  return median_theta, median_r

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