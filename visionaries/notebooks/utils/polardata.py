"""
Useful functions for wrangling with polar data
"""

import random
import numpy as np
import matplotlib.pyplot as plt

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

def get_avg_polar(thetas, rs):
  """
  Given multiple polar data, get the average polar.
  Input:
  thetas[i] and rs[i] correspond to a single cell's polar data.
  That is... cell i likes orientation thetas[i][j] with a magnitude of rs[i][j]
  
  Output:
  the average polar data.
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
  avg_theta = []
  avg_r = []
  for theta, rs in theta_to_rs.items():
      avg_theta.append(theta)
      avg_r.append(np.mean(rs))
  return avg_theta, avg_r

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

def get_new_dsi(responses_df):
    pref_dir = responses_df['mean_dff'].argmax()
    non_pref_dir = (pref_dir + 180) % 360
    pref_dir_val = responses_df.loc[pref_dir].mean_dff
    non_pref_dir_val = responses_df.loc[non_pref_dir].mean_dff
    new_dsi = (pref_dir_val-non_pref_dir_val)/(pref_dir_val+non_pref_dir_val)
    return new_dsi


def circ_var(angles, responses):
    """
    Computes the circular variance of a set of responses.

    See http://tips.vhlab.org/data-analysis/measures-of-orientation-and-direction-selectivity.


    Parameters
    ----------
    angles: array-like, (n_samples, )
        The angles in degrees.

    responses: array-like, (n_samples, )
        The responses.
    """
    responses = np.array(responses)
    angles = np.array(angles)

    z = np.multiply(responses, np.exp(2 * np.pi * 1j * angles / 180))
    return 1 - np.abs(sum(z)) / sum(responses)


def get_pval(angles, responses, n_samples=1000,
             angle_dist='fixed', response_dist='perm'):

    obs_cv = circ_var(angles, responses)
    null_samples = get_null_cv_samples(angles=angles,
                                       responses=responses,
                                       n_samples=n_samples,
                                       angle_dist=angle_dist,
                                       response_dist=response_dist)
    return np.mean(null_samples < obs_cv)


def get_null_cv_samples(angles, responses, n_samples=1000,
                        angle_dist='fixed', response_dist='perm'):
    """
    Samples from the null distribution of the circular variance.

    obs_cv = circ_var(angles, responses)
    null_samples = get_null_cv_samples(obs_responses=responses)
    pval = np.mean(null_samples < obs_cv)

    plt.hist(rand_cvs, color='black', bins=50);
    plt.axvline(obs_cv, color='red',
                label='obs cv {:1.3f} (p = {:1.3f})'.format(obs_cv, pval))
    plt.xlabel('circ variance')
    plt.legend()


    Parameters
    -----------
    responses: array-like, (n_samples, )
        The observed angles.

    responses: array-like, (n_samples, )
        The observed responses.

    n_samples: int
        How many null samples to draw.

    angle_dist: str
        How to sample angles for null distribution. Must be one of ['fixed', 'boot', 'rand']. 'fixed' uses the observed angles (respones must be random). 'boot' resamples angles from observed distribution. 'rand' samples angles from uniform distribution.

    response_dist: str
            How to sample responses for null distribution. Must be one of ['fixed', 'boot', 'perm']. 'fixed' uses the observed responsees (angles must be random). 'boot' resamples responses from observed distribution. 'perm' permutes the responses.

    Output
    ------
    samples: array-like, (n_samples, )
        The null samples.

    """

    assert angle_dist in ['fixed', 'boot', 'rand']
    assert response_dist in ['fixed', 'boot', 'perm']
    assert not all([angle_dist == 'fixed', response_dist == 'fixed'])

    # n_samples = len(responses)

    def sample():
        if angle_dist == 'fixed':
            ang = angles
        if angle_dist == 'rand':
            ang = np.random.uniform(0, 360, size=n_samples)
        elif angle_dist == 'boot':
            ang = np.random.choice(angles, replace=True, size=n_samples)

        if response_dist == 'fixed':
            resp = responses
        elif response_dist == 'boot':
            resp = np.random.choice(responses, replace=True, size=n_samples)
        elif response_dist == 'perm':
            resp = np.random.permutation(responses)

        return circ_var(ang, resp)

    return np.array([sample() for _ in range(n_samples)])


def plot_test(angles, responses, n_samples=1000,
              angle_dist='rand', response_dist='boot'):
    """
    Runs test and plots histogram of results.
    """
    obs_cv = circ_var(angles, responses)
    null_samples = get_null_cv_samples(angles=angles,
                                       responses=responses,
                                       n_samples=n_samples,
                                       angle_dist=angle_dist,
                                       response_dist=response_dist)
    pval = np.mean(null_samples < obs_cv)

    plt.hist(null_samples, color='black', bins=50)
    plt.axvline(obs_cv, color='red',
                label='obs cv {:1.3f} (p = {:1.3f})'.format(obs_cv, pval))
    plt.xlabel('circ variance')
    plt.legend()


def run_sim():
    """
    Simulation showing one example of a reponse with tuning and one example with no tuning.
    """
    n = 100
    angles = np.random.uniform(0, 360, size=n)
    scale = abs(np.sin(np.deg2rad(angles)))
    responses_pref = np.random.exponential(scale=scale, size=n)
    responses_no_pref = np.random.exponential(size=n)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_test(angles, responses_pref)
    plt.title('true preference (should reject null)')

    plt.subplot(1, 2, 2)
    plot_test(angles, responses_no_pref)
    plt.title('no preference (should not reject null)')
