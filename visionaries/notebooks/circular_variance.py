import numpy as np
import matplotlib.pyplot as plt


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
    z = np.multiply(responses, np.exp(2 * np.pi * 1j * angles / 180))
    return 1 - np.abs(sum(z)) / sum(responses)



def get_null_cv_samples(obs_responses, n_samples=10000, null_dist='boot'):
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
    obs_responses: array-like, (n_samples, )
        The observed responses.

    n_samples: int
        How many null samples to draw.

    null_dist: str
        Which null distribution to use.
        Must be one of ['boot', 'rand'].
    """

    if null_dist == 'boot':
        sample_fun = sample_rand_cv_boot
    elif null_dist == 'rand':
        sample_fun = sample_rand_cv

    return np.array([sample_fun(obs_responses) for _ in range(n_samples)])


def sample_rand_cv(obs_responses):
    """
    Sample random angles from uniform distribution then compute
    circ var with observed responses.

    Parameters
    -----------
    obs_responses: array-like, (n_samples, )
        The observed responses.
    """
    rand_angles = np.random.uniform(0, 360, size=len(obs_responses))
    return circ_var(rand_angles, obs_responses)


def sample_rand_cv_boot(obs_responses):
    """
    Sample random angles from uniform distribution, resample observed responses
    from empirical distribution then compute cirv var.

    Parameters
    -----------
    obs_responses: array-like, (n_samples, )
        The observed responses.
    """
    rand_angles = np.random.uniform(0, 360, size=len(obs_responses))
    boot_responses = np.random.choice(obs_responses, replace=True, size=len(obs_responses))

    return circ_var(rand_angles, boot_responses)


def plot_test(angles, responses):
    """
    Runs test and plots histogram of results.
    """
    obs_cv = circ_var(angles, responses)
    null_samples = get_null_cv_samples(obs_responses=responses)
    pval = np.mean(null_samples < obs_cv)

    plt.hist(null_samples, color='black', bins=50)
    plt.axvline(obs_cv, color='red',
                label='obs cv {:1.3f} (p = {:1.3f})'.format(obs_cv, pval))
    plt.xlabel('circ variance')
    plt.legend()


def run_sim():
    """
    Simulation showing one example of a reponse with tuning and one example with no tuning
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
