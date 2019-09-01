"""
Useful plotting functions
""" 
import matplotlib.pyplot as plt
import numpy as np

def polar_plot(ax, theta, r, color='black', linewidth = 2):
    """
    - theta - in radians
    - r - [0,1.0]. same size as theta
    - color - E.g. 'black'. The line color
    - linewidth - thickness of polar plot line.
    """
    # Need to re-add the first point so the polygon closes.
    theta = np.append(theta, theta[0])
    r = np.append(r, r[0])

    p = ax.plot(theta, r, color=color, ls='-', linewidth=linewidth)
    #p[0].MarkerSize = 10
    # So enclosed region is grey instead of blank / white.
    #ax.fill(theta,r,'0.75')

    # Because default is 0 pointing to east, we want 0 at north.
    ax.set_theta_zero_location('N')
    # Default is ccw, we want cw
    ax.set_theta_direction(-1)

    ax.set_xticks(np.radians([0,45,90,135,180,225,270,315]))
    # How often to show the 0 ... 1.0 gradation in r values.
    # Have to set y ticks, or the 
    ax.set_yticks([0,1.0])
    ax.get_yaxis().set_visible(False)
    # If you want to turn on / off the gray scaling circles and r grids
    ax.grid(False)
    
    # rmax has to be set after plotting. See https://stackoverflow.com/questions/54653423/matplotlib-set-rmax-and-set-rticks-not-working
    ax.set_rmax(1.0)

def polar_plot_population(ax, bold_theta, bold_r, grey_thetas, grey_rs):
  """
  Polar plot (bold_theta, bold_r) in dark black. You probably want to pass in the average here.
  Then, plot multiple polar plots for each of the grey_thetas and grey_rs.
  You probably want to pass in sample populations that make up the average here so the viewer can get a sense
  of the distribution.
  """
  polar_plot(ax, bold_theta, bold_r, color='black', linewidth=4)
  for i in range(len(grey_thetas)):
      polar_plot(ax, grey_thetas[i], grey_rs[i], color='grey', linewidth=2)