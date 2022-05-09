import matplotlib.pylab as plt
import numpy as np
from scipy.stats import kde


def int_norm(xs, ys):
    dx = xs[1] - xs[0]
    I = np.sum(ys*dx)
    return (ys / I)


def sample_gaussian(xs, A, B, sigma):
    ys = A/(sigma*np.sqrt(2*np.pi)) * np.exp((-0.5*((xs-B)/sigma)**2))
    return ys


def project_observable(
        data, populations=None, x_range=None, n_points=1000, sigma=None):
    """projects MSM onto an order parameter
    
    Inputs
    ----------
    data : nd.array, shape=(n_states, )
        The value of the order parameter for each cluster center.
    populations : nd.array, shape=(n_states, )
        The population of each state from the MSM.
    x_range : array, shape=(2, ), default=None,
        The x-axis plot range. i.e. [1, 5].
    n_points : int, default=1000,
        Number of points to use for plotting data.
    sigma : float, default=None,
        The width to use for each gaussian. If none is supplied, defaults to
        1/20th of the `x_range`.
    
    Outputs
    ----------
    xs : nd.array, shape=(n_points, ),
        The x-axis values of the resultant projection.
    ys : nd.array, shape=(n_points, )
        The y-axis values of the resultant projection.
    """
    data_spread = data.max() - data.min()
    if populations is None:
        populations = np.ones(data.shape[0])/data.shape[0]
    if x_range is None:
        delta_0p1 = data_spread*0.1
        x_range = [data.min()-delta_0p1, data.max()+delta_0p1]
    if sigma is None:
        sigma = data_spread/20.
    range_spread = x_range[1] - x_range[0]
    xs = range_spread*(np.arange(n_points)/n_points) + x_range[0]
    ys = np.zeros(xs.shape[0])
    for n in np.arange(data.shape[0]):
        ys += sample_gaussian(xs, populations[n], data[n], sigma=sigma)
    return xs, ys


def project_2D(
        data, n_bins=50, extra_spread=0.3, eq_probs=None, color_palette=plt.cm.hot_r):
    x_spread = x.max() - x.min()
    x_add = x_spread*extra_spread
    y_spread = y.max() - y.min()
    y_add = y_spread*extra_spread
    color_palette = plt.cm.hot_r

    if eq_probs is None:
        eq_probs = np.zeros(shape=data[0].shape[0]) + 1
    k = kde.gaussian_kde(data.T, weights=eq_probs)
    xi, yi = np.mgrid[x.min()-x_add:x.max()+x_add:nbins*1j, y.min()-y_add:y.max()+y_add:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    z_scatter = k(np.vstack([x,y]))

    fig = plt.figure(figsize=(8,6))
    plt.pcolormesh(
        xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=color_palette)
    plt.colorbar()
    plt.contour(
        xi, yi, zi.reshape(xi.shape), linewidths=2, colors='black', antialiased=True)
    plt.scatter(
        x, y, c=z_scatter, cmap=color_palette, s=20, edgecolors='black', alpha=0.5,
        linewidths=1.0, rasterized=True)
    plt.scatter(
        [data[0,0]], [data[0,1]], s=80, color='green', linewidth=0.5, edgecolors='black', zorder=3,
        rasterized=True)
    return fig


def plot_fig(
        xs, ys, ax=None, label=None, color='black', bar=False, norm_ys=True, alpha=1.0, **kwargs):
    if norm_ys:
        ys = int_norm(xs, ys)
    if ax is None:
        pfig = plt.figure(figsize=(12,5))
        ax = plt.subplot(
            111, **kwargs) 
        for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
        ax.tick_params(direction='out', length=10, width=3, colors='black')
    else:
        pfig = ax.figure
    if bar:
        width = (xs[1] - xs[0])
    if label is None:
        if bar:
            plt.bar(xs, ys, color=color, width=width, edgecolor='black', alpha=alpha)
        else:
            plt.plot(xs, ys, color=color, linewidth=3)
    else:
        if bar:
            plt.bar(
                xs, ys, color=color, label=label, width=width,
                edgecolor='black', alpha=alpha)
        else:
            plt.plot(
                xs, ys, label=label,
                color=color, linewidth=3)
        plt.legend(loc=2, prop={'size': 18})
    return pfig, ax
