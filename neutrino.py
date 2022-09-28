import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import RBFInterpolator

import catalogs

# constants
EMin = 30 * 10 ** 3
EMax = 30 * 10 ** 6
EMean = 1 * 10 ** 6

bins_edges = np.linspace(np.log10(EMin), np.log10(EMax), 50)

mu = np.log10(EMean)
sigma = 0.6

binsNumber = 100

mSun = -26.74
auInPc = 206265
NSunAu = 7 * 10 ** 10


# TODO выбрать столбик, из него взять поток (число частиц) для каждой галактики (если надо можно домножить)
# TODO выбрать площадку на небе (10' x 10') (можно варьировать)
# TODO каждую галактику размазать гауссом (сигма 10", ширина гауссианы на половине максимума)


def normal_pdf_logx_hist(n_particles):
    n_distr = 10 ** 5
    distr = np.random.normal(mu, sigma, n_distr)

    hist = np.histogram(distr, bins_edges)[0].astype('float64')

    # duplicates array n times
    hist = np.tile(hist, (len(n_particles), 1))
    hist = (hist.T * n_particles).T
    hist /= n_distr

    return hist


def normal_pdf_logx_graph(n_particles):
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    hist = normal_pdf_logx_hist(n_particles)
    ax.stairs(hist, bins_edges, fill=False, color='tab:red')

    x = np.linspace(np.log10(EMin), np.log10(EMax), 10 ** 4)
    y = norm.pdf(x, loc=mu, scale=sigma) * n_particles * (bins_edges[1] - bins_edges[0])
    ax.plot(x, y, linestyle='dotted', color='tab:blue')

    ax.set_xlabel('$E$, eV', fontsize=20)
    ax.set_ylabel('$N$', fontsize=20)
    ax.set_xlim(np.log10(EMin), np.log10(EMax))
    ax.set_ylim(0)

    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)

    plt.tight_layout()
    plt.show()


def scaling(data):
    data['NEU'] = NSunAu * 10 ** (0.4 * (mSun - data['MAG']))


def slice_data(n_grid):
    glon0 = 90
    glat0 = 60
    dglon = 15
    dglat = 15

    glon_min = glon0 - dglon / 2
    glon_max = glon0 + dglon / 2
    glat_min = glat0 - dglat / 2
    glat_max = glat0 + dglat / 2

    data = catalogs.read_2mrsg()
    data = data.loc[(data['GLON'] > glon_min) &
                    (data['GLON'] < glon_max) &
                    (data['GLAT'] > glat_min) &
                    (data['GLAT'] < glat_max)]
    scaling(data)

    # extracts specific column
    data['BAR'] = normal_pdf_logx_hist(data['NEU'].to_numpy())[:, 25]

    x = np.linspace(glon_min, glon_max, n_grid + 1)
    y = np.linspace(glat_min, glat_max, n_grid + 1)
    xi = np.searchsorted(x, data['GLON']) - 1
    yi = np.searchsorted(y, data['GLAT']) - 1

    z = np.zeros(shape=(n_grid, n_grid))
    z[yi, xi] = data['BAR'].to_numpy()

    return x, y, z


def slice_data_interp(n_grid, epsilon):
    glon0 = 90
    glat0 = 60
    dglon = 15
    dglat = 15

    glon_min = glon0 - dglon / 2
    glon_max = glon0 + dglon / 2
    glat_min = glat0 - dglat / 2
    glat_max = glat0 + dglat / 2

    data = catalogs.read_2mrsg()
    data = data.loc[(data['GLON'] > glon_min) &
                    (data['GLON'] < glon_max) &
                    (data['GLAT'] > glat_min) &
                    (data['GLAT'] < glat_max)]
    scaling(data)

    # extracts specific column
    data['BAR'] = normal_pdf_logx_hist(data['NEU'].to_numpy())[:, 25]

    print(data)
    print()

    n_grid_j = complex(0, n_grid)
    xy_grid = np.mgrid[glon_min:glon_max:n_grid_j, glat_min:glat_max:n_grid_j]
    xy_grid_flat = xy_grid.reshape(2, -1).T  # array of grid points, shape (n_grid, 2)

    x = data['GLON'].to_numpy()
    y = data['GLAT'].to_numpy()
    z = data['BAR'].to_numpy()

    interp = RBFInterpolator(list(zip(x, y)), z, kernel='gaussian', epsilon=epsilon)
    z_flat = interp(xy_grid_flat)
    z_grid = z_flat.reshape(n_grid, n_grid)

    return xy_grid, z_grid


def make_heights_equal(fig, rect, ax1, ax2, pad):
    import mpl_toolkits.axes_grid1.axes_size as Size
    from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider

    # pad in inches

    h1, v1 = Size.AxesX(ax1), Size.AxesY(ax1)
    h2, v2 = Size.AxesX(ax2), Size.AxesY(ax2)

    pad_v = Size.Scaled(1)
    pad_h = Size.Fixed(pad)

    my_divider = HBoxDivider(fig, rect,
                             horizontal=[h1, pad_h, h2],
                             vertical=[v1, pad_v, v2])

    ax1.set_axes_locator(my_divider.new_locator(0))
    ax2.set_axes_locator(my_divider.new_locator(2))


def slice_graph():
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig = plt.figure(figsize=(19.2, 10.8))

    n_grid = 300
    epsilon = 10
    vmin = 0
    vmax = 5 * 10 ** -5
    cmap = 'jet'

    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')
    x, y, z = slice_data(n_grid)
    pc = ax2.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap=cmap)

    ax1 = fig.add_subplot(121)
    ax1.set_aspect('equal')
    xy, z = slice_data_interp(n_grid, epsilon)
    ax1.pcolormesh(*xy, z, vmin=vmin, vmax=vmax, cmap=cmap)

    make_heights_equal(fig, 111, ax1, ax2, pad=0.75)

    axins2 = inset_axes(ax2, width="5%", height="100%", loc='right', borderpad=-5)
    fig.colorbar(pc, cax=axins2)

    # plt.tight_layout()
    plt.show()


slice_graph()
