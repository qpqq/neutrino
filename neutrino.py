import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import RBFInterpolator

import catalogs

# TODO презентацию с описанием работы и итоговыми картинками

# TODO распределение галактик в каждом каталоге на небесном сфере
# TODO гистограммы по звездной величине и по числу объектов в каждом каталоге
# TODO сделать те картинки которые выделялись отдельно
# TODO погуглить с шестиугольниками

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


def normal_pdf_logx_hist(n_particles):
    # TODO поделить на (z + 1)

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


def gauss(glon, glat, dgl, n_grid, fwhm, offset=10):
    # TODO make bugfix
    # TODO вывод объектов на картинке в файл (+ название каталога) (сортировка по яркости)
    # TODO шрифт, подписи (оси с единицами измерения и палитра), название
    # TODO разбить сферу на равные пиксели (подумать о сетке координат) (а лучше найти решение)

    from scipy.ndimage import gaussian_filter

    glon_min = glon - dgl / 2 - offset
    glon_max = glon + dgl / 2 + offset
    glat_min = glat - dgl / 2 - offset
    glat_max = glat + dgl / 2 + offset

    offset_grid = int(n_grid / dgl * offset)
    offset_grid += offset_grid % 2
    n_grid += offset_grid

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

    z = np.zeros(shape=(n_grid + 1, n_grid + 1))
    z[yi, xi] = data['BAR'].to_numpy()

    fwhm *= n_grid / dgl  # from degrees to pixels
    sigma = fwhm / 2.355  # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    z = gaussian_filter(z, sigma=sigma)

    offset_grid //= 2
    # x = x[offset_grid:-offset_grid - 1]
    # y = y[offset_grid:-offset_grid - 1]
    # z = z[offset_grid:-offset_grid - 1, offset_grid:-offset_grid - 1]

    return x, y, z


def gauss_graph():
    import matplotlib.colors as colors

    fig = plt.figure(figsize=(19.2, 10.8))

    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    glon = 90
    glat = 60
    dgl = 20
    n_grid = 100
    fwhm = 1.5
    x, y, z = gauss(glon, glat, dgl, n_grid, fwhm)
    pc = ax.pcolormesh(x, y, z, norm=colors.LogNorm(vmin=10 ** -8), cmap='jet')

    fig.colorbar(pc)

    plt.tight_layout()
    plt.show()


gauss_graph()
