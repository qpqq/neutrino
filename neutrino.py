import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import catalogs

# TODO презентацию с описанием работы и итоговыми картинками

# TODO распределение галактик в каждом каталоге на небесном сфере
# TODO гистограммы по звездной величине и по числу объектов в каждом каталоге
# TODO сделать те картинки которые выделялись отдельно
# TODO погуглить с шестиугольниками (разбить сферу на равные пиксели (может не квадратные)) (а лучше найти решение)

# constants
EMin = 30 * 10 ** 3
EMax = 30 * 10 ** 6
EMean = 1 * 10 ** 6

binsNumber = 100
binsEdges = np.linspace(np.log10(EMin), np.log10(EMax), binsNumber)

mu = np.log10(EMean)
sigma = 0.6

mSun = -26.74
auInPc = 206265
NSunAu = 7 * 10 ** 10


def normal_pdf_logx_hist(n_particles):
    # TODO поделить на (z + 1)

    n_distr = 10 ** 5
    distr = np.random.normal(mu, sigma, n_distr)

    hist = np.histogram(distr, binsEdges)[0].astype('float64')

    if not np.isscalar(n_particles):
        hist = np.tile(hist, (len(n_particles), 1))  # duplicates array n times
    hist = (hist.T * n_particles).T
    hist /= n_distr

    return hist


def normal_pdf_logx_graph(n_particles):
    from scipy.stats import norm
    from matplotlib.ticker import AutoMinorLocator

    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    hist = normal_pdf_logx_hist(n_particles)
    ax.stairs(hist, binsEdges, fill=False, color='tab:red')

    x = np.linspace(np.log10(EMin), np.log10(EMax), 10 ** 4)
    y = norm.pdf(x, loc=mu, scale=sigma) * n_particles * (binsEdges[1] - binsEdges[0])
    ax.plot(x, y, linestyle='dotted', color='tab:blue')

    ax.set_xlabel('$E$, eV', fontsize=20)
    ax.set_ylabel('$N$', fontsize=20)
    ax.set_xlim(np.log10(EMin), np.log10(EMax))
    ax.set_ylim(0)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.offsetText.set_fontsize(18)

    ax.grid(True, linewidth=0.3)
    ax.grid(True, 'minor', linewidth=0.1)

    title = fr'Normal distribution {n_particles} particles, $\sigma = {sigma}$, $\mu = {mu}$'
    ax.set_title(title, fontsize=25, pad=15)

    fig.tight_layout()
    plt.show()


def scaling(data):
    data['NEU'] = NSunAu * 10 ** (0.4 * (mSun - data['MAG']))


def gauss(catalog, glon, glat, dgl, n_grid, fwhm):
    from scipy.ndimage import gaussian_filter

    offset = 5 * fwhm
    fwhm *= n_grid / dgl  # from degrees to pixels

    glon_min = glon - dgl / 2 - offset
    glon_max = glon + dgl / 2 + offset
    glat_min = glat - dgl / 2 - offset
    glat_max = glat + dgl / 2 + offset

    offset_grid = 2 * int(n_grid / dgl * offset)
    n_grid += offset_grid

    data = catalogs.read(catalog)
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

    # grid with min float values
    z = np.full(shape=(n_grid, n_grid), fill_value=sys.float_info.min)
    for i in range(len(xi)):
        z[yi[i], xi[i]] += data.iloc[i]['BAR']

    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    sigma_gauss = fwhm / 2.355
    z = gaussian_filter(z, sigma=sigma_gauss)

    offset_grid //= 2
    x = x[offset_grid:-offset_grid]
    y = y[offset_grid:-offset_grid]
    z = z[offset_grid:-offset_grid, offset_grid:-offset_grid]

    os.makedirs('gauss graphs', exist_ok=True)
    data = data.sort_values('BAR', ascending=False).reset_index(drop=True)
    data.to_csv(f'gauss graphs/{catalog}.csv', index=False, float_format='%.15f')

    return x, y, z


def gauss_graph(catalog):
    import matplotlib.colors as colors

    fig = plt.figure(figsize=(14, 10.5))

    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    glon = 90
    glat = 0
    dgl = 60
    n_grid = 100
    fwhm = 1.5
    x, y, z = gauss(catalog, glon, glat, dgl, n_grid, fwhm)

    vmin = 10 ** -8
    pc = ax.pcolormesh(x, y, z, norm=colors.LogNorm(vmin=vmin), cmap='jet')

    cbar = fig.colorbar(pc, pad=0.01)
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel(r'Neutrino flux, [s$^{-1}$ cm$^{-2}$]', fontsize=20)

    ax.set_xlabel('Galactic longitude', fontsize=20)
    ax.set_ylabel('Galactic latitude', fontsize=20)

    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.set_title(f'{catalogs.fullName[catalog]}, {n_grid}x{n_grid}, fwhm = {fwhm}', fontsize=25, pad=15)

    fig.tight_layout()
    fig.savefig(f'gauss graphs/{catalog}.png', dpi=120)
    plt.show()


gauss_graph('2mrsg')
